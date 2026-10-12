#!/usr/bin/env python3
"""Matched map sweeps and SAC budgets for FIFO, rule and learned content."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-finite-contact-matplotlib')
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.learning import Learner
from finite_contact.maps import configured_map, training_map_index
from finite_contact_train import append_json, checkpoint, evaluate, provenance, rollout
from finite_contact_multimap_content import weight_change
from finite_contact_multimap_report import audit_segments, physical_audit, rows

ROOT = Path(__file__).resolve().parent
GROUPS = {'A_fifo': ('A', 'fifo'), 'A_rule': ('A', 'rule'), 'AB_learned': ('AB', 'learned')}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seed_episode(seed, episode):
    value = (seed + episode * 10007) % 2 ** 32
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    return value


def common_state(movement, content, actor_rate, seed):
    """Transfer Adam/network state, discard observations from the old FIFO MDP."""
    if movement.get('format') != 'ir2-finite-contact-v1' or movement['stage'] != 'movement' or 'replay' not in movement:
        raise ValueError('A full movement checkpoint is required')
    if content.get('format') != 'ir2-finite-contact-v1' or content['learner']['content_updates'] < 1:
        raise ValueError('A trained communication checkpoint is required')
    if Config.from_checkpoint(movement['config']).to_dict() != Config.from_checkpoint(content['config']).to_dict():
        raise ValueError('Movement and head must use the same source configuration')
    a, b = movement['learner'], content['learner']
    for name in ('policy', 'critics', 'targets'):
        if weight_change(a, b, name) != 0:
            raise ValueError('The communication head must come from this frozen movement model')
    if not torch.equal(a['log_alpha'], b['log_alpha']) or a['updates'] != b['updates']:
        raise ValueError('Source temperature or learning counters differ')
    cfg = Config.from_checkpoint(movement['config'])
    cfg.seed = seed
    cfg.policy_learning_rate = actor_rate
    # A complete first rollout is collected before learning, rather than an online warmup.
    cfg.warmup = cfg.batch_size
    cfg.validate()
    learner = copy.deepcopy(a)
    for key in ('content', 'context_encoder', 'content_optimizer', 'content_updates'):
        learner[key] = copy.deepcopy(b[key])
    for group in learner['policy_optimizer']['param_groups']:
        group['lr'] = actor_rate
    return {'format': 'ir2-finite-contact-v1', 'stage': 'finetune', 'episode': 0,
            'config': cfg.to_dict(), 'learner': learner, 'best_validation': None,
            'training_wall_seconds': 0., 'reset': ['movement replay', 'success episodes', 'sampling RNG'],
            'retained': ['movement weights', 'Adam moments', 'temperature', 'trained head and frozen context']}


def prepare(root, args):
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new output directory')
    if not args.movement_state or not args.content_checkpoint:
        raise ValueError('Specify movement-state and content-checkpoint')
    movement = torch.load(args.movement_state, map_location='cpu')
    content = torch.load(args.content_checkpoint, map_location='cpu')
    saved = common_state(movement, content, args.policy_learning_rate, args.seed)
    cfg = Config.from_checkpoint(saved['config'])
    if (cfg.validation_maps, cfg.test_maps) != (20, 20) or cfg.map_family != 'simple_rooms_v1':
        raise ValueError('Use the existing simple-map family and 20/20 evaluation splits')
    cfg.train_maps = getattr(args, 'train_maps', 100)
    cfg.success_episode_limit = max(cfg.success_episode_limit, cfg.train_maps)
    cfg.compact_replay = getattr(args, 'compact_replay', False)
    cfg.validate()
    saved['config'] = cfg.to_dict()
    root.mkdir(parents=True, exist_ok=True)
    torch.save(saved, root / 'common_start.pth')
    plan = {'scope': 'Three-group matched adaptation pilot; one seed, no test policy execution.',
            'movement_source': str(args.movement_state.resolve()),
            'movement_sha256': sha256(args.movement_state),
            'head_source': str(args.content_checkpoint.resolve()), 'head_sha256': sha256(args.content_checkpoint),
            'common_sha256': sha256(root / 'common_start.pth'),
            'reference_root': str(args.reference_root.resolve()) if args.reference_root else None,
            'seed': cfg.seed, 'dataset_seed': cfg.map_seed, 'training_maps': cfg.train_maps,
            'sweeps': args.sweeps, 'updates_per_map': args.updates_per_map,
            'new_sac_updates_per_group': cfg.train_maps * args.sweeps * args.updates_per_map,
            'source_sac_updates': saved['learner']['updates'],
            'frozen_head_updates': saved['learner']['content_updates'],
            'actor_learning_rate': cfg.policy_learning_rate, 'critic_learning_rate': cfg.learning_rate,
            'max_hours_per_group': args.max_hours, 'groups': list(GROUPS),
            'status': 'prepared', 'replay_policy': 'All groups collect new, mode-consistent observations; no FIFO replay/SIL transfer.',
            'schedule': 'One complete sampled team rollout per map, then exactly K actual SAC updates.',
            'episode_rng': '(training_seed + episode * 10007) modulo 2**32'}
    (root / 'pipeline.json').write_text(json.dumps(plan, indent=2))


def group_config(common, group):
    cfg = Config.from_checkpoint(common['config'])
    cfg.variant, cfg.content = GROUPS[group]
    cfg.coupled = True
    cfg.validate()
    return cfg


def archive_after_checkpoint(directory, episode, updates):
    """Retain superseded work while restarting at the last committed full boundary."""
    filters = {'training.jsonl': lambda r: r['episode'] <= episode,
               'trajectories.jsonl': lambda r: r['episode'] <= episode,
               'communication.jsonl': lambda r: r['episode'] <= episode,
               'updates.jsonl': lambda r: r['updates'] <= updates}
    archive = directory / ('superseded_' + str(time.time_ns()))
    archive.mkdir()
    for name, retain in filters.items():
        path = directory / name
        if not path.exists():
            continue
        records = rows(path)
        shutil.move(str(path), str(archive / name))
        for record in records:
            if retain(record):
                append_json(path, record)


def train_group(root, plan, group, device, resume=None):
    common = torch.load(root / 'common_start.pth', map_location='cpu')
    cfg = group_config(common, group)
    directory = root / group
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ValueError('Group directory is nonempty; use its latest.pth with --resume')
    directory.mkdir(exist_ok=True)
    torch.set_num_threads(cfg.torch_threads)
    seed_episode(cfg.seed, 0)
    learner = Learner(cfg, device)
    state = torch.load(resume, map_location='cpu') if resume else common
    learner.load(state['learner'], optimizers=True)
    learner.content.requires_grad_(False).eval()
    learner.context_encoder.requires_grad_(False).eval()
    completed = 0
    previous_wall = 0.
    if resume:
        if Path(resume).resolve() != (directory / 'latest.pth').resolve() or state['config'] != cfg.to_dict():
            raise ValueError('Resume requires this group latest.pth and identical configuration')
        completed = state['episode']
        previous_wall = state['training_wall_seconds']
        learner.replay.extend(state['replay'])
        learner.success_episodes = state['success_episodes']
        archive_after_checkpoint(directory, completed, learner.updates)
        snapshot = directory / ('resume_' + str(time.time_ns()))
        snapshot.mkdir()
        provenance(snapshot, cfg)
    else:
        provenance(directory, cfg)
        checkpoint(directory / 'initial.pth', learner, cfg, 'finetune', 0, None, 0., full=False)
        checkpoint(directory / 'latest.pth', learner, cfg, 'finetune', 0, None, 0.)
    context = {'episode': completed, 'map_index': None}
    learner.on_update = lambda row: append_json(directory / 'updates.jsonl', {**context, **row})
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    should_stop = lambda: stop[0] or (directory / 'STOP').exists()
    started = time.monotonic()
    deadline = started + plan['max_hours_per_group'] * 3600
    status = {'status': 'training', 'group': group, 'completed_episodes': completed}
    def save_status():
        (directory / 'status.json').write_text(json.dumps(status, indent=2))
    save_status()
    try:
        for episode in range(completed + 1, cfg.train_maps * plan['sweeps'] + 1):
            if should_stop() or time.monotonic() >= deadline:
                raise RuntimeError('Stopped before the complete matched training budget')
            action_seed = seed_episode(cfg.seed, episode)
            index = training_map_index(cfg, episode)
            context.update(episode=episode, map_index=index)
            before = learner.updates
            metrics, env = rollout(learner, cfg, 'train', index, 'finetune', deadline=deadline,
                                   stop_requested=should_stop, episode_update_limit=0)
            if metrics['interrupted']:
                append_json(directory / 'interruptions.jsonl', {'episode': episode, **metrics})
                raise RuntimeError('Interrupted map; resume uses the last full saved boundary')
            assert learner.updates == before
            for _ in range(plan['updates_per_map']):
                if should_stop() or time.monotonic() >= deadline:
                    raise RuntimeError('Interrupted update block; last full saved boundary retained')
                if not learner.update():
                    raise RuntimeError('Insufficient replay; refusing to count skipped updates')
            assert learner.updates - before == plan['updates_per_map']
            metrics.update(episode=episode, action_seed=action_seed, online_sac_updates=0,
                           post_episode_sac_updates=learner.updates - before,
                           learner_updates=learner.updates, content_updates=learner.content_updates,
                           wall_seconds=previous_wall + time.monotonic() - started)
            append_json(directory / 'training.jsonl', metrics)
            append_json(directory / 'trajectories.jsonl', {'episode': episode, 'map': env.map_info,
                                                          'trajectory': env.trajectory})
            for event in env.radio.events:
                append_json(directory / 'communication.jsonl', {'episode': episode, **event})
            np.savez_compressed(directory / ('training_map_%03d.npz' % episode),
                                truth=env.truth, beliefs=np.stack(env.beliefs), positions=env.positions)
            completed = episode
            if episode % 10 == 0:
                checkpoint(directory / 'latest.pth', learner, cfg, 'finetune', completed, None,
                           previous_wall + time.monotonic() - started)
            if episode % 50 == 0:
                checkpoint(directory / ('at_episode_%d.pth' % episode), learner, cfg, 'finetune', completed,
                           None, previous_wall + time.monotonic() - started, full=False)
            status.update(completed_episodes=completed, sac_updates=learner.updates)
            save_status()
            print(json.dumps({'group': group, 'episode': episode, 'map': index,
                              'success': metrics['success'], 'updates': learner.updates}), flush=True)
        assert learner.updates == plan['source_sac_updates'] + plan['new_sac_updates_per_group']
        checkpoint(directory / 'latest.pth', learner, cfg, 'finetune', completed, None,
                   previous_wall + time.monotonic() - started)
        checkpoint(directory / 'final.pth', learner, cfg, 'finetune', completed, None,
                   previous_wall + time.monotonic() - started, full=False)
        status['status'] = 'evaluating_final'
        save_status()
        evaluation = directory / 'validation_final'
        if evaluation.exists() and any(evaluation.iterdir()):
            shutil.move(str(evaluation), str(directory / ('superseded_validation_' + str(time.time_ns()))))
        evaluation.mkdir(exist_ok=True)
        provenance(evaluation, cfg)
        results = evaluate(learner, cfg, 20, directory=evaluation, stop_requested=should_stop)
        if len(results) != 20:
            raise RuntimeError('Incomplete final evaluation')
        for record in results:
            append_json(evaluation / 'evaluation.jsonl', record)
        status.update(status='complete', validation_successes=sum(r['success'] for r in results),
                      final_sac_updates=learner.updates, content_updates=learner.content_updates)
        print(json.dumps(status), flush=True)
    except BaseException:
        status['status'] = 'interrupted_or_failed'
        raise
    finally:
        # Never replace a full boundary checkpoint with a partially executed map/update block.
        save_status()


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def summarize(records):
    return {'maps': len(records), 'successes': sum(r['success'] for r in records),
            'mean_capped_time': float(np.mean([r['time'] for r in records])),
            'mean_minimum_coverage': float(np.mean([min(r['coverage']) for r in records])),
            'mean_coverage_deficit': float(np.mean([r['coverage_deficit'] for r in records])),
            'failed_maps': [r['map']['index'] for r in records if not r['success']]}


def report(root, plan):
    common = torch.load(root / 'common_start.pth', map_location='cpu')
    assert sha256(root / 'common_start.pth') == plan['common_sha256']
    assert sha256(plan['movement_source']) == plan['movement_sha256']
    assert sha256(plan['head_source']) == plan['head_sha256']
    result = {'scope': plan['scope'], 'training_seed': plan['seed'], 'dataset_seed': plan['dataset_seed'],
              'new_sac_updates_per_group': plan['new_sac_updates_per_group'],
              'source_sac_updates': plan['source_sac_updates'],
              'frozen_head_updates': plan['frozen_head_updates'], 'groups': {},
              'test_policy_evaluations': 0,
              'limitations': ['One training seed and one complete adaptation sweep.',
                  'Inherited network/Adam state but fresh mode-consistent replay and success memory.',
                  'Training success counts refer to changing per-episode policies, not the fixed final on all training maps.',
                  'Prior frozen validation is reused only after checking common network identity and evaluation-relevant configuration.']}
    all_segments = 0
    for group in GROUPS:
        directory = root / group
        assert json.loads((directory / 'status.json').read_text())['status'] == 'complete'
        cfg = group_config(common, group)
        initial = torch.load(directory / 'initial.pth', map_location='cpu')
        final = torch.load(directory / 'latest.pth', map_location='cpu')
        assert identical(initial['learner'], common['learner']), 'Unequal initial network/Adam states'
        assert initial['config'] == final['config'] == cfg.to_dict()
        assert not initial.get('replay') and not initial.get('success_episodes')
        training = rows(directory / 'training.jsonl')
        updates = rows(directory / 'updates.jsonl')
        expected_episodes = cfg.train_maps * plan['sweeps']
        assert len(training) == final['episode'] == expected_episodes
        assert [r['episode'] for r in training] == list(range(1, expected_episodes + 1))
        assert [r['map']['index'] for r in training] == [training_map_index(cfg, e) for e in range(1, expected_episodes + 1)]
        assert all(not r['interrupted'] and r['online_sac_updates'] == 0 and
                   r['post_episode_sac_updates'] == plan['updates_per_map'] for r in training)
        before, after = initial['learner']['updates'], final['learner']['updates']
        assert len(updates) == after - before == plan['new_sac_updates_per_group']
        assert [r['updates'] for r in updates] == list(range(before + 1, after + 1))
        for episode in range(1, expected_episodes + 1):
            selected = [r for r in updates if r['episode'] == episode]
            assert len(selected) == plan['updates_per_map']
            assert all(r['map_index'] == training[episode - 1]['map']['index'] for r in selected)
        assert all(np.isfinite(v) for r in updates for v in r.values() if isinstance(v, (int, float)))
        changes = {name: weight_change(initial['learner'], final['learner'], name)
                   for name in ('policy', 'critics', 'targets', 'content', 'context_encoder')}
        assert all(changes[k] > 0 for k in ('policy', 'critics', 'targets'))
        assert changes['content'] == changes['context_encoder'] == 0
        assert final['learner']['content_updates'] == plan['frozen_head_updates']
        success = {}
        traces = rows(directory / 'trajectories.jsonl')
        assert len(traces) == len(training)
        segments = 0
        for r, trace in zip(training, traces):
            assert r['episode'] == trace['episode'] and r['map'] == trace['map']
            truth, _, info = configured_map(cfg, 'train', r['map']['index'])
            assert info == r['map']
            segments += audit_segments(truth, trace['trajectory'])
            with np.load(directory / ('training_map_%03d.npz' % r['episode'])) as data:
                np.testing.assert_array_equal(data['truth'], truth)
                free = truth == 255
                coverage = [float(np.count_nonzero((b == 255) & free) / free.sum()) for b in data['beliefs']]
                np.testing.assert_array_equal(coverage, r['coverage'])
            assert r['success'] == all(c >= cfg.completion for c in coverage)
            assert r['sent_bytes'] <= 2 * cfg.bytes_per_second * r['connected_seconds'] + 1e-6
            assert r['delivered_bytes'] <= r['sent_bytes'] and r['control_bytes'] <= r['sent_bytes']
            assert r['total_distance'] <= 2 * cfg.speed * r['time'] + 1e-6
            assert r['time'] <= cfg.horizon_seconds + 1e-6
            for a, b in zip(trace['trajectory'][:-1], trace['trajectory'][1:]):
                assert np.all(np.linalg.norm(np.array(b['positions']) - a['positions'], axis=1)
                              <= cfg.speed * (b['time'] - a['time']) + 1e-6)
            if r['success']:
                key = r['map']['layout_id']
                success[key] = min(success.get(key, float('inf')), r['time'])
        cache = final['success_episodes']
        assert len(cache) == len({e['map_key'] for e in cache})
        assert {e['map_key'] for e in cache} == set(success)
        assert all(e['duration'] == success[e['map_key']] for e in cache)
        development = rows(directory / 'validation_final/evaluation.jsonl')
        assert all(r['learner_updates'] == after and not r['interrupted'] for r in development)
        physical = physical_audit(directory / 'validation_final', cfg.to_dict())
        all_segments += segments + physical['movement_segments']
        group_result = {'training_episodes': len(training), 'distinct_training_maps': cfg.train_maps,
            'online_successful_training_episodes': sum(r['success'] for r in training),
            'new_sac_updates': after - before, 'cumulative_sac_updates': after,
            'weight_change_l2': changes, 'success_memory_maps': len(cache),
            'training_physical_audit': {'rollouts': len(training), 'movement_segments': segments,
                'wall_crossings': 0, 'coverage_speed_and_radio_caps_passed': True},
            'validation_final': {**summarize(development), 'physical_audit': physical}}
        if plan['reference_root']:
            reference = Path(plan['reference_root']) / ('validation_' + cfg.content)
            reference_cfg = Config.from_checkpoint(json.loads((reference / 'config.json').read_text())).to_dict()
            current_cfg = cfg.to_dict()
            for key in ('warmup', 'policy_learning_rate'):
                reference_cfg[key] = current_cfg[key]
            # A and AB share the exact rendezvous observation path; content selects the radio behavior.
            if reference_cfg['variant'] == 'A' and current_cfg['variant'] == 'AB':
                reference_cfg['variant'] = 'AB'
            assert reference_cfg == current_cfg, 'Evaluation-relevant baseline config differs'
            source_head = torch.load(plan['head_source'], map_location='cpu')['learner']
            assert all(weight_change(source_head, initial['learner'], k) == 0
                       for k in ('policy', 'critics', 'targets', 'content', 'context_encoder'))
            previous = rows(reference / 'evaluation.jsonl')
            assert [r['map'] for r in previous] == [r['map'] for r in development]
            group_result['validation_initial'] = summarize(previous)
            group_result['paired_outcomes'] = {
                'gained_success': sum(not a['success'] and b['success'] for a, b in zip(previous, development)),
                'lost_success': sum(a['success'] and not b['success'] for a, b in zip(previous, development)),
                'both_successful': sum(a['success'] and b['success'] for a, b in zip(previous, development)),
                'both_failed': sum(not a['success'] and not b['success'] for a, b in zip(previous, development))}
        result['groups'][group] = group_result
    result['combined_physical_audit'] = {'rollouts': len(GROUPS) * (cfg.train_maps * plan['sweeps'] + cfg.validation_maps),
                                         'movement_segments': all_segments, 'wall_crossings': 0}
    output = root / 'report'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    shutil.copy2(__file__, output / 'analysis_source.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = list(GROUPS)
    for ax, key, title in [(axes[0, 0], 'successes', 'Final: same 20 development maps'),
                            (axes[0, 1], 'mean_capped_time', 'Final mean time; failures = 256 s')]:
        values = [result['groups'][g]['validation_final'][key] for g in names]
        bars = ax.bar(names, values, color=['gray', 'tab:orange', 'tab:blue'])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value, '%.1f' % value if key != 'successes' else '%d/20' % value,
                    ha='center', va='bottom')
        ax.set(title=title, ylim=(0, 22 if key == 'successes' else 280))
    for group in names:
        updates = rows(root / group / 'updates.jsonl')
        axes[1, 0].plot(range(1, len(updates) + 1), [r['policy_loss'] for r in updates], lw=.5, label=group)
        records = rows(root / group / 'validation_final/evaluation.jsonl')
        axes[1, 1].plot(range(20), [min(r['coverage']) for r in records], 'o-', lw=.8, label=group)
    axes[1, 0].set(title='Actual Actor updates, matched %d-step budget' % plan['new_sac_updates_per_group'], xlabel='Additional SAC updates', ylabel='Actor loss')
    axes[1, 0].legend()
    axes[1, 1].set(title='Final minimum robot coverage', xlabel='Development map', ylabel='Coverage', ylim=(0, 1.02))
    axes[1, 1].axhline(.99, color='black', ls='--', lw=.5)
    axes[1, 1].legend()
    fig.tight_layout(); fig.savefig(output / 'matched_adaptation.png', dpi=160)
    fig.savefig(output / 'matched_adaptation.pdf'); plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('all', 'prepare', 'group', 'report'), default='all')
    p.add_argument('--movement-state', type=Path)
    p.add_argument('--content-checkpoint', type=Path)
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--group', choices=list(GROUPS))
    p.add_argument('--resume', type=Path)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--train-maps', type=int, default=100)
    p.add_argument('--compact-replay', action='store_true')
    p.add_argument('--sweeps', type=int, default=1)
    p.add_argument('--updates-per-map', type=int, default=20)
    p.add_argument('--policy-learning-rate', type=float, default=1e-5)
    p.add_argument('--max-hours', type=float, default=1.5)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = p.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/wyz/anaconda3/envs/ir2_train').resolve():
        p.error('Use ir2_train Python')
    if min(args.sweeps, args.updates_per_map, args.policy_learning_rate, args.max_hours) <= 0:
        p.error('Positive learning budgets required')
    if args.resume and args.stage != 'group':
        p.error('Resume one group from its latest.pth')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; no silent CPU fallback')
    root = args.output.resolve()
    if args.stage in ('all', 'prepare'):
        prepare(root, args)
    plan = json.loads((root / 'pipeline.json').read_text())
    assert sha256(root / 'common_start.pth') == plan['common_sha256']
    if args.stage == 'group':
        if not args.group:
            p.error('--stage group needs --group')
        train_group(root, plan, args.group, args.device, args.resume)
    elif args.stage == 'report':
        report(root, plan)
    elif args.stage == 'all':
        plan['status'] = 'running'
        (root / 'pipeline.json').write_text(json.dumps(plan, indent=2))
        children = []
        try:
            for group in GROUPS:
                log = (root / (group + '.log')).open('w')
                command = [sys.executable, str(Path(__file__).resolve()), '--stage', 'group',
                           '--output', str(root), '--group', group, '--device', args.device]
                children.append((group, subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))
            while any(child.poll() is None for _, child, _ in children):
                failed = [group for group, child, _ in children if child.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError('Group failures: ' + ', '.join(failed))
                time.sleep(1)
            failed = [group for group, child, _ in children if child.returncode]
            if failed:
                raise RuntimeError('Group failures: ' + ', '.join(failed))
            report(root, plan)
            plan['status'] = 'complete'
        except BaseException:
            for _, child, _ in children:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
            for _, child, _ in children:
                child.wait()
            plan['status'] = 'interrupted_or_failed'
            raise
        finally:
            for _, _, log in children:
                log.close()
            (root / 'pipeline.json').write_text(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()


