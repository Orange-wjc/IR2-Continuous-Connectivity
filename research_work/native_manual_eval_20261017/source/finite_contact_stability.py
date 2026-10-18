#!/usr/bin/env python3
"""Paired full-state continuation: pooled versus equal-map/tail success imitation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-finite-contact-matplotlib')
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.learning import Learner
from finite_contact.maps import configured_map, training_map_index
from finite_contact_train import append_json, checkpoint, evaluate, provenance, rollout
from finite_contact_multimap_joint import GROUPS, archive_after_checkpoint, identical, seed_episode, sha256, summarize
from finite_contact_multimap_content import weight_change
from finite_contact_multimap_report import audit_segments, rows
from finite_contact_multiseed import run_worker_queue

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('pooled', 'map_tail')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def recover_initialization(directory):
    """Archive startup interrupted before the first full checkpoint, without losing work."""
    directory = Path(directory)
    if not directory.exists() or not any(directory.iterdir()) or (directory/'latest.pth').exists():
        return None
    if any(path.exists() and rows(path) for path in
           (directory/'training.jsonl', directory/'updates.jsonl')):
        raise RuntimeError('Training records exist without a full checkpoint; explicit recovery audit required')
    archived = directory.with_name(directory.name + '_initialization_interrupted_' + str(time.time_ns()))
    directory.rename(archived)
    return archived


def evaluate_to(learner, cfg, directory, split, count, stop):
    if directory.exists():
        shutil.move(str(directory), str(directory.with_name(directory.name + '_superseded_' + str(time.time_ns()))))
    directory.mkdir()
    write(directory/'config.json', cfg.to_dict())
    results = evaluate(learner, cfg, count, split=split, directory=directory, stop_requested=stop)
    if len(results) != count:
        raise RuntimeError('Incomplete fixed-map monitoring')
    for record in results:
        append_json(directory/'evaluation.jsonl', record)
    return summarize(results)


def train_worker(root, condition, group, device, resume=False):
    plan = read(root/'plan.json')
    source = plan['sources'][group]
    directory = root/condition/group
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ValueError('Nonempty worker output; use --resume')
    directory.mkdir(parents=True, exist_ok=True)
    saved = torch.load(directory/'latest.pth' if resume else source['path'], map_location='cpu')
    cfg = Config.from_checkpoint(saved['config'])
    if resume and cfg.success_sampling != condition:
        raise ValueError('Wrong resume condition')
    cfg.success_sampling = condition
    cfg.validate()
    torch.set_num_threads(cfg.torch_threads)
    seed_episode(cfg.seed, source['episode'])
    learner = Learner(cfg, device)
    learner.load(saved['learner'], optimizers=True)
    learner.replay.extend(saved['replay'])
    learner.content_replay.extend(saved['content_replay'])
    learner.success_episodes = saved['success_episodes']
    learner.content.requires_grad_(False).eval()
    learner.context_encoder.requires_grad_(False).eval()
    completed = saved['episode'] - source['episode'] if resume else 0
    previous_wall = saved['training_wall_seconds'] if resume else 0.
    if not 0 <= completed <= plan['training_maps']:
        raise ValueError('Invalid complete-map resume boundary')
    assert learner.updates == source['sac_updates'] + completed * plan['updates_per_map']
    if resume:
        archive_after_checkpoint(directory, saved['episode'], learner.updates)
    else:
        provenance(directory, cfg)
        checkpoint(directory/'initial.pth', learner, cfg, 'finetune', source['episode'], None, 0., full=False)
        write(directory/'initial_cache.json', {'source_sha256': source['sha256'],
            'replay_transitions': len(learner.replay), 'content_replay': len(learner.content_replay),
            'success_episodes': [{'map_key': e['map_key'], 'duration': e['duration'],
                                  'transitions': len(e['transitions'])} for e in learner.success_episodes]})
        checkpoint(directory/'latest.pth', learner, cfg, 'finetune', source['episode'], None, 0.)
    del saved
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    should_stop = lambda: stop[0] or (directory/'STOP').exists()
    start = time.monotonic()
    deadline = start + plan['max_hours_per_worker'] * 3600
    status = {'status': 'training', 'condition': condition, 'group': group, 'completed_maps': completed}
    context = {}
    learner.on_update = lambda record: append_json(directory/'updates.jsonl', {**context, **record})
    try:
        if not resume:
            evaluate_to(learner, cfg, directory/'train_at_000', 'train', plan['monitor_maps'], should_stop)
        elif not (directory/'train_at_000/evaluation.jsonl').exists():
            if completed:
                raise RuntimeError('Missing initial diagnostic after training began')
            evaluate_to(learner, cfg, directory/'train_at_000', 'train', plan['monitor_maps'], should_stop)
        for local in range(completed + 1, plan['training_maps'] + 1):
            episode = source['episode'] + local
            if should_stop() or time.monotonic() >= deadline:
                raise RuntimeError('Stopped before fixed continuation budget')
            seed_episode(cfg.seed, episode)
            index = training_map_index(cfg, episode)
            context.update(episode=episode, map_index=index)
            before = learner.updates
            metric, env = rollout(learner, cfg, 'train', index, 'finetune', deadline=deadline,
                                   stop_requested=should_stop, episode_update_limit=0)
            if metric['interrupted']:
                append_json(directory/'interruptions.jsonl', {'episode': episode, **metric})
                raise RuntimeError('Interrupted map; retained full boundary must be used')
            assert learner.updates == before
            for _ in range(plan['updates_per_map']):
                if should_stop() or time.monotonic() >= deadline:
                    raise RuntimeError('Interrupted update block; retained full boundary must be used')
                if not learner.update():
                    raise RuntimeError('Skipped SAC update is not a completed update')
            assert learner.updates - before == plan['updates_per_map']
            metric.update(episode=episode, continuation_map=local, online_sac_updates=0,
                          post_episode_sac_updates=learner.updates-before, learner_updates=learner.updates,
                          wall_seconds=previous_wall+time.monotonic()-start)
            append_json(directory/'training.jsonl', metric)
            append_json(directory/'trajectories.jsonl', {'episode': episode, 'map': env.map_info,
                                                        'trajectory': env.trajectory})
            for event in env.radio.events:
                append_json(directory/'communication.jsonl', {'episode': episode, **event})
            np.savez_compressed(directory/('training_map_%03d.npz'%episode),
                                truth=env.truth, beliefs=np.stack(env.beliefs), positions=env.positions)
            completed = local
            if local % 10 == 0 or local == plan['training_maps']:
                checkpoint(directory/'latest.pth', learner, cfg, 'finetune', episode, None,
                           previous_wall+time.monotonic()-start)
            if local in plan['monitor_at']:
                checkpoint(directory/('at_map_%03d.pth'%local), learner, cfg, 'finetune', episode,
                           None, previous_wall+time.monotonic()-start, full=False)
                monitor = evaluate_to(learner, cfg, directory/('train_at_%03d'%local), 'train',
                                      plan['monitor_maps'], should_stop)
                print(json.dumps({'condition': condition, 'group': group, 'monitor_at': local,
                                  'train_monitor': monitor}), flush=True)
            status.update(completed_maps=local, sac_updates=learner.updates)
            write(directory/'status.json', status)
            print(json.dumps({'condition': condition, 'group': group, 'maps': local,
                              'success': metric['success'], 'updates': learner.updates}), flush=True)
        # A stop during monitoring is recoverable from the preceding full training boundary.
        for local in [0]+plan['monitor_at']:
            if not (directory/('train_at_%03d'%local)/'evaluation.jsonl').exists():
                saved_monitor = torch.load(directory/('at_map_%03d.pth'%local), map_location='cpu')
                monitor_learner = Learner(cfg, device)
                monitor_learner.load(saved_monitor['learner'])
                evaluate_to(monitor_learner, cfg, directory/('train_at_%03d'%local), 'train',
                            plan['monitor_maps'], should_stop)
                del monitor_learner, saved_monitor
        assert learner.updates == source['sac_updates'] + plan['training_maps'] * plan['updates_per_map']
        checkpoint(directory/'final.pth', learner, cfg, 'finetune', source['episode']+completed, None,
                   previous_wall+time.monotonic()-start, full=False)
        development = evaluate_to(learner, cfg, directory/'validation_final', 'validation',
                                  cfg.validation_maps, should_stop)
        status.update(status='complete', completed_maps=completed, validation=development)
    except BaseException:
        status['status'] = 'interrupted_or_failed'
        raise
    finally:
        write(directory/'status.json', status)


def audit_evaluation(directory, cfg, split, count):
    records = rows(directory/'evaluation.jsonl')
    traces = rows(directory/'evaluation_trajectories.jsonl')
    assert len(records) == len(traces) == count
    assert [r['map']['index'] for r in records] == list(range(count))
    segments = 0
    for record, trace in zip(records, traces):
        assert record['map'] == trace['map'] and record['map']['split'] == split
        data = np.load(directory/('evaluation_map_%d_repeat_0.npz'%record['map']['index']))
        segments += audit_record(record, trace, data['truth'], data['beliefs'], cfg)
    return {'rollouts': count, 'movement_segments': segments, 'wall_crossings': 0}


def audit_record(record, trace, truth, beliefs, cfg, geometry_details=False):
    assert not record['interrupted']
    assert hashlib.sha256(truth.tobytes()).hexdigest() == record['map']['sha256']
    free = truth == 255
    coverage = [float(np.count_nonzero((b==255)&free)/free.sum()) for b in beliefs]
    np.testing.assert_array_equal(coverage, record['coverage'])
    union = float(np.count_nonzero(((beliefs[0]==255)|(beliefs[1]==255))&free)/free.sum())
    assert union == record['union_coverage']
    assert record['success'] == all(c >= cfg.completion for c in coverage)
    assert 0 < record['time'] <= cfg.horizon_seconds+1e-6
    assert record['sent_bytes'] <= 2*cfg.bytes_per_second*record['connected_seconds']+1e-6
    assert record['delivered_bytes'] <= record['sent_bytes'] and record['control_bytes'] <= record['sent_bytes']
    assert record['total_distance'] <= 2*cfg.speed*record['time']+1e-6
    trajectory = trace['trajectory']
    for a, b in zip(trajectory[:-1], trajectory[1:]):
        elapsed = b['time'] - a['time']
        lengths = np.linalg.norm(np.array(b['positions'])-a['positions'],axis=1)
        assert elapsed > 0 and np.all(lengths <= cfg.speed*elapsed+1e-6)
    return audit_segments(truth, trajectory, return_details=geometry_details)


def report(root):
    plan = read(root/'plan.json')
    result = {'scope': plan['scope'], 'training_seed': plan['seed'], 'test_policy_evaluations': 0,
              'groups': {}, 'paired_conditions': {}, 'combined_physical_audit':
              {'rollouts': 0, 'movement_segments': 0, 'wall_crossings': 0}}
    total = result['combined_physical_audit']
    for group in GROUPS:
        src = plan['sources'][group]
        assert sha256(src['path']) == src['sha256']
        source = torch.load(src['path'], map_location='cpu')
        initial_control = None
        for condition in CONDITIONS:
            directory = root/condition/group
            assert read(directory/'status.json')['status'] == 'complete'
            initial = torch.load(directory/'initial.pth', map_location='cpu')
            final = torch.load(directory/'latest.pth', map_location='cpu')
            cfg = Config.from_checkpoint(final['config'])
            expected_cfg = Config.from_checkpoint(source['config']).to_dict()
            expected_cfg['success_sampling'] = condition
            assert initial['config'] == final['config'] == expected_cfg
            assert identical(initial['learner'], source['learner'])
            cache = read(directory/'initial_cache.json')
            assert cache['replay_transitions'] == len(source['replay'])
            assert cache['content_replay'] == len(source['content_replay'])
            assert cache['success_episodes'] == [{'map_key': e['map_key'], 'duration': e['duration'],
                'transitions': len(e['transitions'])} for e in source['success_episodes']]
            if initial_control is None:
                initial_control = initial['learner']
            else:
                assert identical(initial_control, initial['learner'])
            assert final['episode'] == src['episode'] + plan['training_maps']
            assert final['learner']['updates'] == src['sac_updates'] + plan['training_maps']*plan['updates_per_map']
            changes = {k: weight_change(initial['learner'], final['learner'], k)
                       for k in ('policy','critics','targets','content','context_encoder')}
            assert all(changes[k] > 0 for k in ('policy','critics','targets'))
            assert changes['content'] == changes['context_encoder'] == 0
            assert identical(final['learner']['content_optimizer'], initial['learner']['content_optimizer'])
            assert final['learner']['content_updates'] == src['head_updates']
            training, traces = rows(directory/'training.jsonl'), rows(directory/'trajectories.jsonl')
            updates = rows(directory/'updates.jsonl')
            assert len(training) == len(traces) == plan['training_maps']
            assert [r['episode'] for r in training] == list(range(src['episode']+1, final['episode']+1))
            assert {r['map']['index'] for r in training} == set(range(cfg.train_maps))
            assert [r['updates'] for r in updates] == list(range(src['sac_updates']+1,final['learner']['updates']+1))
            assert all(np.isfinite(v) for r in updates for v in r.values() if isinstance(v,(int,float)))
            segments = 0
            for record, trace in zip(training, traces):
                assert record['episode'] == trace['episode'] and record['map'] == trace['map']
                assert record['map']['split'] == 'train'
                assert record['online_sac_updates'] == 0 and record['post_episode_sac_updates'] == plan['updates_per_map']
                data = np.load(directory/('training_map_%03d.npz'%record['episode']))
                truth, _, info = configured_map(cfg, 'train', record['map']['index'])
                assert info == record['map'] and np.array_equal(truth, data['truth'])
                segments += audit_record(record, trace, truth, data['beliefs'], cfg)
            total['rollouts'] += len(training); total['movement_segments'] += segments
            train_monitor = {}
            for local in [0]+plan['monitor_at']:
                d = directory/('train_at_%03d'%local)
                audit = audit_evaluation(d, cfg, 'train', plan['monitor_maps'])
                for key in total: total[key] += audit[key]
                train_monitor[str(local)] = summarize(rows(d/'evaluation.jsonl'))
            audit = audit_evaluation(directory/'validation_final', cfg, 'validation', cfg.validation_maps)
            for key in total: total[key] += audit[key]
            stats = summarize(rows(directory/'validation_final/evaluation.jsonl'))
            if condition == 'map_tail':
                assert rows(root/'pooled'/group/'train_at_000/evaluation.jsonl') == rows(directory/'train_at_000/evaluation.jsonl')
            source_steps = [v['step'].item() for v in source['learner']['policy_optimizer']['state'].values()]
            final_steps = [v['step'].item() for v in final['learner']['policy_optimizer']['state'].values()]
            assert len(set(source_steps)) == len(set(final_steps)) == 1
            assert final_steps[0]-source_steps[0] == plan['training_maps']*plan['updates_per_map']
            result['groups'].setdefault(group,{})[condition] = {'validation_final':stats,
                'training_monitor':train_monitor, 'training_successes':sum(r['success'] for r in training),
                'new_sac_updates':final['learner']['updates']-src['sac_updates'],
                'cumulative_sac_updates':final['learner']['updates'], 'actor_adam_steps':final_steps[0],
                'frozen_head_updates':final['learner']['content_updates'],'weight_change_l2':changes}
            del final, initial
        a, b = [result['groups'][group][c]['validation_final'] for c in CONDITIONS]
        result['paired_conditions'][group] = {'success_gain':b['successes']-a['successes'],
            'time_saved_seconds':a['mean_capped_time']-b['mean_capped_time']}
        del source, initial_control
    output = root/'report'; output.mkdir(exist_ok=True)
    write(output/'summary.json', result)
    shutil.copy2(__file__, output/'analysis_source.py')
    plot_report(result, output)
    print(json.dumps(result, indent=2), flush=True)


def plot_report(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for group, ax in zip(GROUPS, axes):
        for condition in CONDITIONS:
            monitor=result['groups'][group][condition]['training_monitor']
            xs=sorted(int(x) for x in monitor)
            ax.plot(xs,[monitor[str(x)]['successes'] for x in xs],'o-',label=condition)
        ax.set(title=group,xlabel='Additional training maps',ylabel='Greedy train-monitor successes',ylim=(-.5,10.5))
        ax.legend()
    fig.tight_layout();fig.savefig(output/'train_monitor.png',dpi=160);fig.savefig(output/'train_monitor.pdf');plt.close(fig)


def prepare(root, source_root):
    if (root/'plan.json').exists():
        raise ValueError('Existing plan; use --resume')
    root.mkdir(parents=True,exist_ok=True)
    sources={}
    common_cfg=None
    for group in GROUPS:
        path=(source_root/group/'latest.pth').resolve()
        saved=torch.load(path,map_location='cpu')
        cfg=Config.from_checkpoint(saved['config'])
        assert cfg.success_sampling=='pooled' and cfg.success_memory_mode=='per_map'
        assert cfg.train_maps==100 and cfg.validation_maps==cfg.test_maps==20
        assert saved['episode']==100 and saved['learner']['updates']==10895
        normalized={k:v for k,v in cfg.to_dict().items() if k not in ('variant','content')}
        if common_cfg is None: common_cfg=normalized
        else: assert common_cfg==normalized
        sources[group]={'path':str(path),'sha256':sha256(path),'episode':saved['episode'],
            'sac_updates':saved['learner']['updates'],'head_updates':saved['learner']['content_updates']}
        del saved
    plan={'scope':'One-seed paired continuation diagnostic; shared pretrained chain; development only.',
        'seed':common_cfg['seed'],'sources':sources,'conditions':list(CONDITIONS),'groups':list(GROUPS),
        'training_maps':100,'updates_per_map':20,'monitor_maps':10,'monitor_at':[50,100],
        'max_hours_per_worker':2.,'workers':3,'status':'prepared',
        'intervention':'Only successful imitation sampling; half whole episode, half final physical-time quarter; equal maps.',
        'selection':'All fixed endpoints reported; train prefix 0..9 monitoring only; no intermediate validation selection.',
        'test_policy_evaluations':0}
    write(root/'plan.json',plan)
    shutil.copy2(__file__,root/'orchestration_source.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT/'experiments/multiseed_20261012/seed_2')
    parser.add_argument('--stage',choices=['run','worker','report'],default='run')
    parser.add_argument('--condition',choices=CONDITIONS)
    parser.add_argument('--group',choices=list(GROUPS))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if Path(sys.prefix).resolve()!=Path('/home/wyz/anaconda3/envs/ir2_train'):
        parser.error('Use the ir2_train interpreter')
    if args.stage=='worker':
        if not args.group or not args.condition:parser.error('Worker requires group and condition')
        train_worker(root,args.condition,args.group,args.device,args.resume);return
    if args.stage=='report':report(root);return
    if not args.resume:prepare(root,args.source_root)
    plan=read(root/'plan.json');jobs=[]
    for condition in CONDITIONS:
        for group in GROUPS:
            d=root/condition/group
            if (d/'status.json').exists() and read(d/'status.json')['status']=='complete':continue
            if args.resume:
                archived = recover_initialization(d)
                if archived is not None:
                    plan.setdefault('startup_recoveries', []).append({'condition':condition,'group':group,
                        'archived':str(archived),'discarded_training_maps':0,'discarded_sac_updates':0,
                        'reason':'Coordinator interrupted while writing initial full checkpoint.'})
            (root/condition).mkdir(exist_ok=True)
            command=[sys.executable,str(Path(__file__).resolve()),'--stage','worker','--output',str(root),
                     '--condition',condition,'--group',group,'--device',args.device]
            if args.resume and (d/'latest.pth').exists():command.append('--resume')
            jobs.append((root/condition,group,command))
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        plan['status']='running';write(root/'plan.json',plan)
        run_worker_queue(jobs,plan['workers'],args.resume)
        report(root)
        plan['status']='complete';write(root/'plan.json',plan)
    except BaseException:
        plan['status']='interrupted_or_failed';write(root/'plan.json',plan);raise


if __name__=='__main__':main()
