#!/usr/bin/env python3
"""Collect all training layouts, fit a map-balanced head, compare frozen policies."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-finite-contact-matplotlib')
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner
from finite_contact.maps import configured_map, training_map_index
from finite_contact_train import append_json, checkpoint, evaluate, provenance
from finite_contact_multimap_report import audit_segments, physical_audit, rows


class Reservoir:
    """Uniform per-map priority reservoir; independent of action/learning RNG."""
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.RandomState(seed)
        self.data = None
        self.priorities = np.empty(0)
        self.seen = 0
        self.positive_seen = 0

    def add(self, samples):
        if not samples:
            return
        arrays = tuple(np.stack(x).astype(np.float32) for x in zip(*samples))
        self.seen += len(samples)
        self.positive_seen += int((arrays[2] > 0).sum())
        priorities = self.rng.uniform(size=len(samples))
        if self.data is not None:
            arrays = tuple(np.concatenate((a, b)) for a, b in zip(self.data, arrays))
            priorities = np.concatenate((self.priorities, priorities))
        keep = np.argsort(priorities, kind='stable')[:self.capacity]
        self.data = tuple(a[keep] for a in arrays)
        self.priorities = priorities[keep]


def balanced_batch(datasets, rng, size):
    maps = rng.randint(len(datasets), size=size)
    batch = []
    for index in maps:
        data = datasets[index]
        row = rng.randint(len(data[2]))
        batch.append(tuple(a[row] for a in data))
    return batch, maps


def load_learner(plan, device, trained=False, root=None):
    source = Path(plan['source']) if not trained else root / 'head/final.pth'
    saved = torch.load(source, map_location='cpu')
    cfg = Config.from_checkpoint(saved['config'])
    torch.set_num_threads(cfg.torch_threads)
    learner = Learner(cfg, device)
    learner.load(saved['learner'])
    if not trained:
        learner.context_encoder = copy.deepcopy(learner.policy).requires_grad_(False).eval()
    learner.policy.requires_grad_(False).eval()
    for network in learner.critics + learner.targets:
        network.requires_grad_(False).eval()
    return cfg, learner


def collect(root, plan, device):
    directory = root / 'collection'
    directory.mkdir()
    cfg, learner = load_learner(plan, device)
    assert cfg.content == 'fifo' and cfg.variant == 'A'
    provenance(directory, cfg)
    for episode in range(1, cfg.train_maps + 1):
        index = training_map_index(cfg, episode)
        env = Environment(cfg, 'train', index, scorer=learner.score)
        reservoir = Reservoir(plan['samples_per_map'], plan['seed'] + index + 1000)
        decisions = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if decisions % plan['label_every'] == 0:
                contexts = learner.contexts(env)
                env.feature_provider = lambda sender, ids: contexts[sender][ids]
                for sender in range(2):
                    reservoir.add(env.collect_labels(sender))
            observations = [env.observation(sender) for sender in range(2)]
            actions = learner.act(observations, greedy=True)
            _, _, terminated, truncated = env.step(actions)
            decisions += 1
        assert reservoir.data is not None and len(reservoir.data[2]) >= cfg.batch_size
        f, c, y = reservoir.data
        np.savez_compressed(directory / ('labels_%03d.npz' % index), features=f, context=c, labels=y)
        np.savez_compressed(directory / ('evaluation_map_%d_repeat_0.npz' % index),
                            truth=env.truth, beliefs=np.stack(env.beliefs), positions=env.positions)
        record = {**env.metrics(terminated), 'map': env.map_info, 'decisions': decisions,
                  'labels_seen': reservoir.seen, 'positive_seen': reservoir.positive_seen,
                  'labels_retained': len(y), 'positive_retained': int((y > 0).sum()),
                  'interrupted': False, 'evaluation_policy': 'greedy', 'repeat': 0}
        append_json(directory / 'evaluation.jsonl', record)
        append_json(directory / 'evaluation_trajectories.jsonl',
                    {'map': env.map_info, 'repeat': 0, 'trajectory': env.trajectory})
        print(json.dumps({'phase': 'collect', 'episode': episode, 'map': index,
                          'success': terminated, 'samples': len(y)}), flush=True)
    records = rows(directory / 'evaluation.jsonl')
    assert len(records) == 100 and {r['map']['index'] for r in records} == set(range(100))
    # This deterministic prefix was declared before collection, not selected by results.
    probe = root / 'training_probe'
    probe.mkdir()
    shutil.copy2(directory / 'config.json', probe / 'config.json')
    for record in rows(directory / 'evaluation_trajectories.jsonl'):
        if record['map']['index'] < 10:
            append_json(probe / 'evaluation_trajectories.jsonl', record)


def fit(root, plan, device):
    cfg, learner = load_learner(plan, device)
    records = rows(root / 'collection/evaluation.jsonl')
    assert len(records) == 100 and {r['map']['index'] for r in records} == set(range(100))
    datasets = []
    for index in range(cfg.train_maps):
        with np.load(root / 'collection' / ('labels_%03d.npz' % index)) as data:
            datasets.append(tuple(data[key].copy() for key in ('features', 'context', 'labels')))
    directory = root / 'head'
    directory.mkdir()
    provenance(directory, cfg)
    started = time.monotonic()
    checkpoint(directory / 'initial.pth', learner, cfg, 'content', 0, None, 0., full=False)
    learner.on_content_update = lambda row: append_json(directory / 'content_updates.jsonl', row)
    rng = np.random.RandomState(plan['seed'])
    counts = np.zeros(cfg.train_maps, dtype=int)
    target_updates = learner.content_updates + plan['head_updates']
    try:
        while learner.content_updates < target_updates:
            batch, indices = balanced_batch(datasets, rng, cfg.batch_size)
            counts += np.bincount(indices, minlength=cfg.train_maps)
            learner.update_content(batch)
            if learner.content_updates % 1000 == 0:
                checkpoint(directory / 'latest.pth', learner, cfg, 'content', 100, None,
                           time.monotonic() - started)
                print(json.dumps({'phase': 'fit', 'head_updates': learner.content_updates}), flush=True)
        assert np.all(counts > 0)
        checkpoint(directory / 'final.pth', learner, cfg, 'content', 100, None,
                   time.monotonic() - started, full=False)
    finally:
        checkpoint(directory / 'latest.pth', learner, cfg, 'content', 100, None,
                   time.monotonic() - started)
        (directory / 'sampling.json').write_text(json.dumps({
            'samples_per_training_map': counts.tolist(), 'sampler_state': 'fresh deterministic RandomState(seed)',
            'dataset_mean_target': float(np.mean([d[2].mean() for d in datasets])),
            'training_wall_seconds': time.monotonic() - started}, indent=2))


def compare(root, plan, device):
    cfg, learner = load_learner(plan, device, trained=True, root=root)
    for mode in ('fifo', 'rule', 'learned'):
        settings = copy.deepcopy(cfg)
        settings.content = mode
        directory = root / ('validation_' + mode)
        directory.mkdir()
        provenance(directory, settings)
        results = evaluate(learner, settings, 20, directory=directory)
        assert len(results) == 20 and {r['map']['index'] for r in results} == set(range(20))
        for record in results:
            append_json(directory / 'evaluation.jsonl', record)
        print(json.dumps({'phase': 'compare', 'content': mode,
                          'successes': sum(r['success'] for r in results)}), flush=True)


def weight_change(first, final, name):
    a, b = first[name], final[name]
    pairs = zip(a, b) if isinstance(a, list) else [(a, b)]
    return float(sum((x[k] - y[k]).double().square().sum().item()
                     for x, y in pairs for k in x) ** .5)


def report(root, plan):
    source = torch.load(plan['source'], map_location='cpu')
    initial = torch.load(root / 'head/initial.pth', map_location='cpu')
    final = torch.load(root / 'head/final.pth', map_location='cpu')
    cfg = Config.from_checkpoint(final['config'])
    changes = {name: weight_change(initial['learner'], final['learner'], name)
               for name in ('policy', 'critics', 'targets', 'context_encoder', 'content')}
    assert changes['content'] > 0 and all(v == 0 for k, v in changes.items() if k != 'content')
    assert weight_change(source['learner'], final['learner'], 'policy') == 0
    for key in initial['learner']['context_encoder']:
        assert torch.equal(initial['learner']['context_encoder'][key], source['learner']['policy'][key])
    assert torch.equal(initial['learner']['log_alpha'], final['learner']['log_alpha'])
    assert final['learner']['updates'] == source['learner']['updates']
    updates = rows(root / 'head/content_updates.jsonl')
    assert [r['content_updates'] for r in updates] == list(range(
        source['learner']['content_updates'] + 1, final['learner']['content_updates'] + 1))
    assert len(updates) == plan['head_updates']
    assert all(np.isfinite(v) for row in updates for v in row.values())
    collection = rows(root / 'collection/evaluation.jsonl')
    traces = rows(root / 'collection/evaluation_trajectories.jsonl')
    assert len(collection) == len(traces) == 100
    assert {r['map']['index'] for r in collection} == set(range(100))
    segments = 0
    for record, trace in zip(collection, traces):
        assert record['map'] == trace['map'] and record['map']['split'] == 'train'
        truth, _, info = configured_map(cfg, 'train', record['map']['index'])
        assert info == record['map']
        segments += audit_segments(truth, trace['trajectory'])
        with np.load(root / 'collection' / ('evaluation_map_%d_repeat_0.npz' % info['index'])) as data:
            np.testing.assert_array_equal(data['truth'], truth)
            free = truth == 255
            coverage = [float(np.count_nonzero((b == 255) & free) / free.sum()) for b in data['beliefs']]
            np.testing.assert_array_equal(coverage, record['coverage'])
        assert record['success'] == all(c >= cfg.completion for c in coverage)
        assert record['sent_bytes'] <= 2 * cfg.bytes_per_second * record['connected_seconds'] + 1e-6
        assert record['total_distance'] <= 2 * cfg.speed * record['time'] + 1e-6
    sampling = json.loads((root / 'head/sampling.json').read_text())
    assert min(sampling['samples_per_training_map']) > 0
    groups = {}
    baseline = None
    for mode in ('fifo', 'rule', 'learned'):
        directory = root / ('validation_' + mode)
        records = rows(directory / 'evaluation.jsonl')
        if baseline is None:
            baseline = records
        assert [r['map'] for r in records] == [r['map'] for r in baseline]
        groups[mode] = {'successes': sum(r['success'] for r in records), 'maps': len(records),
            'mean_minimum_coverage': float(np.mean([min(r['coverage']) for r in records])),
            'mean_capped_time': float(np.mean([r['time'] for r in records])),
            'mean_novel_received_cells': float(np.mean([r['novel_received_cells'] for r in records])),
            'mean_novel_cells_per_sent_kib': float(np.mean([
                r['novel_received_cells'] / max(r['sent_bytes'], 1) * 1024 for r in records])),
            'physical_audit': physical_audit(directory, cfg.to_dict())}
    if plan.get('reference_validation'):
        previous = rows(Path(plan['reference_validation']) / 'evaluation.jsonl')
        assert previous == baseline, 'Fresh FIFO evaluation differs from the saved movement reference'
    result = {'scope': 'Frozen movement, multi-map supervised communication head; no joint SAC training.',
        'source': plan['source'], 'source_sha256': plan['source_sha256'],
        'training_seed': plan['seed'], 'dataset_seed': cfg.map_seed,
        'collected_training_maps': len(collection),
        'collected_labels': sum(r['labels_seen'] for r in collection),
        'retained_labels': sum(r['labels_retained'] for r in collection),
        'training_fifo_successes': sum(r['success'] for r in collection),
        'new_head_updates': len(updates), 'cumulative_sac_updates': final['learner']['updates'],
        'new_sac_updates': 0, 'weight_change_l2': changes,
        'sampling': sampling, 'validation': groups, 'test_policy_evaluations': 0,
        'training_physical_audit': {'rollouts': 100, 'movement_segments': segments,
                                    'wall_crossings': 0, 'coverage_and_caps_passed': True},
        'fifo_matches_saved_reference': bool(plan.get('reference_validation')),
        'limitations': ['One trained movement seed and one content seed.',
                        'Changing content changes received maps, rendezvous features and subsequent actions.',
                        'Deployed communication totals refer to different trajectories; use fixed-state probes for ranking attribution.']}
    directory = root / 'report'
    directory.mkdir(exist_ok=True)
    (directory / 'summary.json').write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot([r['content_updates'] for r in updates], [r['content_loss'] for r in updates], lw=.5)
    axes[0].set(title='Map-balanced head training', xlabel='Actual head updates', ylabel='Batch MSE')
    names = list(groups)
    for ax, key, title in [(axes[1], 'successes', 'Same 20 development maps'),
                            (axes[2], 'mean_capped_time', 'Mean time (failures = 256 s)')]:
        bars = ax.bar(names, [groups[n][key] for n in names])
        ax.set_title(title)
        for bar, name in zip(bars, names):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    str(groups[name][key]) if key == 'successes' else '%.1f' % groups[name][key],
                    ha='center', va='bottom')
        ax.set_ylim(0, 22 if key == 'successes' else 280)
    fig.tight_layout()
    fig.savefig(directory / 'frozen_content.png', dpi=160)
    fig.savefig(directory / 'frozen_content.pdf')
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initialize-from', type=Path)
    p.add_argument('--reference-validation', type=Path)
    p.add_argument('--stage', choices=('all', 'collect', 'train', 'evaluate', 'report'), default='all')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--samples-per-map', type=int, default=2048)
    p.add_argument('--label-every', type=int, default=4)
    p.add_argument('--head-updates', type=int, default=5000)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = p.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/wyz/anaconda3/envs/ir2_train').resolve():
        p.error('Use ir2_train Python')
    if min(args.samples_per_map, args.label_every, args.head_updates) < 1:
        p.error('Positive sample and update budgets required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; refusing to silently change the experiment device')
    root = args.output.resolve()
    if args.stage in ('all', 'collect'):
        if not args.initialize_from or (root.exists() and any(root.iterdir())):
            p.error('Collection needs a movement checkpoint and a new output directory')
        saved = torch.load(args.initialize_from, map_location='cpu')
        cfg = Config.from_checkpoint(saved['config'])
        if saved['stage'] != 'movement' or cfg.map_family != 'simple_rooms_v1' or (
                cfg.train_maps, cfg.validation_maps, cfg.test_maps) != (100, 20, 20):
            p.error('Use the completed 100/20/20 movement experiment')
        root.mkdir(parents=True, exist_ok=True)
        plan = {'source': str(args.initialize_from.resolve()), 'source_sha256': hashlib.sha256(
                    args.initialize_from.read_bytes()).hexdigest(),
                'reference_validation': str(args.reference_validation.resolve()) if args.reference_validation else None,
                'seed': args.seed, 'samples_per_map': args.samples_per_map,
                'label_every': args.label_every, 'head_updates': args.head_updates,
                'training_probe_indices': list(range(10)), 'status': 'running', 'completed_stages': []}
    else:
        plan = json.loads((root / 'pipeline.json').read_text())
    assert hashlib.sha256(Path(plan['source']).read_bytes()).hexdigest() == plan['source_sha256']
    random.seed(plan['seed']); np.random.seed(plan['seed']); torch.manual_seed(plan['seed'])
    stages = [('collect', collect), ('train', fit), ('evaluate', compare), ('report', report)]
    try:
        for name, function in stages:
            if args.stage != 'all' and name != args.stage:
                continue
            plan['active_stage'] = name
            (root / 'pipeline.json').write_text(json.dumps(plan, indent=2))
            function(root, plan, args.device) if name != 'report' else function(root, plan)
            plan['completed_stages'].append(name)
            (root / 'pipeline.json').write_text(json.dumps(plan, indent=2))
        plan['status'] = 'complete' if len(set(plan['completed_stages'])) == 4 else 'stage_complete'
    except BaseException:
        plan['status'] = 'interrupted_or_failed'
        raise
    finally:
        (root / 'pipeline.json').write_text(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
