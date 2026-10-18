#!/usr/bin/env python3
"""Replicate matched adaptation while retaining one shared pretrained backbone/head."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from finite_contact_multimap_joint import GROUPS, ROOT, identical, prepare, sha256
from finite_contact_multimap_report import rows


def seed_statistics(records):
    if len(records) < 2 or len({r['seed'] for r in records}) != len(records):
        raise ValueError('At least two distinct adaptation seeds are required')
    groups = {}
    for group in GROUPS:
        metrics = [r['groups'][group] for r in records]
        if any(m['maps'] != 20 or not 0 <= m['successes'] <= 20 or
               not np.isfinite(m['mean_capped_time']) or not 0 < m['mean_capped_time'] <= 256 for m in metrics):
            raise ValueError('Incomplete or invalid development evaluation')
        groups[group] = {}
        for field in ('successes', 'mean_capped_time', 'mean_minimum_coverage', 'mean_coverage_deficit'):
            values = [m[field] for m in metrics]
            if not np.all(np.isfinite(values)):
                raise ValueError('Nonfinite evaluation metric')
            groups[group][field] = {'mean': float(np.mean(values)), 'sample_std': float(np.std(values, ddof=1)),
                                    'per_seed': values}
    paired = {}
    for base in ('A_fifo', 'A_rule'):
        paired[base] = [{'seed': r['seed'],
                        'success_gain': r['groups']['AB_learned']['successes'] - r['groups'][base]['successes'],
                        'time_saved_seconds': r['groups'][base]['mean_capped_time'] - r['groups']['AB_learned']['mean_capped_time'],
                        'fraction_time_saved': 1-r['groups']['AB_learned']['mean_capped_time']/r['groups'][base]['mean_capped_time']}
                       for r in records]
    return {'adaptation_seeds': [r['seed'] for r in records], 'groups': groups, 'paired_learned_vs': paired,
            'aggregation_unit': 'One final model per adaptation seed; sample SD across seeds, not 60 IID maps.'}


def aggregate(root):
    plan = json.loads((root / 'replication.json').read_text())
    records, common, config, source = [], None, None, None
    combined = {'rollouts': 0, 'movement_segments': 0, 'wall_crossings': 0}
    for run in plan['runs']:
        directory = Path(run['directory'])
        pipeline = json.loads((directory / 'pipeline.json').read_text())
        summary = json.loads((directory / 'report/summary.json').read_text())
        if pipeline['status'] != 'complete' or summary['training_seed'] != run['seed']:
            raise ValueError('A complete correctly identified run is required')
        assert summary['test_policy_evaluations'] == 0
        saved = torch.load(directory / 'common_start.pth', map_location='cpu')
        assert sha256(directory / 'common_start.pth') == pipeline['common_sha256']
        normalized = {k: v for k, v in saved['config'].items() if k != 'seed'}
        identity = {k: pipeline[k] for k in ('movement_sha256', 'head_sha256', 'dataset_seed',
                    'sweeps', 'updates_per_map', 'source_sac_updates', 'new_sac_updates_per_group', 'frozen_head_updates')}
        if common is None:
            common, config, source = saved['learner'], normalized, identity
        else:
            assert identical(common, saved['learner']), 'Different pretrained network or optimizer state'
            assert config == normalized and source == identity, 'Configuration or training budget differs'
        groups = {}
        for group in GROUPS:
            metric = summary['groups'][group]
            assert metric['training_episodes'] == metric['distinct_training_maps'] == 100
            assert metric['new_sac_updates'] == 2000 and metric['cumulative_sac_updates'] == 10895
            assert metric['weight_change_l2']['content'] == metric['weight_change_l2']['context_encoder'] == 0
            assert json.loads((directory/group/'status.json').read_text())['status'] == 'complete'
            evaluation = rows(directory/group/'validation_final/evaluation.jsonl')
            assert len(evaluation) == 20 and [r['map']['index'] for r in evaluation] == list(range(20))
            groups[group] = metric['validation_final']
        records.append({'seed': run['seed'], 'groups': groups, 'directory': str(directory)})
        for key in combined:
            combined[key] += summary['combined_physical_audit'][key]
    result = {**seed_statistics(records), 'scope': plan['scope'], 'per_run': records,
              'shared_pretrained_state_and_config_checked': True, 'source_identity': source,
              'combined_physical_audit': combined, 'test_policy_evaluations': 0,
              'limitations': ['Adaptation seeds share one pretrained movement model and communication head.',
                              'All models use the same 20 development maps; no independent test policy execution.',
                              'Three adaptation seeds provide descriptive robustness evidence, not full independent training replication.']}
    output = root/'report'
    output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    shutil.copy2(__file__, output/'analysis_source.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = ['gray', 'tab:orange', 'tab:blue']
    for group, color in zip(GROUPS, colors):
        metric = result['groups'][group]
        axes[0, 0].plot(result['adaptation_seeds'], metric['successes']['per_seed'], 'o-', color=color, label=group)
        axes[0, 1].plot(result['adaptation_seeds'], metric['mean_capped_time']['per_seed'], 'o-', color=color, label=group)
    axes[0, 0].set(title='Same 20 development maps', ylabel='Successful maps', ylim=(0, 21))
    axes[0, 1].set(title='Failures included at 256 s', ylabel='Mean capped time (s)', ylim=(0, 280))
    for ax in axes[0]:
        ax.set_xticks(result['adaptation_seeds']); ax.set_xlabel('Adaptation seed'); ax.legend()
    for ax, field, title in [(axes[1, 0], 'successes', 'Mean and sample SD across adaptation seeds'),
                              (axes[1, 1], 'mean_capped_time', 'Shared pretrained movement and head')]:
        values = [result['groups'][g][field]['mean'] for g in GROUPS]
        errors = [result['groups'][g][field]['sample_std'] for g in GROUPS]
        ax.bar(list(GROUPS), values, yerr=errors, color=colors, capsize=6)
        ax.set(title=title, ylim=(0, 23 if field == 'successes' else 280))
    fig.tight_layout(); fig.savefig(output/'adaptation_seeds.png', dpi=160)
    fig.savefig(output/'adaptation_seeds.pdf'); plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


def run_worker_queue(jobs, limit, resume=False):
    if limit < 1:
        raise ValueError('Positive worker limit required')
    pending, children = list(jobs), []
    try:
        while pending or any(c.poll() is None for _, _, c, _ in children):
            if any(c.poll() not in (None, 0) for _, _, c, _ in children):
                raise RuntimeError('A worker failed; stopping other workers at safe boundaries')
            active = sum(c.poll() is None for _, _, c, _ in children)
            while pending and active < limit:
                directory, group, command = pending.pop(0)
                log = (directory/(group+'.log')).open('a' if resume else 'w')
                children.append((directory, group, subprocess.Popen(command, cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT), log))
                active += 1
            if pending or any(c.poll() is None for _, _, c, _ in children):
                time.sleep(.2)
        if any(c.returncode for _, _, c, _ in children):
            raise RuntimeError('A worker failed')
    except BaseException:
        for _, _, child, _ in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for _, _, child, _ in children:
            child.wait()
        raise
    finally:
        for _, _, _, log in children:
            log.close()


def execute(root, plan, device, resume):
    jobs = []
    for run in plan['runs'][1:]:
        directory = Path(run['directory'])
        pipeline = json.loads((directory/'pipeline.json').read_text())
        pipeline['status'] = 'running'
        (directory/'pipeline.json').write_text(json.dumps(pipeline, indent=2))
        for group in GROUPS:
            status = directory/group/'status.json'
            if status.exists() and json.loads(status.read_text())['status'] == 'complete':
                continue
            command = [sys.executable, str(ROOT/'finite_contact_multimap_joint.py'), '--stage', 'group',
                       '--output', str(directory), '--group', group, '--device', device]
            latest = directory/group/'latest.pth'
            if latest.exists():
                if not resume:
                    raise ValueError('Existing worker state needs --resume')
                command += ['--resume', str(latest)]
            jobs.append((directory, group, command))
    plan['status'] = 'running'
    (root/'replication.json').write_text(json.dumps(plan, indent=2))
    try:
        run_worker_queue(jobs, plan['concurrent_workers'], resume)
        for run in plan['runs'][1:]:
            directory = Path(run['directory'])
            with (directory/'report.log').open('a') as log:
                subprocess.run([sys.executable, str(ROOT/'finite_contact_multimap_joint.py'), '--stage', 'report',
                                '--output', str(directory), '--device', 'cpu'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            pipeline = json.loads((directory/'pipeline.json').read_text())
            pipeline['status'] = 'complete'
            (directory/'pipeline.json').write_text(json.dumps(pipeline, indent=2))
        aggregate(root)
        plan['status'] = 'complete'
    except BaseException:
        plan['status'] = 'interrupted_or_failed'
        raise
    finally:
        (root/'replication.json').write_text(json.dumps(plan, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--existing-run', type=Path)
    parser.add_argument('--stage', choices=('run', 'report'), default='run')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Positive worker limit required')
    if Path(sys.prefix).resolve() != Path('/home/wyz/anaconda3/envs/ir2_train'):
        parser.error('Use the ir2_train environment')
    root = args.output.resolve()
    if args.stage == 'report':
        aggregate(root); return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; no silent CPU fallback')
    if not args.resume:
        if not args.existing_run or (root.exists() and any(root.iterdir())):
            parser.error('Specify the completed seed=1 run and a new output directory')
        original = json.loads((args.existing_run/'pipeline.json').read_text())
        if original['status'] != 'complete' or original['seed'] != 1:
            parser.error('Expected the completed adaptation seed=1 run')
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, root/'orchestration_source.py')
        shutil.copy2(ROOT/'finite_contact/MULTISEED_EXPERIMENT.md', root/'pre_run_design.md')
        plan = {'scope': 'Adaptation seeds 1, 2, 3 with one shared pretrained movement/head; no test policy execution.',
                'new_adaptation_seeds': [2, 3], 'concurrent_workers': args.workers,
                'runs': [{'seed': 1, 'directory': str(args.existing_run.resolve())}], 'status': 'prepared'}
        for seed in (2, 3):
            directory = root/('seed_%d' % seed)
            preparation = argparse.Namespace(movement_state=Path(original['movement_source']),
                    content_checkpoint=Path(original['head_source']), reference_root=None,
                    seed=seed, policy_learning_rate=original['actor_learning_rate'], sweeps=1,
                    updates_per_map=20, max_hours=1.5)
            prepare(directory, preparation)
            plan['runs'].append({'seed': seed, 'directory': str(directory)})
        (root/'replication.json').write_text(json.dumps(plan, indent=2))
    else:
        plan = json.loads((root/'replication.json').read_text())
        if plan['concurrent_workers'] != args.workers:
            plan.setdefault('resource_changes', []).append({'previous_workers': plan['concurrent_workers'],
                'resumed_workers': args.workers, 'time_ns': time.time_ns(),
                'reason': 'Replay memory growth; training algorithm and budgets unchanged.'})
        plan['concurrent_workers'] = args.workers
        shutil.copy2(__file__, root/('orchestration_resume_%d.py' % time.time_ns()))
    def terminate(signum, frame):
        raise KeyboardInterrupt('Replication interrupted')
    signal.signal(signal.SIGTERM, terminate)
    execute(root, plan, args.device, args.resume)


if __name__ == '__main__':
    main()
