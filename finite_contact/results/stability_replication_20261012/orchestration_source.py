#!/usr/bin/env python3
"""Confirm the seed-2 sampling pilot on adaptation seeds 1 and 3, at matched budgets."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import torch
from finite_contact_multimap_joint import GROUPS, sha256
from finite_contact_multimap_report import rows
from finite_contact_multiseed import run_worker_queue, seed_statistics
from finite_contact_stability import CONDITIONS, ROOT, prepare, read, recover_initialization, report, write


IDENTITY_FILES = ('finite_contact_stability.py', 'finite_contact_stability_replication.py',
                  'finite_contact/config.py', 'finite_contact/learning.py',
                  'finite_contact_train.py', 'finite_contact_multimap_joint.py',
                  'finite_contact_multiseed.py')


def prepare_replication(root, pilot, previous, workers):
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new output directory; existing experiments are preserved')
    old = read(previous/'replication.json')
    pilot_plan = read(pilot/'plan.json')
    assert old['status'] == pilot_plan['status'] == 'complete'
    assert pilot_plan['seed'] == 2
    assert {r['seed'] for r in old['runs']} == {1, 2, 3}
    sources = {r['seed']: Path(r['directory']) for r in old['runs']}
    for group in GROUPS:
        assert pilot_plan['sources'][group]['sha256'] == sha256(sources[2]/group/'latest.pth')
    root.mkdir(parents=True)
    shutil.copy2(__file__, root/'orchestration_source.py')
    shutil.copy2(ROOT/'finite_contact/STABILITY_REPLICATION_EXPERIMENT.md', root/'pre_run_design.md')
    (root/'checks').mkdir()
    identity = {name: sha256(ROOT/name) for name in IDENTITY_FILES}
    write(root/'checks/source_identity_before.json', identity)
    runs = [{'seed': 2, 'directory': str(pilot.resolve()), 'role': 'existing exploratory pilot'}]
    for seed in (1, 3):
        directory = root/('seed_%d' % seed)
        prepare(directory, sources[seed])
        plan = read(directory/'plan.json')
        assert plan['seed'] == seed
        for field in ('conditions', 'groups', 'training_maps', 'updates_per_map',
                      'monitor_maps', 'monitor_at', 'max_hours_per_worker'):
            assert plan[field] == pilot_plan[field]
        runs.append({'seed': seed, 'directory': str(directory), 'role': 'fixed-condition replication'})
    plan = {'scope': 'Adaptation seeds 1 and 3 replicate the completed seed-2 pilot; shared pretrained chain; development only.',
            'pilot_seed': 2, 'confirmation_seeds': [1, 3], 'concurrent_workers': workers,
            'runs': runs, 'status': 'prepared', 'test_policy_evaluations': 0,
            'previous_adaptation_plan': str((previous/'replication.json').resolve()),
            'previous_adaptation_plan_sha256': sha256(previous/'replication.json'),
            'pilot_plan_sha256': sha256(pilot/'plan.json'),
            'pilot_summary_sha256': sha256(pilot/'report/summary.json'),
            'selection': 'All twelve new fixed endpoints; no early development selection, no test policy executions.'}
    write(root/'replication.json', plan)
    return plan


def audit_pair(directory, identity):
    plan = read(directory/'plan.json')
    result = {'source_snapshot_identity': {}, 'paired_first_rollout': {},
              'map_schedules_equal': True, 'optimizer_steps': {}, 'test_policy_evaluations': 0}
    for group in GROUPS:
        source = torch.load(plan['sources'][group]['path'], map_location='cpu')
        first, first_trace, schedule = None, None, None
        for condition in CONDITIONS:
            worker = directory/condition/group
            # The wrapper is saved once at the experiment root; all actual training files
            # are also independently saved by each worker's provenance routine.
            for name, digest in identity.items():
                assert sha256(ROOT/name) == digest
                if name != 'finite_contact_stability_replication.py':
                    assert sha256(worker/'source'/name) == digest
            result['source_snapshot_identity'][condition+'/'+group] = True
            training = rows(worker/'training.jsonl')
            metric = {k:v for k,v in training[0].items() if k != 'wall_seconds'}
            trace = rows(worker/'trajectories.jsonl')[0]
            actual = [r['map']['index'] for r in training]
            if first is None:
                first, first_trace, schedule = metric, trace, actual
            else:
                assert (metric, trace, actual) == (first, first_trace, schedule)
            assert len(actual) == 100 and set(actual) == set(range(100))
            final = torch.load(worker/'final.pth', map_location='cpu')['learner']
            initial = source['learner']
            optimizers = [(k, initial[k], final[k]) for k in ('policy_optimizer', 'alpha_optimizer')]
            optimizers += [('q_optimizer_%d'%i, a, b) for i,(a,b) in enumerate(
                           zip(initial['q_optimizers'], final['q_optimizers']))]
            steps = {}
            for name, old, new in optimizers:
                a, b = old['state'], new['state']
                assert a.keys() == b.keys() and a
                deltas = [float(b[k]['step']-a[k]['step']) for k in a]
                assert set(deltas) == {2000.}
                steps[name] = {'state_parameters':len(deltas), 'actual_step_delta':2000}
            result['optimizer_steps'][condition+'/'+group] = steps
            del final
        result['paired_first_rollout'][group] = True
        del source
    (directory/'checks').mkdir(exist_ok=True)
    write(directory/'checks/pair_integrity.json', result)


def aggregate_records(records):
    if sorted(r['seed'] for r in records) != [1, 2, 3]:
        raise ValueError('Exactly one complete record for each seed 1, 2 and 3 is required')
    records = sorted(records, key=lambda r:r['seed'])
    by_condition = {}
    for condition in CONDITIONS:
        converted = []
        for record in records:
            groups = {}
            for group in GROUPS:
                value = record['groups'][group][condition]
                if value['new_sac_updates'] != 2000 or value['cumulative_sac_updates'] != 12895:
                    raise ValueError('All endpoints must have the same full continuation budget')
                groups[group] = value['validation_final']
            converted.append({'seed':record['seed'], 'groups':groups})
        by_condition[condition] = seed_statistics(converted)
    paired = {}
    for group in GROUPS:
        paired[group] = []
        for record in records:
            a,b = [record['groups'][group][c]['validation_final'] for c in CONDITIONS]
            paired[group].append({'seed':record['seed'], 'success_gain':b['successes']-a['successes'],
                'time_saved_seconds':a['mean_capped_time']-b['mean_capped_time']})
    return {'conditions':by_condition, 'paired_sampling_changes':paired,
            'aggregation_unit':'Final model per adaptation seed; sample SD over three seeds; not 60 independent maps.'}


def aggregate(root):
    plan = read(root/'replication.json')
    pilot = next(Path(r['directory']) for r in plan['runs'] if r['seed'] == 2)
    assert sha256(pilot/'plan.json') == plan['pilot_plan_sha256']
    assert sha256(pilot/'report/summary.json') == plan['pilot_summary_sha256']
    assert sha256(plan['previous_adaptation_plan']) == plan['previous_adaptation_plan_sha256']
    records, canonical_maps, canonical_config = [], None, None
    combined = {'rollouts':0, 'movement_segments':0, 'wall_crossings':0}
    for run in sorted(plan['runs'], key=lambda r:r['seed']):
        directory = Path(run['directory'])
        child = read(directory/'plan.json')
        summary = read(directory/'report/summary.json')
        assert child['status'] == 'complete' and summary['training_seed'] == run['seed']
        assert summary['test_policy_evaluations'] == 0
        for group in GROUPS:
            for condition in CONDITIONS:
                worker = directory/condition/group
                assert read(worker/'status.json')['status'] == 'complete'
                cfg = read(worker/'config.json')
                normalized = {k:v for k,v in cfg.items() if k not in ('seed','variant','content','success_sampling')}
                maps = [r['map'] for r in rows(worker/'validation_final/evaluation.jsonl')]
                if canonical_maps is None:
                    canonical_maps, canonical_config = maps, normalized
                else:
                    assert maps == canonical_maps and normalized == canonical_config
                changes = summary['groups'][group][condition]['weight_change_l2']
                assert changes['content'] == changes['context_encoder'] == 0
        records.append({'seed':run['seed'], 'groups':summary['groups'], 'directory':str(directory)})
        for key in combined:
            combined[key] += summary['combined_physical_audit'][key]
    result = {**aggregate_records(records), 'scope':plan['scope'], 'pilot_seed':2,
              'confirmation_seeds':[1,3], 'per_run':records, 'combined_physical_audit':combined,
              'test_policy_evaluations':0,
              'limitations':['One shared pretraining/head chain; three adaptation seeds only.',
                             'Seed 2 selected as weaker exploratory pilot; seeds 1 and 3 follow fixed pilot conditions.',
                             'Same development maps have been used in earlier experiments; no independent test.',
                             'All training-monitor regressions retained; training defaults are unchanged.']}
    output = root/'report'; output.mkdir(exist_ok=True)
    write(output/'summary.json', result)
    shutil.copy2(__file__, output/'analysis_source.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(14,7))
    for column,group in enumerate(GROUPS):
        for condition in CONDITIONS:
            stats = result['conditions'][condition]['groups'][group]
            for ax,field in [(axes[0,column],'successes'),(axes[1,column],'mean_capped_time')]:
                ax.plot([1,2,3],stats[field]['per_seed'],'o-',label=condition)
                ax.set_xticks([1,2,3]);ax.set_xlabel('Adaptation seed');ax.legend()
        axes[0,column].set(title=group,ylabel='Successes / same 20 development maps',ylim=(-.5,20.5))
        axes[1,column].set(ylabel='Mean capped time (s)',ylim=(0,270))
    fig.tight_layout();fig.savefig(output/'sampling_seeds.png',dpi=160)
    fig.savefig(output/'sampling_seeds.pdf');plt.close(fig)
    print(json.dumps({'conditions':result['conditions'], 'paired_sampling_changes':result['paired_sampling_changes'],
                      'combined_physical_audit':combined},indent=2),flush=True)


def execute(root, plan, device, resume):
    jobs = []
    for run in plan['runs']:
        if run['seed'] == 2:continue
        directory = Path(run['directory']);child = read(directory/'plan.json')
        child['status'] = 'running'
        for condition in CONDITIONS:
            (directory/condition).mkdir(exist_ok=True)
            for group in GROUPS:
                worker = directory/condition/group
                if (worker/'status.json').exists() and read(worker/'status.json')['status'] == 'complete':continue
                if resume:
                    archived = recover_initialization(worker)
                    if archived is not None:
                        child.setdefault('startup_recoveries',[]).append({'condition':condition,'group':group,
                            'archived':str(archived),'discarded_training_maps':0,'discarded_sac_updates':0})
                command = [sys.executable,str(ROOT/'finite_contact_stability.py'),'--stage','worker',
                           '--output',str(directory),'--condition',condition,'--group',group,'--device',device]
                if (worker/'latest.pth').exists():
                    if not resume:raise ValueError('Existing worker state requires --resume')
                    command.append('--resume')
                jobs.append((directory/condition,group,command))
        write(directory/'plan.json',child)
    plan['status'] = 'running';write(root/'replication.json',plan)
    try:
        run_worker_queue(jobs,plan['concurrent_workers'],resume)
        identity = read(root/'checks/source_identity_before.json')
        for run in plan['runs']:
            if run['seed'] == 2:continue
            directory = Path(run['directory'])
            report(directory)
            audit_pair(directory,identity)
            child = read(directory/'plan.json');child['status'] = 'complete';write(directory/'plan.json',child)
            script = directory/'checks/analyze_endpoints.py'
            pilot = next(Path(r['directory']) for r in plan['runs'] if r['seed'] == 2)
            shutil.copy2(pilot/'checks/analyze_endpoints.py',script)
            with (directory/'checks/failure_analysis.log').open('a') as log:
                subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        aggregate(root)
        after = {name:sha256(ROOT/name) for name in identity}
        assert after == identity
        write(root/'checks/source_identity_after.json',after)
        plan['status'] = 'complete'
    except BaseException:
        plan['status'] = 'interrupted_or_failed';raise
    finally:
        write(root/'replication.json',plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--existing-run',type=Path,default=ROOT/'experiments/stability_20261012')
    parser.add_argument('--previous-replication',type=Path,default=ROOT/'experiments/multiseed_20261012')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--stage',choices=['run','report'],default='run')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/wyz/anaconda3/envs/ir2_train'):
        parser.error('Use ir2_train')
    if not 1 <= args.workers <= 3:parser.error('Use one to three workers; preserve the tested memory bound')
    root = args.output.resolve()
    if args.stage == 'report':aggregate(root);return
    if args.device == 'cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
    if args.resume:
        plan = read(root/'replication.json')
        assert plan['concurrent_workers'] == args.workers
    else:
        plan = prepare_replication(root,args.existing_run,args.previous_replication,args.workers)
    def stop(signum,frame):raise KeyboardInterrupt('Replication interrupted')
    signal.signal(signal.SIGTERM,stop)
    execute(root,plan,args.device,args.resume)


if __name__ == '__main__':main()
