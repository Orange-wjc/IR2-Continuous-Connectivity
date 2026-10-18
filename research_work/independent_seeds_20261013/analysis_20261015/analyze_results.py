#!/usr/bin/env python3
"""Read retained experiment artifacts; never train or execute a policy."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-independent-analysis-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'experiments/independent_seeds_20261013'
OUT = REPO / 'finite_contact/results/independent_seeds_20261013'
SOURCE = REPO / 'research_work/independent_seeds_20261013/source'
SEEDS = (101, 202, 303)
GROUPS = ('B0_fifo', 'A_fifo', 'B_learned', 'AB_learned', 'AB_uncoupled')
PAIRS = [('B0_fifo', 'A_fifo'), ('B0_fifo', 'B_learned'),
         ('B_learned', 'AB_learned'), ('A_fifo', 'AB_learned'),
         ('AB_uncoupled', 'AB_learned'), ('B0_fifo', 'AB_learned')]
EPS = 1e-6

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def csv_out(name, records):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def mean(values):
    return float(np.mean(values))

def close(a, b):
    assert abs(a - b) < 1e-8, (a, b)

def same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b

def describe_difference(values):
    a = np.asarray(values, dtype=float)
    order = np.argsort(a)
    return dict(mean_seconds=mean(a), median_seconds=float(np.median(a)),
                faster=int(np.sum(a > EPS)), slower=int(np.sum(a < -EPS)), ties=int(np.sum(abs(a) <= EPS)),
                min_seconds=float(a.min()), max_seconds=float(a.max()),
                mean_without_largest_gain=mean(a[order[:-1]]),
                mean_without_largest_loss=mean(a[order[1:]]),
                leave_one_map_out_range=[float((a.sum() - a).min() / (len(a)-1)),
                                         float((a.sum() - a).max() / (len(a)-1))])

def save_figure(fig, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / (name + '.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert read(ROOT / 'pipeline.json')['status'] == 'complete'
    summary = read(ROOT / 'report/summary.json')
    assert summary['status'] == 'complete' and summary['test_policy_evaluations'] == 0
    before_manifest = read(OUT / 'manifest.json')['files']
    assert all(sha(OUT / f) == value for f, value in before_manifest.items()), 'Previous evidence changed'
    active = read(ROOT / 'source_identity_audit_resume_20261014.json')
    assert active == read(ROOT / 'source_identity_after.json')
    assert len(active) == 54
    for filename, expected in active.items():
        assert sha(SOURCE / filename) == expected
        assert sha(ROOT / 'source_audit_resume_20261014' / filename) == expected
    original = read(ROOT / 'source_identity.json')
    assert all(sha(ROOT / 'source' / f) == value for f, value in original.items())

    # Import only existing read-only auditors, never any training orchestration.
    sys.path.insert(0, str(SOURCE))
    from finite_contact.config import Config
    from finite_contact_thousand_analysis import audit_rollouts, communication_audit
    import torch
    torch.set_num_threads(2)

    data, training, audits = {}, {}, {}
    metrics_csv, curves, group_table, failures = [], [], [], []
    input_hashes = {}
    total_physics = dict(rollouts=0, movement_segments=0, wall_crossings=0, boundary_contacts=0)
    total_radio_events = 0
    verified_development = dict(rollouts=0, movement_segments=0, boundary_contacts=0, radio_events=0)
    model_checks = []
    stages = []
    common_stage_rechecks = []

    def retain(path):
        input_hashes[str(path.relative_to(REPO))] = sha(path)

    for name in ('pipeline.json', 'source_identity.json', 'source_identity_after.json',
                 'source_identity_audit_resume_20261014.json', 'report/summary.json',
                 'report/RESULTS_ANALYSIS.md', 'dataset/manifest.json'):
        path = ROOT / name
        if path.exists(): retain(path)
    for seed in SEEDS:
        seed_root = ROOT / ('seed_%d' % seed)
        common_path = seed_root / 'common_start.pth'
        ci = read(seed_root / 'common_identity.json')
        assert sha(common_path) == ci['sha256']
        common = torch.load(common_path, map_location='cpu')['learner']
        retain(common_path); retain(seed_root / 'common_identity.json')
        for stage in ('movement', 'content'):
            directory = seed_root / stage
            audit = read(directory / 'audit.json')
            assert audit['physical']['rollouts'] == 100 and audit['physical']['wall_crossings'] == 0
            if stage == 'movement':
                assert all(v['increment'] == 9000 for v in audit['optimizer'].values())
                stage_updates = rows(directory / 'updates.jsonl')
                assert [v['updates'] for v in stage_updates] == list(range(1, 9001))
                retain(directory / 'updates.jsonl')
            else:
                assert audit['movement_frozen'] and audit['head_updates'] == 5000
                head_updates = rows(directory / 'head/content_updates.jsonl')
                assert [v['content_updates'] for v in head_updates] == list(range(1, 5001))
                retain(directory / 'head/content_updates.jsonl')
            # Seed 101's original common-stage audit predates the reviewed
            # continuous geometry audit. Recompute all six stages consistently.
            records_dir = directory if stage == 'movement' else directory / 'collection'
            stage_cfg = Config.from_checkpoint(read(records_dir / 'config.json'))
            physical = audit_rollouts(records_dir, stage_cfg, 100)
            for key, value in audit['physical'].items():
                assert physical[key] == value
            communication = communication_audit(records_dir, rows(records_dir / 'training.jsonl'))
            assert communication == audit['communication']
            stages.append(dict(seed=seed, stage=stage, physical=physical, communication=communication))
            common_stage_rechecks.append(dict(seed=seed, stage=stage, physical=physical, communication=communication))
            for filename in ('training.jsonl', 'trajectories.jsonl', 'communication.jsonl', 'config.json'):
                retain(records_dir / filename)
            retain(directory / 'audit.json'); retain(directory / 'status.json')
        for group in GROUPS:
            directory = seed_root / group
            evaluation = directory / 'validation_final'
            records = rows(evaluation / 'evaluation.jsonl')
            train = rows(directory / 'training.jsonl')
            updates = rows(directory / 'updates.jsonl')
            cfg = read(directory / 'config.json')
            assert cfg['seed'] == seed and cfg['completion'] == .99 and cfg['horizon_seconds'] == 256
            assert len(records) == 20 and len(train) == 1000 and len(updates) == 20000
            assert [r['episode'] for r in train] == list(range(1, 1001))
            assert sorted(r['map']['index'] for r in train) == list(range(1000))
            assert [r['updates'] for r in updates] == list(range(9001, 29001))
            assert all(r['episode'] == i // 20 + 1 for i, r in enumerate(updates))
            for i, row in enumerate(train, 1):
                assert row['map']['split'] == 'train' and not row['interrupted']
                assert row['learner_updates'] == 9000 + 20 * i
                assert row['post_episode_sac_updates'] == 20 and row['online_sac_updates'] == 0
                assert row['content_updates'] == 5000
            for i, row in enumerate(records):
                assert row['map']['split'] == 'validation' and row['map']['index'] == i
                assert row['evaluation_policy'] == 'greedy' and row['repeat'] == 0
                assert row['learner_updates'] == 29000 and not row['interrupted']
                assert 0 <= row['time'] <= 256 and row['success'] == (min(row['coverage']) >= .99)
                if not row['success']: close(row['time'], 256)
                with np.load(evaluation / ('evaluation_map_%d_repeat_0.npz' % i)) as saved:
                    free = saved['truth'] == 255
                    seen = (saved['beliefs'] == 255) & free
                    cov = seen.sum(axis=(1, 2)) / free.sum()
                    union = seen.any(axis=0)
                    np.testing.assert_allclose(cov, row['coverage'], atol=1e-12, rtol=0)
                    close(float(union.sum() / free.sum()), row['union_coverage'])
                    if not row['success']:
                        worst = int(np.argmin(cov))
                        failures.append(dict(seed=seed, group=group, map_index=i, time=row['time'],
                            coverage=row['coverage'], union_coverage=row['union_coverage'],
                            direct_task_deficit=('team_exploration_below_threshold' if row['union_coverage'] < .99 else 'local_map_synchronization_below_threshold'),
                            unexplored_free_cells=int((free & ~union).sum()),
                            explored_cells_missing_at_worst_robot=int((union & ~seen[worst]).sum()),
                            queue_bytes=row['queue_bytes'], connected_seconds=row['connected_seconds']))
                retain(evaluation / ('evaluation_map_%d_repeat_0.npz' % i))
                metrics_csv.append(dict(seed=seed, group=group, map_index=i, success=row['success'],
                    time_seconds=row['time'], union_completion_time=row['union_completion_time'],
                    union_coverage=row['union_coverage'], min_local_coverage=min(row['coverage']),
                    total_distance=row['total_distance'], team_wait_seconds=sum(row['wait_seconds']),
                    connected_seconds=row['connected_seconds'], sent_bytes=row['sent_bytes'],
                    delivered_bytes=row['delivered_bytes'], control_bytes=row['control_bytes'],
                    novel_received_cells=row['novel_received_cells'], duplicate_received_cells=row['duplicate_received_cells']))
            data[seed, group], training[seed, group] = records, train
            result = summary['seeds'][str(seed)]['groups'][group]
            close(mean([r['time'] for r in records]), result['mean_capped_time'])
            assert sum(r['success'] for r in records) == result['successes']
            audit = read(directory / 'audit.json')
            assert read(directory / 'status.json')['status'] == 'complete'
            recomputed = audit_rollouts(evaluation, Config.from_checkpoint(cfg), 20, True)
            assert recomputed == audit['development_physical']
            comm = communication_audit(evaluation, records, True)
            assert comm == audit['development_communication']
            verified_development['rollouts'] += 20
            verified_development['movement_segments'] += recomputed['movement_segments']
            verified_development['boundary_contacts'] += recomputed['boundary_contacts']
            verified_development['radio_events'] += comm['events']
            assert all(v['increment_per_parameter'] == 20000 for k, v in audit['optimizer'].items() if isinstance(v, dict))
            assert audit['optimizer']['content_optimizer_unchanged']
            stages.append(dict(seed=seed, stage=group+'/train', physical=audit['training_physical'], communication=audit['training_communication']))
            stages.append(dict(seed=seed, stage=group+'/dev', physical=recomputed, communication=comm))
            audits[seed, group] = audit
            initial = torch.load(directory / 'initial.pth', map_location='cpu')['learner']
            final = torch.load(directory / 'final.pth', map_location='cpu')['learner']
            assert same(initial, common), (seed, group, 'initial != common')
            for key in ('content', 'context_encoder', 'content_optimizer'):
                assert same(initial[key], final[key]), (seed, group, key)
            assert final['updates'] == 29000 and final['content_updates'] == 5000
            for key in ('policy_optimizer', 'alpha_optimizer'):
                assert {int(x['step']) for x in final[key]['state'].values()} == {29000}
            for opt in final['q_optimizers']:
                assert {int(x['step']) for x in opt['state'].values()} == {29000}
            model_checks.append(dict(seed=seed, group=group, common_initial_exact=True, frozen_content_context_optimizer=True, final_sac_updates=29000))
            del initial, final, updates
            for path in [directory / n for n in ('initial.pth','final.pth','audit.json','status.json','training.jsonl','updates.jsonl','config.json')]+[evaluation / n for n in ('evaluation.jsonl','evaluation_trajectories.jsonl','evaluation_communication.jsonl')]:
                retain(path)
            times = [r['time'] for r in records]
            group_table.append(dict(seed=seed, group=group, successes=sum(r['success'] for r in records),
                                    mean_seconds=mean(times), median_seconds=float(np.median(times)),
                                    training_successes=sum(r['success'] for r in train),
                                    training_wall_seconds=train[-1]['wall_seconds'],
                                    peak_rss_mib=read(directory / 'status.json')['peak_rss_mib']))
            for start in range(0, 1000, 100):
                block = train[start:start+100]
                curves.append(dict(seed=seed, group=group, episode_start=start+1, episode_end=start+100,
                    success_fraction=mean([r['success'] for r in block]), mean_capped_time=mean([r['time'] for r in block])))
            print('Verified seed %d %s' % (seed, group), flush=True)
        del common

    # Explicitly pair by full map identity, not only array position.
    reference = [r['map'] for r in data[101, 'B0_fifo']]
    assert all([r['map'] for r in data[key]] == reference for key in data)
    for stage in stages:
        p, c = stage['physical'], stage['communication']
        assert p['wall_crossings'] == 0 and p['radio_and_time_caps_passed']
        assert p['coverage_union_and_per_segment_speed_recomputed']
        assert c['rollouts'] == p['rollouts'] and c['per_sender_capacity_passed'] and c['transmission_and_delivery_totals_match']
        for k in total_physics: total_physics[k] += p[k]
        total_radio_events += c['events']
    assert total_physics['rollouts'] == 15900

    group_stats = {}
    for g in GROUPS:
        times = [mean([r['time'] for r in data[s,g]]) for s in SEEDS]
        successes = [sum(r['success'] for r in data[s,g]) for s in SEEDS]
        group_stats[g] = dict(seed_means=times, mean_of_seed_means=mean(times), sample_sd_seed_means=statistics.stdev(times),
                              seed_successes=successes, total_successes=sum(successes), total_evaluations=60)
        close(mean(times), summary['aggregate'][g]['mean_of_seed_means'])
        close(statistics.stdev(times), summary['aggregate'][g]['sample_std_seed_means'])

    pair_stats, paired_csv = {}, []
    for first, second in PAIRS:
        key = first + '_vs_' + second
        per_seed, all_differences = {}, []
        for seed in SEEDS:
            a, b = data[seed,first], data[seed,second]
            differences = [x['time']-y['time'] for x,y in zip(a,b)]
            stats = describe_difference(differences)
            both = [x['time']-y['time'] for x,y in zip(a,b) if x['success'] and y['success']]
            stats.update(percent_mean_time_reduction=100*mean(differences)/mean([r['time'] for r in a]),
                         both_successful_count=len(both), both_successful_mean_seconds_saved=mean(both),
                         slower_maps=[i for i,d in enumerate(differences) if d < -EPS],
                         largest_loss_map=int(np.argmin(differences)), largest_gain_map=int(np.argmax(differences)),
                         differences=differences)
            if key in summary['seeds'][str(seed)]['pairs']:
                expected=summary['seeds'][str(seed)]['pairs'][key]
                close(stats['mean_seconds'],expected['mean_capped_time_saved_seconds'])
                close(stats['median_seconds'],expected['median_capped_time_saved_seconds'])
                assert stats['faster']==expected['second_faster_maps']
                np.testing.assert_allclose(differences,expected['per_map_time_saved_seconds'],atol=1e-8)
            per_seed[str(seed)] = stats
            all_differences.append(differences)
            for i,d in enumerate(differences):
                paired_csv.append(dict(contrast=key,seed=seed,map_index=i,seconds_saved=d,first_time=a[i]['time'],second_time=b[i]['time'],first_success=a[i]['success'],second_success=b[i]['success']))
        matrix=np.asarray(all_differences)
        seed_effects=matrix.mean(axis=1)
        aggregate=describe_difference(matrix.ravel())
        aggregate.update(seed_mean_seconds_saved=seed_effects.tolist(), sample_sd_seed_effects=float(seed_effects.std(ddof=1)),
            percent_reduction_of_grand_means=100*mean(seed_effects)/group_stats[first]['mean_of_seed_means'],
            mean_of_seed_percentage_reductions=mean([v['percent_mean_time_reduction'] for v in per_seed.values()]),
            seed_averaged_map_summary=describe_difference(matrix.mean(axis=0)),
            leave_one_seed_out_mean_range=[float((seed_effects.sum()-seed_effects).min()/2),float((seed_effects.sum()-seed_effects).max()/2)])
        pair_stats[key]=dict(per_seed=per_seed, descriptive_aggregate=aggregate)

    interactions={}
    for seed in SEEDS:
        vals=np.array([r['time'] for r in data[seed,'A_fifo']])+np.array([r['time'] for r in data[seed,'B_learned']])-np.array([r['time'] for r in data[seed,'B0_fifo']])-np.array([r['time'] for r in data[seed,'AB_learned']])
        close(mean(vals),summary['seeds'][str(seed)]['interaction_seconds'])
        interactions[str(seed)]={**describe_difference(vals), 'per_map_seconds': vals.tolist()}
    interaction_means=[v['mean_seconds'] for v in interactions.values()]

    mechanism={}
    for group in GROUPS:
        per_seed={}
        for seed in SEEDS:
            rs=data[seed,group]
            per_seed[str(seed)]=dict(mean_total_distance=mean([r['total_distance'] for r in rs]),
                mean_team_wait_seconds=mean([sum(r['wait_seconds']) for r in rs]),
                mean_team_wait_without_new_map_seconds=mean([sum(r['wait_without_new_map_seconds']) for r in rs]),
                mean_team_sync_idle_seconds=mean([sum(r['synchronization_idle_seconds']) for r in rs]),
                mean_union_completion_time=mean([r['union_completion_time'] if r['union_completion_time'] is not None else 256 for r in rs]),
                mean_task_minus_union_time=mean([r['time']-(r['union_completion_time'] if r['union_completion_time'] is not None else 256) for r in rs]),
                mean_sent_bytes=mean([r['sent_bytes'] for r in rs]),
                mean_connected_seconds=mean([r['connected_seconds'] for r in rs]),
                ratio_sum_novel_cells_per_sent_byte=sum(r['novel_received_cells'] for r in rs)/sum(r['sent_bytes'] for r in rs),
                duplicate_fraction_of_received_cells=sum(r['duplicate_received_cells'] for r in rs)/sum(r['novel_received_cells']+r['duplicate_received_cells'] for r in rs))
        mechanism[group]=dict(per_seed=per_seed, mean_of_seed_values={key:mean([v[key] for v in per_seed.values()]) for key in per_seed['101']})

    result=dict(status='passed', analysis_scope='Retained fixed-budget development endpoints; no new training/policy evaluation',
        seeds=list(SEEDS), maps_per_seed=20, independent_training_seeds=3, test_policy_evaluations_added=0,
        groups=group_stats, contrasts=pair_stats, interaction=dict(formula='T_A + T_B - T_B0 - T_AB; positive means super-additive time saving',
            per_seed=interactions, mean_seconds=mean(interaction_means),sample_sd_seed_means=statistics.stdev(interaction_means)),
        mechanism=mechanism, development_failures=failures,
        statistics_note='Same 20 development maps across 3 training seeds; 60 seed-map outcomes are not 60 independent seeds. Sample SD is not a confidence interval. No significance claim.',
        totals=dict(sac_updates=327000, supervised_head_updates=15000, training_side_rollouts=15600, development_rollouts=300),
        physical_audits=total_physics, communication_events_audited_by_pipeline=total_radio_events,
        newly_recomputed_development_audits=verified_development, newly_recomputed_common_stage_audits=common_stage_rechecks, model_checks=model_checks,
        source_identity_files_verified=len(active), original_source_snapshot_preserved=True,
        recovery_note='Four isolated audit/resume source changes reviewed on 2026-10-14; completed B0 model/evaluation retained with zero extra updates/evaluations.')
    write('supplemental_analysis.json', result)
    write('analysis_input_hashes.json', {'files':input_hashes})
    csv_out('endpoint_metrics.csv',metrics_csv);csv_out('endpoint_by_seed.csv',group_table)
    csv_out('paired_differences.csv',paired_csv);csv_out('training_blocks.csv',curves)
    write('final_readonly_review.json',dict(status='passed',source_files=54,original_snapshot_files=len(original),
        development_rows_recomputed=300,training_rows_checked=15000,sac_update_rows_checked=300000,
        head_updates_from_stage_audits=15000,movement_updates_from_stage_audits=27000,
        head_update_rows_checked=15000,movement_sac_update_rows_checked=27000,
        paired_maps_identical=True,summary_matches_raw_records=True,physical=total_physics,
        newly_recomputed_development_audits=verified_development,models_checked=15,
        extra_training_updates=0,extra_policy_evaluations=0,common_stage_rollouts_reaudited=600,training_audit_scope='Existing adaptation full physical/radio audits inspected and aggregated; all 600 common-stage and 300 development rollouts reaudited from retained trajectories/maps/events.',
        protected_previous_manifest_verified=True,input_files_hashed=len(input_hashes)))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#777777','#e09f3e','#3182bd','#169b62','#9163b8']
    fig,ax=plt.subplots(figsize=(9,4.6))
    for seed,marker in zip(SEEDS,('o','s','^')):
        ax.plot(range(5),[group_stats[g]['seed_means'][SEEDS.index(seed)] for g in GROUPS],marker=marker,alpha=.55,label='Seed '+str(seed))
    av=[group_stats[g]['mean_of_seed_means'] for g in GROUPS]
    ax.scatter(range(5),av,c='black',s=65,marker='D',label='Mean of 3 seeds',zorder=5)
    ax.set_xticks(range(5),GROUPS);ax.set_ylabel('Mean capped task time (simulation seconds)')
    ax.set_title('Fixed endpoints: 20 shared development maps per training seed')
    ax.legend(ncol=2);ax.grid(axis='y',alpha=.2);save_figure(fig,'independent_seed_endpoints')

    fig,axes=plt.subplots(1,3,figsize=(13.2,4.7),sharey=True)
    heat_pairs=[('B0_fifo','AB_learned'),('B_learned','AB_learned'),('AB_uncoupled','AB_learned')]
    for ax,(a,b) in zip(axes,heat_pairs):
        matrix=np.asarray([pair_stats[a+'_vs_'+b]['per_seed'][str(s)]['differences'] for s in SEEDS]).T
        im=ax.imshow(matrix,cmap='RdBu',vmin=-120,vmax=120,aspect='auto')
        ax.set_xticks(range(3),[str(s) for s in SEEDS]);ax.set_xlabel('Training seed')
        ax.set_title(a+' minus AB_learned');ax.set_yticks(range(20),range(20))
    axes[0].set_ylabel('Development map index');fig.colorbar(im,ax=axes.ravel().tolist(),label='Seconds saved by AB (positive = faster)',shrink=.85)
    save_figure(fig,'paired_map_savings')

    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,label in zip(axes,('mean_total_distance','mean_team_wait_seconds','ratio_sum_novel_cells_per_sent_byte'),('Team path length (map units)','Team waiting time (seconds)','Novel received cells / sent byte')):
        values=[mechanism[g]['mean_of_seed_values'][key] for g in GROUPS]
        ax.bar(range(5),values,color=colors)
        ax.set_xticks(range(5),GROUPS,rotation=35,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Descriptive mechanism measures; equal weight per seed')
    fig.tight_layout();save_figure(fig,'mechanism_measures')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for group,color in zip(GROUPS,colors):
        for ax,key in zip(axes,('success_fraction','mean_capped_time')):
            arr=np.asarray([[row[key] for row in curves if row['seed']==s and row['group']==group] for s in SEEDS])
            ax.plot(np.arange(100,1001,100),arr.mean(axis=0),label=group,color=color)
            ax.fill_between(np.arange(100,1001,100),arr.min(axis=0),arr.max(axis=0),alpha=.08,color=color)
    axes[0].set_ylabel('Training success fraction (100-map block)');axes[0].set_ylim(0,1.02)
    axes[1].set_ylabel('Mean capped training task seconds');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Adaptation episode');ax.grid(alpha=.2)
    fig.suptitle('Changing policies and map blocks; shading = seed range, not uncertainty')
    fig.tight_layout();save_figure(fig,'training_progress')
    print(json.dumps({g:group_stats[g] for g in GROUPS},ensure_ascii=False,indent=2))
    print('READ-ONLY ANALYSIS PASSED', flush=True)

if __name__ == '__main__':
    main()
