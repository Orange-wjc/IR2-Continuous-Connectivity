"""Read-only post hoc summaries and compact exports; never loads or runs a policy."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent.parent
DEST = ROOT/'finite_contact/results/stability_replication_20261012'
GROUPS = ('A_fifo', 'A_rule', 'AB_learned')
CONDITIONS = ('pooled', 'map_tail')

def read(p):
    return json.loads(p.read_text())

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def describe(values):
    assert all(math.isfinite(v) for v in values)
    return {'per_seed': values, 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values)}

plan = read(RUN/'replication.json')
assert plan['status'] == 'complete', 'Wait for all training and automatic audits to complete'
summary = read(RUN/'report/summary.json')
runs = sorted(summary['per_run'], key=lambda r:r['seed'])
assert [r['seed'] for r in runs] == [1,2,3]
assert summary['combined_physical_audit']['rollouts'] == 2700
assert summary['combined_physical_audit']['wall_crossings'] == summary['test_policy_evaluations'] == 0
identity = read(RUN/plan['source_identity_file'])
assert identity == read(RUN/'checks/source_identity_after.json')
assert all(digest(ROOT/name) == sha for name,sha in identity.items())

result = {'scope': 'Post hoc descriptive supplement after every predeclared endpoint; no extra policy executions or model selection.',
          'confirmation_only_seeds':[1,3], 'test_policy_evaluations':0,
          'endpoint_rows':[], 'training_monitor_rows':[], 'sampling_changes':{},
          'learned_vs_rule':{}, 'failure_counts':{},
          'warning':'Shared pretraining and repeated development maps; two confirmation seeds are not enough for a reliable significance claim.'}

for run in runs:
    seed = run['seed']; directory = Path(run['directory'])
    failure = read(directory/'report/failure_analysis.json')
    result['failure_counts'][str(seed)] = failure['counts']
    for group in GROUPS:
        a,b = [run['groups'][group][c] for c in CONDITIONS]
        assert all(v['new_sac_updates'] == 2000 and v['cumulative_sac_updates'] == 12895 for v in (a,b))
        assert all(v['actor_adam_steps'] == 11395 and v['frozen_head_updates'] == 5000 for v in (a,b))
        values = [v['validation_final'] for v in (a,b)]
        result['endpoint_rows'].append({'seed':seed, 'group':group, 'pooled':values[0], 'map_tail':values[1]})
        result['training_monitor_rows'].append({'seed':seed, 'group':group,
            **{c:{'successes':[run['groups'][group][c]['training_monitor'][str(n)]['successes'] for n in (0,50,100)],
                  'mean_capped_times':[run['groups'][group][c]['training_monitor'][str(n)]['mean_capped_time'] for n in (0,50,100)]} for c in CONDITIONS}})
        result['sampling_changes'].setdefault(group,[]).append({'seed':seed,
            'success_gain':values[1]['successes']-values[0]['successes'],
            'time_saved_seconds':values[0]['mean_capped_time']-values[1]['mean_capped_time'],
            **failure['paired_conditions'][group]})
    for c in CONDITIONS:
        rule, learned = [run['groups'][g][c]['validation_final'] for g in ('A_rule','AB_learned')]
        result['learned_vs_rule'].setdefault(c,[]).append({'seed':seed,
            'success_gain':learned['successes']-rule['successes'],
            'time_saved_seconds':rule['mean_capped_time']-learned['mean_capped_time'],
            'relative_mean_time_reduction_percent':100*(1-learned['mean_capped_time']/rule['mean_capped_time'])})

result['confirmation_only_sampling'] = {g:{field:describe([r[field] for r in result['sampling_changes'][g] if r['seed'] in (1,3)])
    for field in ('success_gain','time_saved_seconds')} for g in GROUPS}
recovery = read(RUN/'checks/recovery_equivalence.json')
assert recovery['status'] == 'passed'
extra = recovery['extra_executions_physical_audit']
result['compute_accounting'] = {'new_retained_training_maps':1200, 'new_retained_sac_updates':24000,
    'discarded_replayed_training_maps':10, 'discarded_replayed_sac_updates':200,
    'new_actual_sac_updates_including_discarded':24200,
    'pilot_retained_sac_updates':12000, 'combined_retained_continuation_updates':36000,
    'combined_actual_continuation_updates_including_discarded':36200,
    'combined_primary_rollouts':2700, 'additional_complete_rollouts':14,
    'combined_complete_executions_including_recovery':2714,
    'primary_movement_segments':summary['combined_physical_audit']['movement_segments'],
    'additional_movement_segments':extra['movement_segments'],
    'all_complete_movement_segments':summary['combined_physical_audit']['movement_segments']+extra['movement_segments'],
    'wall_crossings':0}
write(RUN/'checks/descriptive_supplement.json', result)

sources = {
 'replication.json': RUN/'replication.json', 'summary.json':RUN/'report/summary.json',
 'descriptive_supplement.json':RUN/'checks/descriptive_supplement.json',
 'pre_run_design.md':RUN/'pre_run_design.md',
 'orchestration_source.py':RUN/'orchestration_source.py',
 'source_identity_before.json':RUN/'checks/source_identity_before.json',
 'source_identity_resume.json':RUN/plan['source_identity_file'],
 'source_identity_after.json':RUN/'checks/source_identity_after.json',
 'pilot_worker_equivalence.json':RUN/'checks/pilot_worker_equivalence.json',
 'resource_interruption.json':RUN/'checks/resource_interruption.json',
 'recovery_equivalence.json':RUN/'checks/recovery_equivalence.json',
 'oom_kernel.log':RUN/'checks/oom_kernel.log',
 'behaviour_64.log':RUN/'checks/behaviour_64.log',
 'background_process_original.json':RUN/'background_process.json',
 'background_process_resumed.json':RUN/'background_process_current.json',
 'sampling_seeds.png':RUN/'report/sampling_seeds.png',
 'sampling_seeds.pdf':RUN/'report/sampling_seeds.pdf',
 'finalize_evidence.py':Path(__file__).resolve(),
 'write_final_docs.py':RUN/'checks/write_final_docs.py',
}
resume_stamp = Path(plan['source_identity_file']).stem.replace('source_identity_resume_', '')
sources['orchestration_resume.py'] = RUN/('orchestration_resume_'+resume_stamp+'.py')
for run in runs:
    seed=run['seed']; d=Path(run['directory'])
    for target,source in [('plan.json','plan.json'),('summary.json','report/summary.json'),
                           ('failure_analysis.json','report/failure_analysis.json'),
                           ('pair_integrity.json','checks/pair_integrity.json'),
                           ('train_monitor.png','report/train_monitor.png')]:
        sources['seed_%d/'%seed+target]=d/source
    for pic in (d/'report').glob('*_failures.png'):
        sources['seed_%d/'%seed+pic.name]=pic
DEST.mkdir(parents=True, exist_ok=True)
for name,source in sources.items():
    target=DEST/name;target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source,target)
    assert digest(source)==digest(target)
readme = '''# 收尾抽样三适应种子的描述性复现证据

seed=2是既有探索试验，seed=1、3为固定条件复现；新12组全部完成，未重训seed=2。每组从自己的10895完整状态继续2000次SAC，固定终点12895；通信头5000且冻结。主结果按适应种子汇总，共用同一预训练链和20张开发图，不能解释为60张独立地图。

`summary.json`完整保留三种子、两抽样、三通信方法及训练监测。`descriptive_supplement.json`为事后只读汇总，列出全部配对得失、训练监测退步、种子1/3单独描述和额外计算。不能只使用探索种子2的正结果，也不能把续训收益全归因于新抽样。

各`seed_*`目录包含完整汇总、失败重算、配对初始化及真实优化器步数核对。主结果2700条完整执行轨迹；OOM恢复另有14条完整执行和200次重做更新，单列而不计入最终模型预算。恢复轨迹、数组及更新记录逐条一致。实际并发由3降至2，训练算法与固定预算不变。

首次和恢复后编排源码分别保留；source_identity_resume与after一致，实际训练文件身份未变。manifest提供所有复制文件的SHA256；manifest不包含自身。64项行为检查通过；测试策略执行0。

本目录不包含模型、回放和原始轨迹；原始数据在本机`experiments/stability_replication_20261012/`及先行`experiments/stability_20261012/`，引用的绝对路径无法在远端直接使用。默认仍为pooled。完整结果见[实验记录](../../STABILITY_REPLICATION_EXPERIMENT.md)。
'''
(DEST/'README.md').write_text(readme)
write(DEST/'manifest.json', {'scope':'Exact copies of retained local evidence; model weights and raw executions excluded.',
                           'files':{str(p.relative_to(DEST)):digest(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
print(json.dumps({'evidence':str(DEST), 'files':len(read(DEST/'manifest.json')['files']),
                  'compute_accounting':result['compute_accounting'],
                  'confirmation_only_sampling':result['confirmation_only_sampling']},indent=2))
