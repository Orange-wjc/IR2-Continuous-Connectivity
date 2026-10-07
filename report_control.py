"""Independent paired analysis of baseline and corrected evaluation JSONL."""
import argparse, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/connectivity_control_20261006'
BASE=ROOT/'reproduction_results/pretrained_stage2_20261005'
KINDS=('corridor','hybrid','complex')

def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l] if path.exists() else []

def key(row): return row['map_type'],int(row['map_index'])

def intervals(values,seed):
    if not values: return None
    a=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    means=np.mean(a[rng.integers(0,len(a),size=(5000,len(a)))],axis=1)
    return [float(v) for v in np.percentile(means,[2.5,97.5])]

def exact_mcnemar(lost,gained):
    n=lost+gained
    return min(1.0,2*sum(math.comb(n,i) for i in range(min(lost,gained)+1))/2**n) if n else 1.0

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--partial',action='store_true'); args=parser.parse_args()
    baseline=read_rows(BASE/'episodes.jsonl'); controls=read_rows(OUT/'both/episodes.jsonl'); maps_only=read_rows(OUT/'map_only/episodes.jsonl')
    old={key(r):r for r in baseline}; new={key(r):r for r in controls}; map_rows={key(r):r for r in maps_only}
    assert len(baseline)==len(old)==300; assert len(controls)==len(new)
    complete=len(new)==300 and len(map_rows)==6 and (OUT/'full_complete.json').exists()
    if not args.partial: assert complete,'Full evaluation has not completed'
    provenance=json.loads((ROOT/'reproduction_results/audit_20261006/audit_provenance.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((BASE/'episodes.jsonl').read_bytes()).hexdigest()==manifest['baseline_sha256']
    assert hashlib.sha256((ROOT/'model/stage2/checkpoint.pth').read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    for filename,digest in provenance['core_sha256'].items(): assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest
    dataset=json.loads((OUT/'dataset_integrity.json').read_text())
    for filename,digest in dataset['sha256'].items(): assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest
    comparisons={}; validation=[]
    for kind in KINDS:
        pairs=[(old[k],n) for k,n in sorted(new.items()) if k[0]==kind]
        for a,b in pairs:
            assert a['seed']==b['seed'] and a['map_file']==b['map_file'] and a['num_robots']==b['num_robots'] and a['step_limit']==b['step_limit']
            assert b['independent_checks_pass'] and b['copy_violations']==0
            assert b['success']==bool(b['execution_ok'] and all(x>=.99 for x in json.loads(b['individual_explored'])))
            if not b['success'] and b['execution_ok']: assert b['steps']==b['step_limit']
            validation.append(key(b))
        lost=sum(a['success'] and not b['success'] for a,b in pairs); gained=sum(not a['success'] and b['success'] for a,b in pairs)
        valid=[(a,b) for a,b in pairs if a['execution_ok'] and b['execution_ok']]
        step_delta=[float(b['steps'])-float(a['steps']) for a,b in valid]
        distance_delta=[float(b['max_distance_coordinate_units'])-float(a['max_distance_coordinate_units']) for a,b in valid]
        success_delta=[100*(int(b['success'])-int(a['success'])) for a,b in pairs]
        def mean(items,k): return statistics.mean(float(r[k]) for r in items) if items else None
        comparisons[kind]=dict(paired_maps=len(pairs),baseline_success=sum(a['success'] for a,b in pairs),corrected_success=sum(b['success'] for a,b in pairs),success_lost=lost,success_gained=gained,corrected_execution_errors=sum(not b['execution_ok'] for a,b in pairs),completed_metric_pairs=len(valid),baseline_steps_mean=mean([a for a,b in valid],'steps'),corrected_steps_mean=mean([b for a,b in valid],'steps'),steps_delta_mean=statistics.mean(step_delta) if step_delta else None,steps_delta_ci95=intervals(step_delta,100+KINDS.index(kind)),baseline_distance_mean=mean([a for a,b in valid],'max_distance_coordinate_units'),corrected_distance_mean=mean([b for a,b in valid],'max_distance_coordinate_units'),distance_delta_mean=statistics.mean(distance_delta) if distance_delta else None,distance_delta_ci95=intervals(distance_delta,200+KINDS.index(kind)),success_delta_percentage_points=statistics.mean(success_delta) if success_delta else None,success_delta_ci95=intervals(success_delta,300+KINDS.index(kind)),mcnemar_exact_two_sided=exact_mcnemar(lost,gained))
    pilot=[]
    for kind,index in manifest['pilot_cases']:
        k=(kind,index)
        if k in new and k in map_rows:
            pilot.append(dict(case=kind+'_'+str(index),baseline=old[k],map_only=map_rows[k],both=new[k]))
    result=dict(complete=complete,count_both=len(new),count_map_only=len(map_rows),paired=comparisons,pilot=pilot,independent_checks_pass=len(validation)==len(controls),unchanged_core_and_weights=True,physical_distance_unit='map-coordinate units',interval_method='5000 paired-map bootstrap resamples, fixed RNG seed; descriptive for tested maps, not independent policy training trials',failure_accounting='all attempts retained for success rate; steps/distance use mutually execution-complete pairs including exploration timeouts')
    target=OUT/('comparison.json' if complete else 'comparison_partial.json'); target.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    lines=['# 相同预训练权重：信息隔离修复前后对照','',f"状态：{'全部完成' if complete else '运行中'}；两处修复 {len(new)}/300 例，仅地图修复 {len(map_rows)}/6 例。",'',
        '基线为此前已核验的 300 次原代码评估；两处修复使用完全相同的 Stage2 权重、地图索引、种子、机器人数量、步数上限与贪心策略。没有重新训练。核心源码和权重哈希核对一致。','',
        '## 修复定义与定向验证','',
        '1. 传感器更新前复制当前个人地图。初始化保持原样；共享的合并地图和缓存图因此成为快照，不会因另一个机器人的扫描而被原地修改。',
        '2. 预测到旧队友位置的假想连接时，只使用自己的个人地图；未知单元沿用信号模型中的阻挡处理。实际物理连接仍使用环境真值。这是保守的可实施修复，未知区域的估计方式可以有其他选择。','',
        '两处修复通过全部 8 项定向测试；仅地图修复通过 7 项，仍未通过隐藏地形对旧位置清除的隔离测试。与未修复版的 6/8 项通过形成对照。',
        '完整方法差异见 both.patch / map_only.patch，运行时在独立 Env 子类上应用，原 Env 文件未编辑。','',
        '## 配对评估结果','',
        '成功率分母包含所有尝试；执行错误与达到步数上限的探索失败分别计数。步数与路程使用两边均正常执行完成的配对案例，包含超时失败。','',
        '| 场景 | 完成配对数 | 原实现成功 | 修复后成功 | 丢失 / 新增成功 | 修复后执行错误 |',
        '|---|---:|---:|---:|---:|---:|']
    for kind,r in comparisons.items(): lines.append(f"| {kind} | {r['paired_maps']} | {r['baseline_success']} | {r['corrected_success']} | {r['success_lost']} / {r['success_gained']} | {r['corrected_execution_errors']} |")
    lines += ['', '| 场景 | 原平均步数 | 修复后平均步数 | 原平均最大路程 | 修复后平均最大路程 |', '|---|---:|---:|---:|---:|']
    def fmt(v): return f'{v:.2f}' if v is not None else '—'
    for kind,r in comparisons.items(): lines.append(f"| {kind} | {fmt(r['baseline_steps_mean'])} | {fmt(r['corrected_steps_mean'])} | {fmt(r['baseline_distance_mean'])} | {fmt(r['corrected_distance_mean'])} |")
    lines += ['', '路程单位为地图坐标单位，取团队中累计路径最长的机器人；未将通信模型的物理标定疑点默认为已解决。均值不是只对成功案例计算。', '',
        '## 6 例的分项对照','',
        '这 6 例按场景与成功/失败覆盖选择，不是随机抽样，不能用它们推算全部地图的成功率。用它们观察仅地图修复与进一步消除旧位置真值依赖的差别。', '',
        '| 案例 | 原实现 成功/步数 | 仅地图修复 成功/步数 | 两处修复 成功/步数 |', '|---|---|---|---|']
    def cell(r): return ('成功' if r['success'] else '失败')+'/'+str(r['steps'])+('（执行错误）' if not r['execution_ok'] else '')
    for r in pilot: lines.append(f"| {r['case']} | {cell(r['baseline'])} | {cell(r['map_only'])} | {cell(r['both'])} |")
    lines += ['', '## 证据范围','',
        '每个新案例保存完整轨迹、逐动作事件、通信分组和最终个人地图，并从原 PNG 真值独立核算距离、步数、覆盖率、错误自由单元。运行时检查传感器输入不与任何个人或缓存地图共用内存。已完成案例的独立核算与复制隔离检查均通过。',
        '这里验证的是已识别的两类信息泄漏及其修复，不等于排除了所有可能的实现问题。成绩变化也不能作为作者造假的证据；修复导致输入分布变化，而发布的权重是在原环境中训练的。',
        'comparison.json 提供配对地图 bootstrap 95% 区间和成功状态变化的精确 McNemar 检验；这些是固定测试集上的配对分析，不是多次独立训练的置信区间。若只有部分结果，应等待全部测试后解释总体变化。','',
        '## 文件','',
        '- manifest.json：配置、权重与代码哈希、基线来源。',
        '- both/episodes.jsonl 与 episodes.csv：300 例两处修复记录。',
        '- map_only/：6 例仅地图修复记录。',
        '- 每例 JSON 与 maps.npz：轨迹、事件、独立核算、最终个人地图。',
        '- unit_tests/both、unit_tests/map_only：定向测试记录。',
        '- run.log：评估进度与错误。','']
    report=OUT/('对照报告.md' if complete else '当前进度.md'); report.write_text('\n'.join(lines))
    print(json.dumps(dict(complete=complete,both=len(new),map_only=len(map_rows),paired=comparisons,report=str(report)),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
