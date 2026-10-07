"""Paired analysis with independent layouts as bootstrap blocks."""
import argparse,hashlib,json,statistics
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/independent_20261006'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values,seed):
    if not values:return None
    a=np.asarray(values,float);rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),ci95=[float(v) for v in np.percentile(means,[2.5,97.5])],layout_blocks=len(a))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for name,digest in protocol['core_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in protocol['code_sha256'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'model/stage2/checkpoint.pth')==protocol['checkpoint_sha256']
    for layout in protocol['layouts']:assert sha(ROOT/layout['map_path'])==layout['map_sha256']
    path=OUT/'episodes.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    lookup={(r['layout_index'],r['condition'],r['method']):r for r in rows}
    assert len(lookup)==len(rows)
    expected={(l['layout_index'],c['name'],m) for l in protocol['layouts'] for c in protocol['conditions'] for m in protocol['methods']}
    assert set(lookup)<=expected
    complete=set(lookup)==expected and (OUT/'complete.json').exists()
    if not args.partial:assert complete,'Study incomplete'
    for r in rows:
        assert r['independent_checks_pass'] and r['copy_violations']==0
        assert r['success']==bool(r['execution_ok'] and all(v>=.99 for v in r['individual_explored']))
        assert r['num_robots']==5 and r['step_limit']==384
        c=next(c for c in protocol['conditions'] if c['name']==r['condition'])
        assert r['X_g']==c['X_g'] and r['K']==c['K']
        if r['execution_ok'] and not r['success']:assert r['steps']==384
    per_condition={};pairs={};paired_success=[];paired_steps=[];fully_successful_layouts=Counter()
    def avg(items,key):return statistics.mean(r[key] for r in items) if items else None
    for c in protocol['conditions']:
        name=c['name'];per_condition[name]={};condition_pairs=[]
        for method in protocol['methods']:
            group=[r for r in rows if r['condition']==name and r['method']==method];valid=[r for r in group if r['execution_ok']]
            success=[r for r in valid if r['success']];fail=[r for r in valid if not r['success']]
            per_condition[name][method]=dict(attempts=len(group),successes=len(success),success_rate=len(success)/len(group) if group else None,execution_errors=len(group)-len(valid),exploration_timeouts=len(fail),steps_execution_ok_mean=avg(valid,'steps'),distance_success_mean=avg(success,'max_distance_coordinate_units'),distance_timeout_mean=avg(fail,'max_distance_coordinate_units'),min_coverage_mean=avg(valid,'min_explored'),connectivity_mean=avg(valid,'full_connectivity_fraction_per_action'))
        for l in protocol['layouts']:
            index=l['layout_index'];a=lookup.get((index,name,'ir2'));b=lookup.get((index,name,'frontier_rendezvous'))
            if a is None or b is None:continue
            assert a['seed']==b['seed'] and a['map_path']==b['map_path'] and a['layout_sha256']==b['layout_sha256']
            raw_a=json.loads((OUT/'raw'/(a['case_id']+'.json')).read_text());raw_b=json.loads((OUT/'raw'/(b['case_id']+'.json')).read_text())
            assert raw_a['initial_positions']==raw_b['initial_positions']
            condition_pairs.append((a,b));paired_success.append((index,name,100*(int(a['success'])-int(b['success']))))
            if a['execution_ok'] and b['execution_ok']:paired_steps.append((index,name,a['steps']-b['steps']))
        joint=[(a,b) for a,b in condition_pairs if a['success'] and b['success']]
        pairs[name]=dict(paired=len(condition_pairs),ir2_only_success=sum(a['success'] and not b['success'] for a,b in condition_pairs),baseline_only_success=sum(b['success'] and not a['success'] for a,b in condition_pairs),success_delta_pp=bootstrap([100*(int(a['success'])-int(b['success'])) for a,b in condition_pairs],100+len(pairs)),joint_success_pairs=len(joint),joint_success_distance_delta=bootstrap([a['max_distance_coordinate_units']-b['max_distance_coordinate_units'] for a,b in joint],200+len(pairs)))
    def block_average(values):
        blocks={}
        for index in range(30):
            group=[v for i,c,v in values if i==index]
            if len(group)==3:blocks[index]=statistics.mean(group)
        return list(blocks.values())
    stability={}
    for method in protocol['methods']:
        successes=[per_condition[c['name']][method]['successes'] for c in protocol['conditions']]
        all_three=sum(all(lookup.get((i,c['name'],method),{}).get('success',False) for c in protocol['conditions']) for i in range(30))
        stability[method]=dict(successes_by_condition=successes,layouts_successful_in_all_three=all_three,screen_pass=bool(min(successes)>=24 and max(successes)-min(successes)<=4) if complete else None)
    result=dict(complete=complete,episodes=len(rows),paired_conditions=sum(p['paired'] for p in pairs.values()),per_condition=per_condition,paired=pairs,aggregate_success_delta_pp_layout_bootstrap=bootstrap(block_average(paired_success),300),aggregate_steps_delta_layout_bootstrap=bootstrap(block_average(paired_steps),301),stability=stability,validation='source/weights/maps/protocol hashes; episode uniqueness; paired identities and initial positions; independent accounting; action guards and copy isolation',interpretation='IR2 minus baseline; success delta positive favors IR2; distance/steps delta negative favors IR2; distance intervals condition on both succeeding; layout bootstrap is descriptive, not independent training variability')
    (OUT/('comparison.json' if complete else 'comparison_partial.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# 修正后的 IR2：独立布局与通信条件验证','',f"状态：{'全部完成' if complete else '运行中'}，{len(rows)}/180 次评估，完成 {result['paired_conditions']}/90 组配对。",'',
        '30 个不同 Complex 障碍布局，未发现与公开训练集的精确/旋转/镜像重合。每个布局固定原起点和 5 个机器人，在 X_g=K=0、6.5、13 三种通信衰减条件下比较已发布 Stage2 策略和前沿/会合启发式基线。两种方法均使用两处信息隔离修复；实际物理通信、传感器、图构建和 384 回合上限一致，没有训练。','',
        '实验方案和代码/输入哈希在首个评估结果出现前冻结。没有按成功结果选地图或重试失败。通信条件是固定敏感性试验，不是随机重复训练或三个独立噪声种子。','',
        '## 每种条件','',
        '| 条件 | 方法 | 已完成 | 成功 | 执行错误 | 探索超时 | 平均步数（正常执行） |',
        '|---|---|---:|---:|---:|---:|---:|']
    def fmt(value):return '—' if value is None else f'{value:.2f}'
    for name,methods in per_condition.items():
        for method,r in methods.items():lines.append(f"| {name} | {method} | {r['attempts']} | {r['successes']} | {r['execution_errors']} | {r['exploration_timeouts']} | {fmt(r['steps_execution_ok_mean'])} |")
    lines += ['', '执行错误和探索超时均进入成功率分母；正常执行的步数包含探索超时。路径较短但未完成探索不能作为效率提高的证据。', '',
        '## 配对比较','', '| 条件 | 已配对 | 仅 IR2 成功 | 仅基线成功 | 双方均成功 | 双成功路径差均值（IR2−基线） |', '|---|---:|---:|---:|---:|---:|']
    for name,r in pairs.items():lines.append(f"| {name} | {r['paired']} | {r['ir2_only_success']} | {r['baseline_only_success']} | {r['joint_success_pairs']} | {fmt(r['joint_success_distance_delta']['mean'] if r['joint_success_distance_delta'] else None)} |")
    lines += ['', '距离以地图坐标计；通信物理尺度疑点未解决，不把距离默认为米。只有双方都成功的病例进入主路径比较，其区间是条件性描述，不能替代全部病例的成功率。', '',
        '总体成功率差使用布局块 bootstrap：每次抽取布局并保留它的三种通信条件，避免把同一布局的三个条件当作独立新地图。每个条件内的配对结果及 95% 区间见 comparison.json。', '',
        '## 稳定性工程筛查','',
        '预先规定：每种条件至少成功 24/30 例，条件间最大成功数差不超过 4/30。这是本次工程判断标准，不是领域通用标准，也不保证现实部署。']
    for method,r in stability.items():lines.append(f"- {method}：各条件成功数 {r['successes_by_condition']}；三种条件均成功的布局 {r['layouts_successful_in_all_three']}/30；筛查结果 {r['screen_pass'] if complete else '等待全批完成'}。")
    lines += ['', '## 范围与限制','',
        '- 启发式基线只读取本机观测和自身访问历史：沿已知图选择最近可达正前沿效用节点；无前沿时向最近可达的已知队友节点会合；仍无目标时选择最少访问的合法邻居。它不是论文的 Pursuit/Preplanned，也不是所有强基线。',
        '- 原始权重是在原环境中训练，信息隔离修复会改变输入分布。修复中的未知地形通信估计保守，其他可实施估计也可能合理。',
        '- 30 个布局是有限公开验证集，部分布局此前出现过描述性测试；不得继续在其上调参并将其作为最终独立 holdout。未排除近似重复，也无法证明权重实际训练时完全未见。',
        '- 这里只改变通信衰减；起点和团队规模未改变，没有验证物理单位标定、真实足迹碰撞、Gazebo 或实机表现。',
        '- bootstrap 反映布局采样的描述性不确定性；不显著不表示方法等效，不是多次独立训练的置信区间。','',
        '## 文件','', '- protocol.json / protocol.sha256：冻结方案、地图及代码/权重哈希。', '- episodes.jsonl / episodes.csv：全部评估，包括失败。', '- raw/：路径、通信分组、决策检查与最终个人地图。', '- comparison.json：配对统计和布局 bootstrap。', '- validation.json：全批结束后的独立原始数据复核。','']
    (OUT/('独立布局验证报告.md' if complete else '当前进度.md')).write_text('\n'.join(lines))
    print(json.dumps(dict(complete=complete,episodes=len(rows),paired_conditions=result['paired_conditions'],stability=stability),ensure_ascii=False))


if __name__=='__main__':main()
