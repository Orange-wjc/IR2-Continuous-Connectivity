"""Summarize the completed fixed development evaluation without new rollouts."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/ir2-review-mpl'
from pathlib import Path
import json
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
location = json.loads((WORK / 'evaluation_location.json').read_text())
OUT = Path(location['output'])
r = json.loads((OUT / 'results.json').read_text())
assert json.loads((OUT / 'status.json').read_text())['status'] == 'complete'
EP = r['easy_episodes_per_method']
GROUPS = ('IR2_FIFO', 'AB_learned')
rows = {}
old = {}
motion = {}
for group in GROUPS:
    directory = ROOT / group / ('validation_%08d' % EP)
    records = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    rows[group] = {x['map_index']: x for x in records}
    motion[group] = []
    assert len(rows[group]) == len(records) == 100
    old[group] = json.loads((ROOT / group / 'validation_00000512/summary.json').read_text())
    for index in range(100):
        with gzip.open(ROOT / group / 'validation_00000512/episodes' / ('%08d.json.gz' % (index+1)), 'rt') as handle:
            prior = json.load(handle)
        assert prior['metric']['map']['sha256'] == rows[group][index]['map']['sha256']
        with gzip.open(directory / 'episodes' / ('%08d.json.gz' % (index+1)), 'rt') as handle:
            current_trace = json.load(handle)
        steps = current_trace['trace']['trajectory'][1:]
        zero = [all(float(v) <= 1e-9 for v in step['requested_lengths']) for step in steps]
        run = longest = 0
        for stopped in zero:
            run = run + 1 if stopped else 0
            longest = max(longest, run)
        motion[group].append(dict(map_index=index, success=rows[group][index]['success'],
            decisions=len(steps), both_zero_motion_decisions=sum(zero),
            last20_both_zero_motion=sum(zero[-20:]), longest_both_zero_motion_run=longest))

paired = []
for index in range(100):
    a, b = [rows[group][index] for group in GROUPS]
    assert a['map']['sha256'] == b['map']['sha256']
    paired.append(dict(map_index=index, source_file=a['map']['source_file'],
        baseline_success=a['success'], AB_success=b['success'],
        baseline_time=a['time'], AB_time=b['time'],
        capped_time_saving=a['capped_time']-b['capped_time'],
        both_success_time_saving=(a['time']-b['time']) if a['success'] and b['success'] else None))
(OUT / 'paired_results.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2)+'\n')
both = [x for x in paired if x['both_success_time_saving'] is not None]
savings = np.array([x['both_success_time_saving'] for x in both])
distribution = dict(scope='Descriptive common-success fixed-development pairs, not independent training seeds',
    both_success_maps=len(both), AB_faster=int((savings>1e-8).sum()),
    AB_slower=int((savings< -1e-8).sum()), ties=int((np.abs(savings)<=1e-8).sum()),
    mean_seconds_saved_by_AB=float(savings.mean()) if both else None,
    median_seconds_saved_by_AB=float(np.median(savings)) if both else None,
    most_AB_regression=sorted(both,key=lambda x:x['both_success_time_saving'])[:5],
    largest_AB_improvement=sorted(both,key=lambda x:x['both_success_time_saving'],reverse=True)[:5],
    AB_only_success_maps=[x for x in paired if x['AB_success'] and not x['baseline_success']])
(OUT/'paired_distribution_review.json').write_text(json.dumps(distribution,indent=2)+'\n')
a, b = [r['groups'][g] for g in GROUPS]
diagnostics = {}
for group in GROUPS:
    selected = [rows[group][x['map_index']] for x in both]
    diagnostics[group] = dict(
        both_success_mean_wait_seconds_per_robot=float(np.mean([np.mean(x['wait_seconds']) for x in selected])) if both else None,
        both_success_mean_decisions=float(np.mean([x['decisions'] for x in selected])) if both else None,
        both_success_mean_delivery_reward_per_robot=float(np.mean([np.mean(x['reward_components']['delivery']) for x in selected])) if both else None,
        own_successes_mean_time=float(np.mean([x['time'] for x in rows[group].values() if x['success']])) if r['groups'][group]['successes'] else None)
(OUT / 'behavior_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
stall_summary = {group: dict(
    failed_maps=sum(not x['success'] for x in motion[group]),
    failed_maps_last20_both_stationary_at_least18=sum(not x['success'] and x['last20_both_zero_motion']>=18 for x in motion[group]),
    failed_maps_both_stationary_run_at_least50=sum(not x['success'] and x['longest_both_zero_motion_run']>=50 for x in motion[group])) for group in GROUPS}
(OUT/'motion_stall_diagnostics.json').write_text(json.dumps(dict(summary=stall_summary, maps=motion),indent=2)+'\n')

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'font.size':11,'svg.fonttype':'none'})
colors = ('#3767A6','#D35E35')
labels = ('基线 IR2_FIFO','改进 AB_learned')
fig, axes = plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
panels = [('100张开发图成功率（%）',[a['successes'],b['successes']]),
          ('失败按3000秒计的平均任务耗时（秒）',[a['mean_capped_time'],b['mean_capped_time']]),
          ('双方均成功的%d张图：平均任务耗时（秒）'%len(both),
           [a['both_success_mean_time'],b['both_success_mean_time']])]
for ax,(title,values) in zip((axes[0,0],axes[0,1],axes[1,0]),panels):
    if not both and ax is axes[1,0]:
        continue
    bars=ax.bar(labels,values,color=colors,width=.55)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value,'%.1f'%value,ha='center',va='bottom')
    ax.set_title(title)
    ax.set_ylim(0,100 if ax is axes[0,0] else max(values)*1.18)
    ax.grid(axis='y',alpha=.15)
ax=axes[1,1]
if both:
    x=np.array([p['baseline_time'] for p in both]);y=np.array([p['AB_time'] for p in both])
    ax.scatter(x,y,color=colors[1],s=32,alpha=.75)
    limit=max(x.max(),y.max())*1.08
    ax.plot([0,limit],[0,limit],color='#777777',linestyle='--',label='两者同样快')
    ax.set_xlim(0,limit);ax.set_ylim(0,limit);ax.legend()
ax.set_title('共同成功图逐图比较：对角线上方表示改进组更慢')
ax.set_xlabel('基线任务耗时（秒）');ax.set_ylabel('改进组任务耗时（秒）');ax.grid(alpha=.15)
fig.suptitle('原easy固定模型开发评估｜两组第%d回合 / 各%d次SAC\n100张同图、贪心动作；不更新模型，原测试集未使用。'%(EP,a['sac_updates']),fontsize=14)
for suffix in ('png','svg'):fig.savefig(OUT/('development_comparison.'+suffix),dpi=160)
plt.close(fig)

positive = b['successes'] >= a['successes'] and b['mean_capped_time'] < a['mean_capped_time']
efficiency = bool(both) and b['both_success_mean_time'] < a['both_success_mean_time']
if positive and efficiency:
    conclusion='本次固定开发集同时支持成功/失败惩罚时间与共同成功图完成效率收益，可作为讨论进入复杂阶段的依据；仍只有一个训练种子、100张重复使用的开发图，不能扩展成最终测试结论。'
elif positive:
    conclusion='本次固定开发集支持成功率/失败惩罚时间的收益，但共同成功图完成效率仍有代价；尚未复现预实验的全面效率优势。建议先核对迁移训练协议及失败/长路径案例，暂不直接推进复杂训练。'
else:
    conclusion='本次固定开发集未同时支持成功率与失败惩罚任务时间优势，尚未复现自生成地图预实验的效率收益。建议保持暂停，先核对迁移训练协议和机制接入，再规划有针对性的对照；本次不能证明A/B机制本身无效。'
change=100*(b['both_success_mean_time']/a['both_success_mean_time']-1) if both else None
body=f'''# 原easy固定检查点开发评估：第{EP}回合

{conclusion}

用户明确要求暂停训练验证。两组均完整保存至{EP}回合、各{a['sac_updates']}次SAC后正常退出；训练STOP与监督PAUSE保留。评估使用归档的完整检查点、固定源码和原100张几何隔离easy开发图，贪心执行。两组各100回合，共200条开发轨迹；新增训练策略执行0、新增SAC0、原测试策略执行0。原训练源码、模型、Adam、回放及数据哈希评估前后不变。评估耗时{r['wall_seconds']/60:.2f}分钟。

| 指标 | IR2_FIFO | AB_learned |
|---|---:|---:|
| 成功数 | {a['successes']}/100 | {b['successes']}/100 |
| 失败按3000秒计的平均任务耗时 | {a['mean_capped_time']:.3f}秒 | {b['mean_capped_time']:.3f}秒 |
| 平均联合覆盖率 | {100*a['mean_union_coverage']:.3f}% | {100*b['mean_union_coverage']:.3f}% |
| 较弱机器人平均局部覆盖率 | {100*a['mean_min_local_coverage']:.3f}% | {100*b['mean_min_local_coverage']:.3f}% |
| 双方都成功的{len(both)}图平均实际耗时 | {a['both_success_mean_time']}秒 | {b['both_success_mean_time']}秒 |
| 失败中联合覆盖已≥99%的数量 | {a['failed_union_already_99']} | {b['failed_union_already_99']} |
| 达到196决策上限的失败数 | {a['failed_decision_limit']} | {b['failed_decision_limit']} |

改进成功数变化{b['successes']-a['successes']:+d}，失败惩罚均耗时降低{r['capped_time_reduction_percent']:.3f}%（负值表示更差）；共同成功图耗时变化{change}%（正值表示更慢）。仅AB成功{r['AB_only_success']}图、仅基线成功{r['FIFO_only_success']}图。不能把联合覆盖率当成功率；失败真实运行时间不能当完成时间。

共同成功图中AB更快{distribution['AB_faster']}张、更慢{distribution['AB_slower']}张、相同{distribution['ties']}张，逐图节省时间的中位数{distribution['median_seconds_saved_by_AB']}秒（负值表示更慢），所以均值退步并非只由一个极端样本造成。共同成功图总路程{a['both_success_mean_distance']}→{b['both_success_mean_distance']}原图像素，平均每机器人WAIT {diagnostics['IR2_FIFO']['both_success_mean_wait_seconds_per_robot']}→{diagnostics['AB_learned']['both_success_mean_wait_seconds_per_robot']}秒；耗时增加与更长路程共同出现，不能只归咎WAIT。完整逐图分布及退步/获益最大的案例保存在paired_distribution_review.json，未剔除异常值重报主成绩。

## 与旧512回合开发评估比较

两次100张地图逐图SHA一致。旧512回合：基线{old['IR2_FIFO']['successes']}/100，改进{old['AB_learned']['successes']}/100；失败惩罚均耗时分别{old['IR2_FIFO']['mean_capped_time']:.3f}/{old['AB_learned']['mean_capped_time']:.3f}秒。当前结果属于同图的新固定模型比较，不再用最近训练窗口替代验证；期间工程修复和训练配置历史保留，不能把这次纵向变化视为只增加训练时长的单因素实验。

## 与自生成地图预实验的差别

原simple_rooms_v1三独立训练种子在20张固定开发图上，AB平均75.410秒、B0平均130.133秒，AB为60/60成功，效率收益42.05%。该历史结论保留，不能直接套用到原图。

本次迁移除了地图/原层次图，训练协议也发生变化：embedding64→128、有效batch16→128、Q学习率1e-4→1e-5、移动预训练9000→216次SAC、消息头采样100→16图、每图适应更新20→8、成功轨迹模仿权重1→0（原预实验更新日志确认存在实际模仿更新，当前NativeLearner为原项目式SAC）。传感范围70→80、邻居12→30、决策/时间预算等也不同。完整差异见migration_configuration_review.json。这些是混合变化，不据本次结果直接归因为地图更难、训练不足或移植缺陷；应在后续工程/协议审计与受控实验中区分。

## 停滞与几何诊断

从本次200条保留轨迹读取每个决策的requested_lengths（两机均为0表示该步均未请求移动）：{stall_summary}。这只是行为描述，不能单独证明训练协议或某个模块的因果作用；逐图字段及最长停滞连续段保留在motion_stall_diagnostics.json。

另做离线原图8连通自由空间检查，100张图的原出生点连通分量均覆盖100%自由格，未发现自由空间天然断开导致99%目标不可能的情况。该连通性只是宽松的点机器人几何上界，不保证当前候选图或有限决策预算可达；真值仅用于离线诊断，不输入策略。见geometry_connectivity_review.json。

## 审计与保留状态

200条轨迹均完成连续运动/覆盖/速度/通信总量和容量审计，墙体内部穿越{r['geometry_wall_crossings']}。100张两组同图哈希一致，开发几何与训练几何分组隔离。60文件源码及原数据身份通过核验，训练检查点SHA未变。评估最低可用RAM {r['min_available_memory_gib']:.2f}GiB，最高占用{r['max_used_memory_gib']:.2f}GiB。

训练保持暂停，自动恢复保持PAUSE，自动评估仍关闭，复杂阶段未启动。latest_comparison.json与pipeline最近评估回合已更新到{EP}；没有改模型、奖励或训练参数，也没有发飞书。完整检查点保存在本目录checkpoints/；每组原始轨迹、终点占据图、逐图指标位于对应组validation_{EP:08d}/。

![固定开发集比较](development_comparison.png)

[统计JSON](results.json) · [逐图配对结果](paired_results.json) · [逐图分布与案例](paired_distribution_review.json) · [行为诊断](behavior_diagnostics.json) · [停滞诊断](motion_stall_diagnostics.json) · [训练配置差异](migration_configuration_review.json) · [评估协议与模型身份](protocol.json)
'''
(OUT/'README.md').write_text(body)
notice=f'''## 最新人工开发评估：2026-10-17，训练暂停于两组{EP}回合

用户要求暂停验证，已暂停监督并让训练完整保存退出；两组{EP}回合、各{a['sac_updates']}次SAC。冻结完整检查点，在相同100张原easy开发图上贪心执行，两组各100条，共200开发轨迹；真实训练/梯度更新/测试执行均0。200轨迹运动/通信审计通过、穿墙0，100张配对图SHA一致，60源码/原数据/训练检查点身份不变。

基线成功{a['successes']}/100、AB{b['successes']}/100，失败按3000秒计均耗时{a['mean_capped_time']:.3f}→{b['mean_capped_time']:.3f}秒（AB降低{r['capped_time_reduction_percent']:.3f}%）。共同成功{len(both)}图真实耗时{a['both_success_mean_time']}→{b['both_success_mean_time']}秒（AB变化{change}%）。联合覆盖{100*a['mean_union_coverage']:.2f}%→{100*b['mean_union_coverage']:.2f}%，较弱局部{100*a['mean_min_local_coverage']:.2f}%→{100*b['mean_min_local_coverage']:.2f}%。仅AB成功{r['AB_only_success']}图、仅基线成功{r['FIFO_only_success']}图。{conclusion}

共同成功图AB快{distribution['AB_faster']}张、慢{distribution['AB_slower']}张，中位节省{distribution['median_seconds_saved_by_AB']}秒；原始逐图分布及所有退步案例保留，不是只看单个极端值。总路程{a['both_success_mean_distance']:.1f}→{b['both_success_mean_distance']:.1f}原图像素；共同成功图平均每机器人WAIT仅{diagnostics['IR2_FIFO']['both_success_mean_wait_seconds_per_robot']:.2f}→{diagnostics['AB_learned']['both_success_mean_wait_seconds_per_robot']:.2f}秒，不能把主要效率代价笼统归咎持续WAIT。原图离线8连通诊断未发现自由空间天然断开，但不保证图候选和有限决策可达。

原自生成图三种子42.05%耗时收益仍保留。此次核对确认迁移伴随多项协议变化：batch16→128、Q LR1e-4→1e-5、移动预训练9000→216次SAC、头采样100→16图、适应每图20→8次SAC、成功示例模仿权重1→0，及原图/层次图/128维模型等。不能把差异仅归于地图或训练不足；后续应优先核对机制/训练协议并做受控对照。

STOP及监督PAUSE均保留，验证后没有恢复训练；自动评估关闭、不启动复杂阶段、不发飞书。最近开发评估已更新，旧512条目只作历史。结果/图表/协议/完整检查点：`{OUT.relative_to(REPO)}/README.md`；原轨迹各组validation_{EP:08d}/。评估脚本`research_work/native_manual_eval_20261017/evaluate.py`，分析脚本同目录analyze.py。

'''
paths=(Path('/home/wyz/wjc_ws/papers/多机器人有限通信探索_实验计划与新对话交接.md'),
       REPO/'finite_contact/EXPERIMENT_HANDOFF.md',REPO/'finite_contact/FEISHU_TASK_CONTEXT.md',
       ROOT/'TRAINING_PLAN.md',REPO/'research_work/native_curriculum_20261015/STATUS.md',
       REPO/'finite_contact/results/native_curriculum_20261015/README.md')
for path in paths:
    value=path.read_text();head,rest=value.split('\n',1)
    marker=notice.splitlines()[0]
    if marker in value:
        start=value.index(marker)
        end=value.find('\n## ',start+len(marker))
        path.write_text(value[:start]+notice.rstrip()+'\n'+(value[end:] if end>=0 else ''))
    else:
        path.write_text(head+'\n\n'+notice+rest.lstrip('\n'))
(WORK/'README.md').write_text('# 原easy人工暂停验证\n\n结果报告：'+str(OUT/'README.md')+'\n\n训练及监督保持暂停，测试执行0。\n\n本工作目录captured_checkpoints是暂停过程中的先行备份（FIFO10060/AB10036，capture_review.json已记录），未用于评估；实际验证使用结果目录checkpoints中的两组10060完整快照。\n')
print(body[:2300])
