"""Document all fixed endpoints once training and audits are complete."""
import json
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parent.parent
FC=ROOT/'finite_contact'
read=lambda p:json.loads(p.read_text())
plan=read(RUN/'replication.json')
assert plan['status']=='complete'
summary=read(RUN/'report/summary.json')
supplement=read(RUN/'checks/descriptive_supplement.json')
names={'A_fifo':'FIFO','A_rule':'规则','AB_learned':'学习'}
labels={'pooled':'原抽样 pooled','map_tail':'候选抽样 map_tail'}
audit=summary['combined_physical_audit'];account=supplement['compute_accounting']
new_audits=[read(Path(r['directory'])/'report/summary.json')['combined_physical_audit'] for r in plan['runs'] if r['seed']!=2]
assert sum(a['rollouts'] for a in new_audits)==1800
stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
fmt=lambda v:f'{v:.3f}'
seedlabel=lambda n:str(n)+('（先行探索）' if n==2 else '')

def table(header,rows):
    return '| '+' | '.join(header)+' |\n| '+' | '.join('---' for _ in header)+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)

endpoint=table(['适应种子','通信方式','pooled成功／20','pooled平均秒','map_tail成功／20','map_tail平均秒'],[
    [seedlabel(r['seed']),names[r['group']],r['pooled']['successes'],fmt(r['pooled']['mean_capped_time']),r['map_tail']['successes'],fmt(r['map_tail']['mean_capped_time'])] for r in supplement['endpoint_rows']])
aggregate=table(['抽样','通信方式','成功数均值±SD／20','平均时间均值±SD（秒）'],[
    [labels[c],names[g],fmt(summary['conditions'][c]['groups'][g]['successes']['mean'])+'±'+fmt(summary['conditions'][c]['groups'][g]['successes']['sample_std']),fmt(summary['conditions'][c]['groups'][g]['mean_capped_time']['mean'])+'±'+fmt(summary['conditions'][c]['groups'][g]['mean_capped_time']['sample_std'])] for c in ('pooled','map_tail') for g in names])
monitor=table(['适应种子','通信方式','pooled：0→50→100图','map_tail：0→50→100图'],[
    [seedlabel(r['seed']),names[r['group']],'→'.join(map(str,r['pooled']['successes'])),'→'.join(map(str,r['map_tail']['successes']))] for r in supplement['training_monitor_rows']])
paired=table(['适应种子','通信方式','候选新增成功图','候选丢失成功图','候选节省平均秒'],[
    [seedlabel(r['seed']),names[g],str(r['map_tail_gained_success_maps']),str(r['map_tail_lost_success_maps']),fmt(r['time_saved_seconds'])] for g,values in supplement['sampling_changes'].items() for r in values])
comparison=table(['抽样','适应种子','学习相对规则多成功张数','学习节省平均秒','平均时间减少比例'],[
    [labels[c],seedlabel(r['seed']),r['success_gain'],fmt(r['time_saved_seconds']),fmt(r['relative_mean_time_reduction_percent'])+'%'] for c,values in supplement['learned_vs_rule'].items() for r in values])
confirm=table(['通信方式','确认种子1／3成功数增量均值','确认种子1／3节省秒数均值'],[
    [names[g],fmt(v['success_gain']['mean']),fmt(v['time_saved_seconds']['mean'])] for g,v in supplement['confirmation_only_sampling'].items()])
failrows=[]
for seed,values in supplement['failure_counts'].items():
    for method,v in values.items():
        if v['failed_rollouts']:
            failrows.append([seedlabel(int(seed)),method,v['failed_rollouts'],v['team_exploration_below_threshold'],v['local_map_synchronization_below_threshold']])
failuretable=table(['适应种子','抽样／通信方式','失败数','联合探索不足','局部同步不足'],failrows)
all_faster=all(r['time_saved_seconds']>0 for values in supplement['learned_vs_rule'].values() for r in values)
speed_statement=('在两种抽样的三个固定终点中，学习选包平均时间均低于规则。' if all_faster else '学习相对规则的逐种子时间得失见下表，不能用总体均值掩盖负结果。')
learned_confirm=supplement['confirmation_only_sampling']['AB_learned']
conclusion=('候选收尾抽样没有表现出一致优势：确认种子1、3的规则完成数分别从20/20降到19/20、18/20；种子1学习仍20/20但慢7.545秒；'
            '种子2是先行探索，其正结果须与固定确认种子分开解读。'
            f"确认种子1／3的学习方法平均成功数增量为{learned_confirm['success_gain']['mean']:.3f}张，平均节省时间为{learned_confirm['time_saved_seconds']['mean']:.3f}秒（负数表示变慢）。"
            '保留map_tail作为显式试验选项，默认继续pooled，不宣称稳定性已解决。')

final=f'''\n\n## 完整固定终点结果（{stamp}）

新增12组全部完成；每组在原100张训练图上再完整遍历一次，保留2000次真实SAC更新，到累计12895、Actor Adam=11395；通信头5000及上下文冻结。新保留更新合计24000。已有种子2六组不重训，合并后共18个固定终点。失败按256秒计入下表；成功要求两台机器人各自覆盖率≥99%。

{endpoint}

{conclusion} {speed_statement} 抽样修正属于全部方法共用的训练措施，不能直接当作通信创新贡献。

### 描述统计和确认种子单列

{aggregate}

均值及样本SD以每个适应种子的模型为单位，仅作描述。同一20张开发图反复评估，不是60张独立地图；三个适应种子共享一次预训练和通信头。先行种子2并非事前随机挑选的新确认种子。为避免其正结果掩盖复现结果，单列种子1／3，只有两次确认，仍不作显著性结论。

{confirm}

{comparison}

### 训练监测与全部配对得失

下表为固定前10张训练图在追加0／50／100张采样后的greedy完成数，分母均为10。训练图监测不属于独立泛化证据。种子2候选FIFO的9→7→6等退步全部保留，不以开发改善替换。

{monitor}

开发地图编号均从0开始；负的节省秒数表示候选更慢。配对新增与丢失成功图同时列出：

{paired}

### 全部开发失败重算

从保存的真值与两机最终局部地图独立重算覆盖率，区分团队尚未探索足够区域与团队已达到99%但一机尚未同步达标。零失败条件仍在完整JSON中保留。终点缺口分类不能证明模块因果。

{failuretable}

### 预算、物理与来源审计

新增两种子主结果1800条完整执行轨迹（1200训练、360训练监测、240开发）；与先行种子2合并，主结果{audit['rollouts']}条完整轨迹、{audit['movement_segments']}段移动，穿墙0，覆盖重算、通信容量、速度与时限检查通过。成对初始监测是重复执行，不能作为独立样本。

OOM恢复另有10张重做训练图、200次废弃后重做的实际更新，以及4条被重新执行的完整开发轨迹。额外14条执行／1924段移动审计通过，所有重做的非墙钟指标、完整轨迹、原始数组和200条更新记录与归档一致。新训练实际计算24200次更新，最终模型保留24000；合并完整执行{account['combined_complete_executions_including_recovery']}条、{account['all_complete_movement_segments']}段移动，仍穿墙0。未完成的中断评估不算完整轨迹。

12组源状态、缓存规模、实际网络变化、通信头和上下文冻结、相同100图调度、成对首次完整训练轨迹均已核对；Actor／双Q／温度Adam所有参数的实际步数各增加2000。执行训练源码快照与恢复前／运行后身份一致。原始编排及恢复编排分别保存；允许的变化是并发3→2及外层恢复记录，训练逻辑不变。原种子2计划与汇总、原三种子适应计划SHA256未改变。全量64项行为检查通过（70.424秒）；测试策略执行0。

### 模型、证据和下一步

- 新固定终点：`experiments/stability_replication_20261012/seed_<1或3>/<pooled或map_tail>/<A_fifo或A_rule或AB_learned>/final.pth`；同组`latest.pth`包含回放、优化器与RNG，供完整续训。
- 既有种子2：`experiments/stability_20261012/<pooled或map_tail>/<组>/final.pth`，不覆盖原结果。
- 完整汇总：`experiments/stability_replication_20261012/report/summary.json`；各seed的`report/failure_analysis.json`、`checks/pair_integrity.json`保留全部失败和核对。
- 小型证据：[results/stability_replication_20261012](results/stability_replication_20261012/)，含原始完整JSON复制、两种编排身份、恢复核对、测试日志、图表与SHA256清单；权重、回放及原始轨迹仅保留本机。

两机简单地图的真实训练和开发评估链路已跑通；当前仍属于开发诊断，尚不能宣称最终泛化、全链路独立种子收敛或论文创新成立。下一阶段优先固定原抽样及评估协议，在相同预算下做会合收益反馈／内容选择的模块消融与通信容量对照，确认收益来源；在方法和选模规则固定后，再一次性进行留出测试。暂不因单个结果增加复杂地图或机器人数量，也不自动启用候选抽样。
'''

p=FC/'STABILITY_REPLICATION_EXPERIMENT.md';s=p.read_text()
assert '## 完整固定终点结果' not in s
s=s.replace('## 目的和范围','当前状态：**新增12组训练、最终开发评估和审计全部完成**。最终结果见文末“完整固定终点结果”；前面的启动与恢复记录按当时时点保留。\n\n## 目的和范围',1)
p.write_text(s+final)

stage=f'''## 最新阶段：收尾抽样的另两适应种子复现（2026-10-12，已完成）

conda `ir2_train` 中新增seed=1、3的pooled／map_tail × FIFO／规则／学习12组全部完成。每组从原10895完整状态继续，在原100张训练图上完整遍历一次、保留2000次SAC，固定终点12895；新增保留24000次。通信头5000及上下文冻结，两机器人、100/20/20简单地图和任务标准不变；测试策略执行0。

{endpoint}

{conclusion} {speed_statement} 训练辅助同样用于全部基线，不能直接作为通信创新。三个适应种子仍共享预训练／通信头，反复使用同一20开发图；种子2是探索，种子1／3是固定条件确认。全部训练监测及开发失败保留，包含种子2候选FIFO由9/10降到6/10。

64项行为检查通过。主结果合并{audit['rollouts']}条完整轨迹、{audit['movement_segments']}段移动，穿墙0；新增主结果1800条轨迹。12:32系统OOM后从完整断点恢复，实际并发降为2，算法与固定预算不变。额外10图／200次更新及4条开发执行与旧归档逐条一致，额外14条轨迹／1924段移动单列。最终源状态、全部Adam真实步数、冻结头、配对首次轨迹、地图调度及源码身份审计通过。

完整结果与范围见 `IR2-Continuous-Connectivity/finite_contact/STABILITY_REPLICATION_EXPERIMENT.md`；小型证据 `finite_contact/results/stability_replication_20261012/`。新模型位于 `experiments/stability_replication_20261012/seed_<1或3>/<条件>/<组>/final.pth`，完整续训用同组latest.pth；种子2保留原目录。下一步优先固定原抽样，在同预算下做模块消融及容量对照；协议固定后再执行留出测试，不据局部正结果自动增加训练或切换默认。

'''
for p in (FC/'RESEARCH_DESIGN.md',ROOT.parent/'papers/有限通信窗口下多机器人探索_研究与实验设计.md'):
    s=p.read_text();start=s.index('## 最新阶段：收尾抽样的另两适应种子复现');end=s.index('## 实验运行环境（已确认）',start)
    p.write_text(s[:start]+stage+s[end:])

learning={r['seed']:r for r in supplement['endpoint_rows'] if r['group']=='AB_learned'}
learned_counts='／'.join(str(learning[n]['map_tail']['successes']) for n in (1,2,3))
headline=(f'收尾抽样的固定条件复现已完成：新增seed=1、3共12组，每组保留2000次真实SAC；与先行seed=2合并，候选学习的三个适应种子分别完成{learned_counts}张（每轮20张）。'
          '但规则在确认种子1／3分别由20/20降到19/20、18/20，种子1学习同为20/20却慢7.545秒，候选没有一致优势，默认仍pooled。'
          f'64项行为检查和{audit["rollouts"]}条主轨迹物理审计通过；OOM后以两并发完整恢复，额外计算单列，测试策略执行0。')
for p,link in ((ROOT/'README.md','finite_contact/STABILITY_REPLICATION_EXPERIMENT.md'),(FC/'README.md','STABILITY_REPLICATION_EXPERIMENT.md')):
    s=p.read_text().replace('最新收尾抽样试验已完成：','前一轮收尾抽样试验已完成：')
    lines=s.splitlines();matches=[i for i,line in enumerate(lines) if line.startswith('下一轮固定条件复现已启动：')]
    assert len(matches)==1
    lines[matches[0]]=headline+f'全部结果、模型与范围见[收尾抽样复现]({link})。'
    p.write_text('\n'.join(lines)+'\n')

p=FC/'VALIDATION.md';s=p.read_text().replace('## 2026-10-12：另两种子复现启动与资源恢复检查','## 2026-10-12：另两种子固定终点复现与资源恢复检查（已完成）')
s=s.replace('当前12个新任务仍运行中，测试策略执行0；不将检查通过或局部开发成功当作完整复现结论。见[STABILITY_REPLICATION_EXPERIMENT.md](STABILITY_REPLICATION_EXPERIMENT.md)。',
    f'12个新任务全部完成，源状态、全部Actor／双Q／温度Adam参数真实新增2000步、头冻结、配对首次轨迹、100图调度及源码身份核对通过。新增主轨迹1800条，合并先行种子2后为{audit["rollouts"]}条／{audit["movement_segments"]}段移动，穿墙0，覆盖／通信／速度／时限审计通过。包含恢复额外执行为2714条，额外计算不计入最终模型预算。\n\n{conclusion} 测试策略执行0，共享预训练和重复使用的20开发图仍限制泛化结论。全部固定终点与训练监测见[STABILITY_REPLICATION_EXPERIMENT.md](STABILITY_REPLICATION_EXPERIMENT.md)，证据见[results/stability_replication_20261012](results/stability_replication_20261012/)。')
p.write_text(s)
p=FC/'CHANGELOG.md';s=p.read_text().replace('## 工作区改动：收尾抽样的种子1／3复现（2026-10-12，运行中、尚未提交）','## 工作区改动：收尾抽样的种子1／3复现（2026-10-12，已完成、尚未提交）')
s=s.replace('当前仅部分固定终点完成，完整研究结论待12组全部结束，默认pooled不变、测试策略不执行。双份设计更新v1.7；事前设计与运行／恢复记录见[STABILITY_REPLICATION_EXPERIMENT.md](STABILITY_REPLICATION_EXPERIMENT.md)。',
    headline+' 新12组保留24000次更新、实际计算含重做24200次；预算、来源和优化器审计通过。双份设计v1.7已同步完整结果与负结果；新增含SHA256清单的小型证据。事前设计与运行／恢复记录见[STABILITY_REPLICATION_EXPERIMENT.md](STABILITY_REPLICATION_EXPERIMENT.md)。')
p.write_text(s)
p=FC/'STABILITY_MULTIMAP_EXPERIMENT.md';s=p.read_text()
s += '\n\n后续seed=1、3的固定条件复现已全部完成，候选并未在所有条件下一致改善；包括种子1规则成功数减少、学习变慢等结果。先行seed=2的正结果应结合[完整复现记录](STABILITY_REPLICATION_EXPERIMENT.md)解读，默认仍pooled，测试未执行。\n'
p.write_text(s)
# Both design copies must contain exactly the same latest-stage block.
blocks=[]
for p in (FC/'RESEARCH_DESIGN.md',ROOT.parent/'papers/有限通信窗口下多机器人探索_研究与实验设计.md'):
    s=p.read_text();blocks.append(s[s.index('## 最新阶段：收尾抽样的另两适应种子复现'):s.index('## 实验运行环境（已确认）')])
assert blocks[0]==blocks[1]
print(json.dumps({'documented':True, 'design_mirrors_equal':True, 'updated_at':stamp},ensure_ascii=False))
