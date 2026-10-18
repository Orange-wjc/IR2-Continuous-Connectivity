#!/usr/bin/env python3
"""Produce Chinese interpretation and export figures from the verified analysis."""
import csv
import json
import os
from pathlib import Path
import shutil
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-independent-analysis-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'experiments/independent_seeds_20261013'
OUT=REPO/'finite_contact/results/independent_seeds_20261013'
D=json.loads((OUT/'supplemental_analysis.json').read_text())
SEEDS=(101,202,303)
GROUPS=('B0_fifo','A_fifo','B_learned','AB_learned','AB_uncoupled')

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def save(name,text):
    (OUT/name).write_text(text.strip()+'\n')

def main():
    assert D['status']=='passed'
    for src,dst in [('report/RESULTS_ANALYSIS.md','pipeline_RESULTS_ANALYSIS.md'),('report/summary.json','pipeline_summary.json'),('pipeline.json','pipeline_final.json'),('source_identity_after.json','source_identity_after.json')]:
        shutil.copy2(ROOT/src,OUT/dst)
    fig,axes=plt.subplots(4,2,figsize=(10,16))
    for axes_row,failure in zip(axes,D['development_failures']):
        seed,index=failure['seed'],failure['map_index']
        for ax,group in zip(axes_row,('B0_fifo','AB_learned')):
            directory=ROOT/f'seed_{seed}'/group/'validation_final'
            record=rows(directory/'evaluation.jsonl')[index]
            trace=rows(directory/'evaluation_trajectories.jsonl')[index]
            with np.load(directory/f'evaluation_map_{index}_repeat_0.npz') as saved:
                free=saved['truth']==255
                seen=(saved['beliefs']==255)&free
                union=seen.any(axis=0)
                worst=int(np.argmin(record['coverage']))
                rgb=np.full(free.shape+(3,),.23)
                rgb[free]=[1.,.78,.28]
                rgb[seen[worst]]=[.96,.96,.96]
                rgb[union&~seen[worst]]=[.95,.40,.12]
                ax.imshow(rgb,origin='upper')
            pos=np.asarray([step['positions'] for step in trace['trajectory']])
            for robot,color in enumerate(('#1f77b4','#11aa77')):
                ax.plot(pos[:,robot,0],pos[:,robot,1],color=color,lw=.7,alpha=.65)
                ax.scatter(*pos[-1,robot],c=color,s=13)
            ax.set_title(f'Seed {seed}, map {index}, {group}\nT={record["time"]:.1f}s; union={100*record["union_coverage"]:.2f}%; min local={100*min(record["coverage"]):.2f}%',fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('All four failed baseline endpoints and matched AB endpoints\nYellow: unexplored free cells; orange: explored but missing at worst robot',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('png','pdf'):fig.savefig(OUT/f'baseline_failure_cases.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

    group_lines=[]
    for group in GROUPS:
        a=D['groups'][group]
        vals=a['seed_means']
        group_lines.append(f'| {group} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {a["mean_of_seed_means"]:.3f} ± {a["sample_sd_seed_means"]:.3f} | {a["total_successes"]}/60 |')
    pair_lines=[]
    for key,v in D['contrasts'].items():
        a=v['descriptive_aggregate']
        pair_lines.append(f'| {key.replace("_vs_"," → ")} | {a["mean_seconds"]:.3f} | {a["percent_reduction_of_grand_means"]:.2f}% | {a["median_seconds"]:.3f} | {a["faster"]}/{a["slower"]}/{a["ties"]} |')
    seed_pair_lines=[]
    for key,v in D['contrasts'].items():
        if key=='B0_fifo_vs_AB_learned':continue
        cells=[]
        for seed in SEEDS:
            a=v['per_seed'][str(seed)]
            cells.append(f'{a["mean_seconds"]:.3f} / {a["median_seconds"]:.3f} / {a["faster"]}:{a["slower"]}')
        seed_pair_lines.append('| '+key.replace('_vs_',' → ')+' | '+' | '.join(cells)+' |')
    fail_lines=[]
    for a in D['development_failures']:
        typ='联合探索未达标' if a['union_coverage']<.99 else '本地地图同步未达标'
        fail_lines.append(f'| {a["seed"]} / {a["map_index"]} | {100*a["union_coverage"]:.3f}% | {100*min(a["coverage"]):.3f}% | {a["unexplored_free_cells"]} | {a["explored_cells_missing_at_worst_robot"]} | {typ} |')
    mech_lines=[]
    for g in GROUPS:
        a=D['mechanism'][g]['mean_of_seed_values']
        mech_lines.append(f'| {g} | {a["mean_total_distance"]:.1f} | {a["mean_team_wait_seconds"]:.3f} | {a["mean_union_completion_time"]:.3f} | {a["mean_task_minus_union_time"]:.3f} | {a["ratio_sum_novel_cells_per_sent_byte"]:.5f} |')
    phys=D['physical_audits']
    report=f'''# 完整独立三种子实验：结果复核与结论

本轮于北京时间2026年10月15日14:15完成。本报告从保留的原始记录重新计算固定终点成绩，核对模型、预算和审计，未新增训练或策略评估。

## 结论先行

在当前两机器人、simple_rooms_v1、通信范围65地图单位、每发送端2048字节/秒的固定开发集条件下，完整AB在三个独立训练种子上均取得五组最低的平均任务时间，且全部成功。跨种子平均任务时间75.410秒，相对B0的130.133秒缩短42.05%，相对去耦合AB的83.795秒缩短10.01%。这支持完整方法在本协议下的效率优势，以及耦合机制的平均增益。

需要收紧两点表述：A单独使用不是每个种子都获益；耦合增益也不是每张图、每个种子同样稳定。2×2交互效应符号不一致，不能据此声称稳定的超加性协同。此次仍是反复使用的20张开发图，尚无独立测试、带宽扫描、其他地图族或多于两机器人的证据。

## 协议与方法含义

- 三个训练种子101、202、303独立初始化、预训练和训练消息头；同一种子内五组共享本种子的共同起点。每种子移动预训练100图/9000次SAC，消息头在100训练图采样后监督5000步；五组各1000训练图/新增20000次SAC，固定终点累计29000次。消息头和上下文编码器在适应阶段冻结。旧轮A/FIFO起点与本轮中性B0/FIFO起点不同，不混合排名。
- B0_fifo也是学习的移动策略，保留基础通信、地图差异会合特征和相同训练目标，不是无通信或随机策略。A_fifo用预计可交付收益及距离成本构造会合价值；B_learned使用学习选包；AB_learned组合两者，并把学习选包的预计收益反馈给会合价值。
- AB_uncoupled执行端仍使用学习选包，但会合收益按FIFO估计，因此对比考察收益估计和实际内容选择的一致性，不是关闭消息传输。A包含收益特征和距离成本的整体变化，不是单独对某一个标量做因果隔离。
- 每个固定终点在完全相同的20张开发图上贪心执行一次。成功要求两个机器人的本地自由格覆盖率均达到99%；失败统一记256秒。时间为仿真任务时间，距离为地图单位；地图没有米制标定。
- 总保留预算15600条训练侧轨迹、327000次SAC、15000次消息头监督更新、300条开发评估；留出测试策略执行0。预检和工程探针另列，不计入此预算。

## 固定终点成绩

| 方法 | seed101秒 | seed202秒 | seed303秒 | 三种子均值 ± 样本标准差（秒） | 合计成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(group_lines)}

这里的±表示三个“各自在20图上的均值”的样本标准差，不是标准误或置信区间。60条种子×地图记录只包含三个独立训练种子、20张共用地图，不能当作60个独立训练重复。

![各种子固定终点](independent_seed_endpoints.png)

AB相对B0在三个种子上分别节省14.20%、54.19%、47.07%。AB的种子均值标准差6.271秒小于其他四组，但重复数只有3，只能描述本次观察到的离散程度，不能保证未来初始化同样稳定。

## 模块贡献与逐图配对

下表均以“前者任务时间−后者任务时间”计节省，正数为后者更快。节省比例定义为“三种子等权总体均值差 / 前者总体均值”，不是每个种子百分比的算术平均；两种口径同时保存在机器可读结果。快/慢/平是60条配对结果的描述计数，不作为独立样本显著性检验。B0→AB是为展示整体效果补充的描述性对比，其余五项及交互为开跑前固定对比。

| 对比 | 平均节省秒 | 总体均值缩短比例 | 60条配对节省中位数秒 | 后者快/慢/平 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(pair_lines)}

每个种子的配对结果按“平均节省秒 / 中位节省秒 / 更快:更慢图数”列出：

| 对比 | seed101 | seed202 | seed303 |
| --- | ---: | ---: | ---: |
{chr(10).join(seed_pair_lines)}

B单独使用在三个种子均降低平均任务时间，相对B0总体缩短26.38%。在B的基础上加入A及耦合，AB进一步缩短21.29%；在A的基础上使用学习选包及相应耦合，AB进一步缩短30.77%。这两项比较都包含完整方法相应机制的变化，不能分别解释为与上下文无关的固定独立效应。

A单独相对B0总体缩短16.29%，但seed101反而慢0.652秒，双方均成功的19图慢2.317秒；seed202、303分别节省41.112、23.154秒。因此旧轮“A独立收益偏弱”的判断需要更新为“A收益依赖训练种子/共同起点，在本轮两个种子上明显有效，一个种子未改善平均耗时”。不能把A写成无效，也不能写成每次都有效。

![逐图配对差异](paired_map_savings.png)

## 耦合增益、异常值与交互

耦合相对去耦合的平均节省分别为1.961、10.652、12.540秒，对应2.43%、13.51%、13.66%。三个种子均值方向一致，总体节省8.385秒；逐图为11/20、14/20、15/20更快，合计40/60更快、20/60更慢。对每张图先跨三个种子平均，仍只有14/20图更快。

seed101耦合优势较弱：删掉最大收益图map8后，平均节省从1.961变成−0.818秒；该种子的平均优势依赖这张图。seed202、303各自删掉最大收益图后，仍节省8.446、10.325秒。按先跨种子平均的20张地图做同样敏感性分析，去掉最大收益图后总体节省6.655秒；逐次留出一个训练种子，剩余两个种子的平均节省范围6.307～11.596秒。这些都是事后描述性稳健性检查，不是置信区间。

耦合的最差退步案例之一是seed202/map16，完整AB相对去耦合慢38.781秒；seed101/map6慢32.744秒。完整逐图差异、所有退步地图和最大收益/损失图见[配对CSV](paired_differences.csv)与[supplemental_analysis.json](supplemental_analysis.json)。这些案例保留在结果中，没有剔除后重新报主成绩。

2×2交互定义为 `T_A + T_B − T_B0 − T_AB`，正值代表时间节省超出两个单独收益相加。三个种子为+4.395、−6.393、−0.419秒，均值−0.805秒、种子样本标准差5.404秒。符号不一致，因此本轮不支持稳定的超加性协同。完整AB平均优于去耦合AB与“正交互”是不同命题，前者有当前结果支持，后者没有。

## 成功率和失败类型

B0在三个种子为19/20、20/20、17/20，合计56/60；其余四组均为60/60。仅出现四条失败，不能以此宣称所有通信条件下都可靠。四条B0失败都达到256秒时限：

| seed / 开发图 | 联合覆盖 | 最差本地覆盖 | 未探索自由格 | 已探索但最差机器人缺失格 | 直接未达标项 |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(fail_lines)}

两条联合探索不足、两条本地地图同步不足；这是终点状态分类，不证明失败的算法因果。seed303/map19同时存在75格已探索但本地缺失，因联合覆盖本身未达标归入探索不足。其余方法在相同图成功，不意味着已经隔离出某个动作造成基线失败。

为了判断收益是否只是失败罚时造成，另看双方均成功图：完整AB相对B0在三个种子仍分别平均节省8.325、80.645、50.599秒。因而本轮总体效率优势并非仅来自把四条失败变成功；这一辅助口径也有条件选择限制，主结果仍保留全部失败。

![全部四条基线失败及匹配AB终点](baseline_failure_cases.png)

图中黄色表示未探索自由格，橙色表示联合已探索但最差本地机器人尚未获得的格，白色为该机器人已知自由格；两条线为实际保留轨迹。展示的是全部四条基线失败，没有只挑其中一例。

## 机制指标：支持解释，但不证明因果

| 方法 | 团队总路程 | 两机等待秒之和 | 联合达到99%时间秒 | 任务−联合达标时间秒 | 新增接收格/发送字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(mech_lines)}

各项先在种子内20图取均值，再在种子间等权平均。联合未达标时该项时间封顶256秒，“任务−联合时间”仅为时间差代理，不是精确的通信瓶颈因果分解。“新增格/发送字节”在种子内先求总量之比，再在三个种子间平均；不等于单图比率均值或传输协议净载荷效率。

完整AB相对去耦合，团队路程1740.1→1600.6，等待5.442→4.209秒，任务与联合达标时间差20.330→13.176秒。这与减少路径和后续同步拖延的解释一致。但seed101的AB路程反而略长（1692.8→1703.6），seed303的AB联合覆盖达标更晚（58.899→67.675秒），说明平均收益不能归结为所有阶段都更快。

学习内容选择相对FIFO总体提高新增信息/字节，但完整AB该比率0.05053低于去耦合0.05416，不能宣称完整AB在字节效率上也最好。seed303完整AB发送字节也多于去耦合，任务更快和总通信更少不是同一结论。

![描述性机制指标](mechanism_measures.png)

## 训练过程与机器适配

每组严格1000图、20000次新增SAC，未因开发成绩延长训练或挑选checkpoint。最后100张训练图中，AB成功率在三个种子为99%、100%、100%；但训练地图和策略随时间变化，不是固定模型的验证曲线，不能据此证明收敛或把它与开发贪心成绩直接比较。

![训练分块曲线](training_progress.png)

图为每100回合分块、跨种子均值，阴影仅为三个种子的最小至最大范围。各组完整训练进程峰值RSS最高约3.04GiB，未再因内存不足中断。训练中曾发生一次几何审计误判导致停机，修复审计并补审后恢复，不能写成本轮从未中断。

## 验收与证据边界

- 逐条检查15000条适应训练记录和300000条新增SAC更新日志；重新加载15组初始/最终模型，核对同种子初始状态完全一致、最终Adam步数29000、头/上下文及头优化器冻结；所有300条开发记录与原自动汇总一致。
- 新增只读复核重新执行全部300条开发轨迹的几何、覆盖、速度、时限、通信事件与容量审计，并对600条共同预训练/监督采样轨迹统一重审；其他15000条适应训练轨迹沿用已完成的全量流水线审计并检查结果。
- 合计保留{phys['rollouts']}条轨迹、{phys['movement_segments']}段移动，进入障碍格内部0，边界/角点接触{phys['boundary_contacts']}；通信流水线已核对66136671条事件。几何模型为连续点对像素内部，不能外推成有限半径实体机器人有碰撞裕量。
- 活动源码54文件与完成后身份和隔离快照一致；最初快照另行保留。seed101的共同阶段旧审计没有边界接触字段，本次从原始记录统一重算，没有把缺字段当作零。
- 10月14日审计修复与恢复有单独版本和补丁；B0保留模型、原始记录哈希不变，补审新增训练更新/策略评估均0。详见[恢复记录](PHYSICAL_AUDIT_RECOVERY_20261014.md)。
- 当前只增加结果分析，训练源码、模型与原自动报告均保留；新增策略执行0、测试执行0。复现入口、输入哈希和通过日志见[复核记录](final_readonly_review.json)、[复现说明](analysis_reproduction.json)。

## 论文可以写到哪里

可写：在固定简单室内图和单一受限通信配置下，三个完整独立训练种子的结果支持完整方法降低开发集任务完成时间；学习内容选择具有跨种子一致的平均收益；使会合收益估计与内容选择一致，相对去耦合配置有平均时间优势，但存在地图和种子依赖。

暂不能写：已证明普遍泛化、统计显著优于所有方法、稳定超加性协同、所有带宽/地图/机器人数量均有效、字节利用率全场景最佳，或学习选包多种子优于规则选包。本轮没有规则组，旧轮规则结果只能保留为旧协议的补充证据。

接下来建议先用冻结的15个终点做预先固定的带宽条件扫描，区分当前配置优势与跨带宽鲁棒性；详细执行范围、判据和预算已整理在[NEXT_STEPS.md](NEXT_STEPS.md)。该工作尚未启动，留出测试继续保留。
'''
    save('HUMAN_REVIEW.md',report)
    save('PAPER_CONCLUSIONS.md','''# 可用于论文或组会的结论草稿

在两机器人simple_rooms_v1任务中，我们使用三个完整独立训练种子，对会合价值、学习内容选择及其耦合进行固定预算验证。每个种子独立完成中性B0/FIFO预训练和消息头监督学习，再从同一种子内共同起点训练五种配置。在相同20张开发图、每个机器人本地覆盖率达到99%的标准下，完整方法在三个种子均取得最低平均任务完成时间，合计60/60成功；其跨种子平均时间为75.410±6.271秒，相对FIFO基线130.133±33.183秒降低42.05%，相对去耦合配置83.795±7.010秒降低10.01%。±为三个种子均值的样本标准差。

学习内容选择单独使用时，三个种子的平均耗时均优于基线，总体降低26.38%。会合价值单独使用的效果依赖种子：两个种子改善，一个种子的平均耗时略有退步。完整耦合相对去耦合在三个种子的平均收益分别为1.961、10.652、12.540秒，但60条种子×地图配对中仍有20条退步；其中seed101的优势对一张高收益图敏感。因此结果支持在当前协议下的平均效率收益，不支持逐图一致改进。

2×2交互T_A+T_B−T_B0−T_AB在三个种子分别为+4.395、−6.393、−0.419秒，尚不支持稳定的超加性协同。机制指标显示完整方法总体伴随更短团队路程和更少等待，但新增信息/字节未优于去耦合版本，不能将任务效率收益等同于字节效率最优。

这些结果来自反复使用的20张开发图和单一通信配置（范围65地图单位、每发送端2048字节/秒），训练重复数仅3。当前尚无留出测试、跨带宽、其他地图族、多机器人数量或多种子规则选包对照，不能据此宣称普遍泛化或统计显著性。旧轮采用不同共同起点，不与本轮混合排名。
''')
    save('NEXT_STEPS.md','''# 下一步实验建议（待用户决定，未启动）

当前最需要补齐的是有限通信条件改变后优势是否仍成立，而不是继续延长已完成的训练。建议先做冻结终点的带宽扫描，预算小于重新训练，且能直接检验核心应用条件。随后根据结果决定是否需要带宽随机化训练或新的地图族，最终再锁定协议执行留出测试。

## 建议固定范围

- 使用本轮全部3种子×5方法的15个固定final.pth；不挑最佳种子、不重新训练、不调整消息头、阈值、奖励或等待上限。
- 开发图仍为相同20张，范围65地图单位、速度12、时限256秒、成功阈值每机99%，保持贪心执行和原地图/起点。
- 每发送端带宽取512、1024、2048、4096字节/秒。2048作为本轮原始300条评估的只读基准，明确标记复用；三个新条件合计3×15×20=900条新开发评估，测试策略执行0。
- 属于冻结模型的零样本带宽迁移；执行时依据新带宽更新合法观测和预计接触预算。不把它称作在各带宽重新训练后的最优性能。训练时只见2048字节/秒，低/高带宽结果受分布变化影响。
- 运行前审查当前评估入口能否只覆盖带宽参数、无学习调用、保持地图/种子配对与逐字节容量审计。另建实验目录、冻结计划/源码身份，单worker串行；不能覆盖本轮validation_final。

## 事前比较与判断

主对比仍是AB与B0、AB与B、AB与去耦合；逐种子报告成功数、失败计256秒均值、中位配对差及地图快慢。展示四个带宽下各方法曲线，不在事后只报告最有利带宽。辅助记录发送量、新增信息/字节、等待、队列、本地/联合覆盖和路径。

若低带宽优势消失或成功率下降，报告这种边界，并在查看测试集之前提出单独的带宽随机化训练计划；若多个条件仍保持优势，可把结论扩展到实际验证的带宽集合。不能因某个带宽结果差而临时增加训练或修改主指标。三训练种子仍不作强显著性或普遍鲁棒性结论。

## 留出测试的安排

20张留出测试图尚未执行策略。应在方法、比较对象和最终协议锁定后一次性对全部预定种子/方法评估，不只测开发集最佳checkpoint，不用测试结果调参。测试图数量与地图族仍有限，不能用一次测试替代跨地图分布验证。

本文件只形成下一轮可审阅方案。本次结果分析没有启动训练、开发扫描或测试评估，也不发送飞书消息。
''')
    save('README.md','''# 完整独立三种子实验：已完成并复核

训练与自动评估于北京时间2026年10月15日14:15完成。用户随后授权结果分析，全部15组已从原始记录复核；当前没有启动下一轮。飞书主动通知保持关闭。

主要入口：[完整分析与限制](HUMAN_REVIEW.md)、[论文结论草稿](PAPER_CONCLUSIONS.md)、[下一步建议（未启动）](NEXT_STEPS.md)。

| 方法 | 三种子平均任务秒 ± 种子样本标准差 | 合计成功 |
| --- | ---: | ---: |
| B0_fifo | 130.133 ± 33.183 | 56/60 |
| A_fifo | 108.928 ± 17.098 | 60/60 |
| B_learned | 95.810 ± 11.513 | 60/60 |
| AB_learned | 75.410 ± 6.271 | 60/60 |
| AB_uncoupled | 83.795 ± 7.010 | 60/60 |

完整AB相对基线缩短42.05%，相对去耦合缩短10.01%；三个种子均值方向一致，但逐图不一致，seed101耦合增益对最大收益图敏感。A单独收益随种子变化，2×2交互符号不一致，不支持稳定超加性结论。均为重复使用的20张开发图，三个独立训练种子，测试策略执行0；不能将60条种子×地图记录视为60个独立种子。失败计256秒。

固定计划见[PLAN.md](PLAN.md)。每种子从随机初始化做中性B0/FIFO预训练9000次SAC、训练消息头5000步，再配对训练五组各新增20000次SAC、固定终点29000；不能与旧A/FIFO起点协议混合排名。

证据文件：

- [原自动汇总](pipeline_summary.json)、[原自动报告](pipeline_RESULTS_ANALYSIS.md)、[最终流水线状态](pipeline_final.json)。原件仍在experiments/independent_seeds_20261013/report，不覆盖。
- [补充分析JSON](supplemental_analysis.json)、[逐图指标CSV](endpoint_metrics.csv)、[逐种子终点CSV](endpoint_by_seed.csv)、[全部配对差CSV](paired_differences.csv)、[训练分块CSV](training_blocks.csv)。
- [只读复核](final_readonly_review.json)、[输入SHA256](analysis_input_hashes.json)、[复现入口](analysis_reproduction.json)、manifest.json。完整模型与大体积轨迹仍在原实验目录，本目录为小型结果包。
- [各种子终点图](independent_seed_endpoints.pdf)、[逐图配对图](paired_map_savings.pdf)、[机制图](mechanism_measures.pdf)、[训练曲线](training_progress.pdf)、[全部基线失败案例](baseline_failure_cases.pdf)，均另有PNG。
- [几何审计修复记录](PHYSICAL_AUDIT_RECOVERY_20261014.md)：保留原源码快照，活动身份为source_identity_audit_resume_20261014.json。本次复核统一重审600条共同阶段与300条开发轨迹，其他适应轨迹使用已完成的流水线全量审计。

合计15900条保留轨迹、1349336段移动，进入障碍格内部0，边界/角点接触1008。几何为连续点对像素内部，不等于有限半径机器人安全性。训练进程峰值最高约3.04GiB，未再OOM；曾因几何审计误判停机，修复并补审后完成。
''')
    print('REPORTS WRITTEN')

if __name__=='__main__':main()
