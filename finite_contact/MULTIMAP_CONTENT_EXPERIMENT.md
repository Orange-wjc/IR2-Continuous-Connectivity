# 100 图通信头实验（2026-10-12）

## 运行前确定的方案

环境固定为 `/home/wyz/anaconda3/envs/ir2_train`，两台机器人，使用既有 `simple_rooms_v1` 的训练 100／开发验证 20／测试 20 划分与数据种子 20261011。移动来源为 `experiments/multimap_20261011_vectorized_seed1/movement_A/final.pth`（累计 SAC=8895、head=0），保留已有模型及所有实验。本轮只研究通信头；不增加移动训练，不执行测试策略。

1. 冻结移动 Actor、Q、目标 Q、温度。图上下文由这个已完成多图训练的 Actor 复制一次后冻结。按原确定性打乱顺序，在全部 100 张训练图执行一次 A/FIFO greedy；每 4 个决策采一次双方待发队列标签，每侧最多 64 包，保持原 `spread` 抽取。标签为接收者当前尚未知的单元数／100，接收图只用于训练标签。
2. 每图采用独立随机优先级 reservoir，均匀保留最多 2048 个样本，覆盖该图的前后时刻，避免全局最近 20000 个样本只保留后面少数地图。每个 batch 先等概率选地图，再在该图等概率选样本；batch=16、头学习率 3e-4、seed=1，固定 **5000 次实际监督更新**。全部训练图采集完毕后才开始训练；验证／测试标签不进入样本或训练。
3. 不选择中途最好的头。固定终点权重、同一移动网络，分别在同一组 20 张开发图运行 FIFO、合法规则选包、学习选包。均保留 A 的会合机制，仅替换内容模式；学习评分仍受发送方合法 novelty 上限约束，协议、速度、字节预算、99% 完成标准不变。
4. 重新运行 FIFO，逐记录核对与上轮 18/20 开发结果完全一致。报告成功数、失败计 256 秒的平均耗时、局部最低覆盖、单位发送字节的有效新增量。通信内容会影响收到的地图及会合收益输入，所以“移动网络冻结”并不保证三组实际轨迹相同；不能只用真实轨迹上的接收总量归因于排序。
5. 在预定前 10 张训练图及全部 20 张 FIFO 开发轨迹上，每 32 个决策取固定状态，严格动作回放，比较相同假设四秒接触下 FIFO／规则／学习／理想上界的新增单元；并比较学习预测与训练样本均值常数的误差。验证标签只作离线评价。本诊断不代表实际会合发生。原验收原则继续适用：学习误差不高于常数，且训练和开发状态的学习窗口收益均高于 FIFO，才支持进入下一轮等预算 A／AB 联合微调。

入口 `finite_contact_multimap_content.py` 支持依次执行 collect、train、evaluate、report，也可 `--stage all` 完整运行。输出目录必须新建；单阶段可读取同目录 pipeline 继续下一个阶段，不把不完整数据当成完整采集。训练 100 图每图标签数、逐更新损失、均衡抽样计数、初始／最终权重、物理轨迹和开发逐图结果均保留。多图离线样本在 `collection/labels_*.npz`；`head/latest.pth` 保存模型和优化器，不宣称单独依靠它即可恢复离线采样游标。

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_multimap_content.py \
  --initialize-from experiments/multimap_20261011_vectorized_seed1/movement_A/final.pth \
  --reference-validation experiments/multimap_20261011_vectorized_seed1/validation_final \
  --output experiments/multimap_content_20261012_seed1 --device cuda
```

本轮是一个训练种子的通信可学性与冻结网络比较，不能替代多种子消融或完整创新验证。若门槛失败，保留负结果并诊断，不能用损失下降宣称选包有效。

## 已完成的主实验

上述四阶段已实际在 RTX 3090 上完成，输出为 `experiments/multimap_content_20261012_seed1/`，主 pipeline 状态 `complete`。源移动文件 SHA256 为 `7db04cb34552e3ca182b24fe40dbd1481a78464597dc33483dfab3ab04da339f`，原文件未修改。

- 全部 100 张训练图完成冻结策略采集，共生成 203066 个标签，均匀保留 188770 个，每图 1299～2048 个。标签数值有限且不超过发送方可观测的新增单元上限；验证／测试训练样本为 0。
- 固定 final 移动模型的这 100 次 greedy 为 **87/100** 成功。这是补充的固定模型训练集评估，与前一轮在线变化策略的 78/100 含义不同，不能作前后提升比较。
- 5000 次真实监督更新抽取 80000 条样本，每图实际抽取 714～878 条；累计移动 SAC 仍为 8895，新增 SAC=0。开头／末尾各 100 次更新的平均训练 MSE 为 0.102984／0.016759。
- 从内容阶段初始权重到 final：Actor、双 Q、目标 Q、冻结上下文变化 L2 均为 0；内容头变化 L2=3.854863，温度不变。冻结上下文逐张量等于源移动 Actor。

## 同图实际探索结果

以下三组均冻结移动权重，采用原配置的内容／会合收益耦合，开发图完全相同，失败计 256 秒。重新执行的 FIFO 与上轮保存的全部 20 条开发评估记录逐项一致。

| 内容模式 | 成功数 | 平均任务时间 | 平均较低局部覆盖率 | 每 KiB 发送对应新增单元的逐图均值 |
| --- | --- | --- | --- | --- |
| FIFO | 18/20 | 184.348 s | 98.3723% | 24.089 |
| 合法规则 | 19/20 | 167.641 s | 99.2161% | 60.305 |
| 学习选包 | 19/20 | 175.205 s | 99.2690% | 40.072 |

学习选包比 FIFO 多完成一图、平均时间降低约 4.96%，但**没有超过规则基线**：成功数相同，平均时间反而较长。实际路径、任务时长和连接状态不同，表中的单位字节新增量也不能单独用于说明相同窗口下的排序质量。

两种新选包都完成原 FIFO 失败的 validation 3。validation 5 仍失败：FIFO 的团队覆盖为 86.3949%；规则为 98.6105%，尚缺 618 个自由单元；学习为 98.5430%，尚缺 648 个，属于同一个连通遗漏区域。规则和学习的发送队列均为空，剩余问题是末期区域未探索，不能归因于地图仍排队未交付。未降低 99% 完成门槛。

## 相同状态下的内容诊断

预定训练前 10 图与开发全部 20 图的固定 FIFO 轨迹，每 32 个决策采样，全部严格回放一致。两方向合计的假设四秒窗口新增量，先各图内平均、再地图等权平均：

| 状态来源 | FIFO | 合法规则 | 学习 | 理想选包（不可部署） | 学习 MSE | 训练均值常数 MSE |
| --- | --- | --- | --- | --- | --- | --- |
| 训练前 10 图 | 614.667 | 682.067 | 1019.883 | 1560.867 | 0.013526 | 0.020283 |
| 开发全部 20 图 | 505.933 | 570.725 | 1056.325 | 1666.554 | 0.012346 | 0.018352 |

原预定通信头门槛通过，预测上限违规比例为 0。学习在开发固定窗口下是 FIFO 的约 2.09 倍、规则的约 1.85 倍。这是固定状态、假设连接的诊断，不表示实际运行吞吐或整体探索获得同样倍数。

看到上述排序收益后，额外检查“是否仅由上限裁剪和偏向小包造成”：加入训练标签均值常数，使用相同发送方上限及评分／字节选包。该对照不重新训练、不选择终点、不改变原验收门槛。30 张探针图的原有全部指标再次逐项复现。裁剪常数的训练／开发窗口新增量为 577.983／534.742，MSE 为 0.019847／0.017786；学习在两个划分上均更好，支持确实学到了可用的信息区分。

复现诊断：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_content_check.py \
  --checkpoint experiments/multimap_content_20261012_seed1/head/latest.pth \
  --training-sampling experiments/multimap_content_20261012_seed1/head/sampling.json \
  --evaluation-dirs experiments/multimap_content_20261012_seed1/training_probe \
    experiments/multimap_20261011_vectorized_seed1/validation_final \
  --sample-every 32 --constant-control --device cuda --output experiments/NEW_EMPTY_PROBE_DIR
```

## 主结果后的关闭耦合诊断

因主结果“固定状态排序更好，实际时间却未超过规则”，在主结果确定后另运行两组只读诊断，均固定同一个内容／移动终点，令 `coupled=False`，使用共同的 FIFO 会合收益估计，只替换真实发送排序。这是事后诊断，不取代前述主结果，不追加学习，也不挑检查点。

| 真实发送排序 | 成功数 | 平均任务时间 | 失败图 |
| --- | --- | --- | --- |
| 规则，沿用 FIFO 会合估计 | 17/20 | 169.477 s | 0、5、14 |
| 学习，沿用 FIFO 会合估计 | 17/20 | 163.529 s | 1、5、8 |

关闭耦合后两组都由 19/20 降到 17/20。因此不能简单断言“换掉会合输入导致退步”，也不能因为学习的 163.529 秒更小就取代成功率更高的主结果。排序、接收地图与后续移动存在交互，需要让移动策略适应通信并做等预算对照，不能只调整一个开关便宣称问题解决。

诊断另审计 40 条轨迹、5180 段移动，约束及覆盖重算全部通过。与主实验合计 200 条实际轨迹、28582 段移动，穿墙数为 0。额外的 SAC／头更新及测试执行均为 0。计划与结果在 `uncoupled_plan.json`、`report/uncoupled_summary.json`。

复现单组（规则组把 learned 换成 rule，输出使用另一个新目录）：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_train.py --stage evaluate \
  --initialize-from experiments/multimap_content_20261012_seed1/head/final.pth \
  --variant A --content learned --uncoupled --split validation --eval-maps 20 \
  --max-hours .3 --device cuda --run-dir experiments/NEW_EMPTY_UNCOUPLED_DIR
```

## 验证与产物

46 项行为检查全部通过（35.556 秒）。新增内容检查覆盖均匀历史保留、地图等权抽样、真实头更新与移动冻结，以及零分数退化为 FIFO、假设窗口不改变真实通信状态和局部图。主实验 100 条采集与 60 条开发轨迹物理审计通过，15092＋8310＝23402 段移动无穿墙，覆盖重算、任务标准、速度与通信字节约束一致；测试策略执行为 0。

- 内容模型：`head/final.pth`；含优化器状态：`head/latest.pth`。
- 主汇总：`report/summary.json`；主图：`report/frozen_content.png`／`.pdf`。
- 固定状态结果：`fixed_state_check/summary.json`；额外常数控制：`fixed_state_constant_control/summary.json`；图：`report/communication_checks.png`／`.pdf`。
- 失败区域与实际路线：`report/remaining_regions.png`／`.pdf`、`failure_analysis.json`。
- 主结果后的关闭耦合诊断：`validation_{rule,learned}_uncoupled/`、`report/uncoupled_summary.json`。
- 数据／冻结／抽样审计：`checks/label_audit.json`、`frozen_weights.json`、`constant_control_audit.json`；行为日志：`checks/behaviour_46.log`。

目前结论是：通信头在多图上可学，固定窗口选包有收益；冻结移动权重的实际方法尚未胜过规则，完整创新优势没有验证。下一轮联合微调必须加入规则基线，与 FIFO 和学习方法共享初始移动状态、实际 SAC 更新预算、地图调度、学习率与训练辅助措施，不能仅和 FIFO 比较。
