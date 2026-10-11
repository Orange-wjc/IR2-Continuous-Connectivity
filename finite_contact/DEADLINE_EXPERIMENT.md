# 第四轮：任务终点、温度与覆盖收尾诊断

日期：2026-10-11；环境 ir2_train；两机器人、1 张训练图，仍使用原 99% 局部覆盖完成条件。

## 启动前固定的诊断与计划

上一轮 AB 的训练图 greedy 停在 98.8958%。严格回放发现：192～256 秒的 88 个机器人决策中 75 个选择 WAIT，平均策略熵为 2.235；WAIT 与最佳移动的 Q 差中位数只有 -0.00761。这表明动作排序很弱，不能仅凭 WAIT 多就证明 Q 一直偏好 WAIT。

代码核查另发现明确的任务边界错误：环境有 256 秒任务截止期，观测包含已用时间，截止记为 truncated；回放保存该标志，但 SAC 目标只使用 terminated，因此失败截止仍加入未来软价值。这里的截止是有限任务结束，区别于外部训练预算中断，目标应为即时剩余奖励。修正不是研究创新，也不预先声称它是 WAIT 的唯一原因。

- 新训练默认 horizon_target=terminal；旧检查点缺该字段时保留 bootstrap，以复现旧结果。原奖励、熵系数更新、动作、地图、网络与传输规则不变。
- A/FIFO 和 AB/learned 均从相同 content_20261011_A/final.pth（SAC=1500、头=300）初始化，新优化器与空移动回放。显式 sender 上限、service 队列和 terminal 目标。
- 每组精确新增 500 次移动更新至 2000；最多 20 回合／9 分钟，训练前和终点评估训练图及两张验证图。目录 deadline_20261011_{A_fifo,AB_learned}。
- 相同初始权重和 500 更新预算的上一轮 _bounded 是历史 bootstrap 对照；源码快照、配置和全部失败保留。当前两个新组之间只允许 variant/content 不同。
- 终点各做五次训练图 sampled 与独立 greedy 复测；同一模型五次动作采样不是五个训练种子。始终报告完成率、最低局部覆盖、覆盖亏损、等待、有效接收及数值状态。尚不使用测试集、不扩大训练图。
- 若修正不改善收尾，记录负结果，再用实测证据决定下一步；不修改 99% 阈值、不只挑有利 checkpoint。

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_train.py \
  --stage finetune --variant AB --content learned \
  --initialize-from experiments/content_20261011_A/final.pth \
  --content-score-bound sender --delivery-queue-order service --horizon-target terminal \
  --max-updates 2000 --episodes 20 --max-hours 0.15 \
  --eval-every 100 --eval-maps 2 --eval-train --final-eval --device cuda \
  --run-dir experiments/deadline_20261011_AB_learned
```

A 组替换 variant=A、content=fifo，目录 deadline_20261011_A_fifo。

## 截止期修正的完整结果

两组均精确新增 500 次 SAC 更新至 2000，头及上下文固定。独立 greedy 复测与训练终点评估完全一致，五次 sampled 均完整完成，未使用测试图。

| 指标 | A/FIFO | AB/learned |
| --- | --- | --- |
| 训练图 greedy | 成功，123.24 s | 失败，256 s |
| 最低局部覆盖 | 99.0736% | 97.8474% |
| 训练图覆盖亏损积分↓ | 0.212777 | 0.091297 |
| 两机器人显式等待 | 22.42／29.45 s | 149.95／158.78 s |
| 五次训练图采样成功数 | 3/5 | 2/5 |
| 两张验证图成功数 | 2/2 | 2/2 |
| 验证 0／1 时间 | 141.01／126.94 s | 67.68／135.88 s |
| 验证平均覆盖亏损积分↓ | 0.223706 | 0.117667 |
| 500 个批次抽到的截止转移总数 | 33 | 34 |

相对历史 bootstrap 等预算组，A 训练图完成更快，但验证更慢；AB 验证均值改善，但训练覆盖从 98.90% 降到 97.85%、采样由 4/5 降到 2/5。不能宣布截止期修正解决 WAIT，不能只选验证均值支持它。有限任务目标修正保留，尚需单独处理移动学习。

审计确认初始化权重一致、头及上下文冻结、配置仅改变 horizon_target、两组只差 variant/content、更新序列与独立复测一致。所有更新饱和比例为 0、无零梯度。详见 `experiments/deadline_20261011_report/audit.json`、`learning_curves.png` 与 `numerical_diagnostics.png`；复核命令 `python experiments/deadline_20261011_report/audit.py`。

额外严格回放位于 `deadline_20261011_old_AB_q_diagnosis` 和 `deadline_20261011_AB_q_diagnosis`，只加入离线 Q／效用诊断，部署输入不变。原 AB 在 248 秒以后的 WAIT Q 均值约 2.76，Actor 相对最佳 Q 的差距约 0.017；软 Q 不能直接等同 greedy 轨迹的实际回报，这些数字不能单独证明温度的因果作用。

## 第二个单因素诊断：温度调整速度（启动前固定）

第一对完成 500 次更新：A 训练 greedy 成功（123.24 秒）；AB 仍失败（97.8474%，256 秒，等待约 149.95／158.78 秒）。修正有限任务终点仍有必要，但不能称为收尾已解决。AB 终点批次熵约 1.916、alpha 约 0.1118；与有效动作数对应的目标约 0.1 相差较大。

单独暴露 alpha_learning_rate，旧检查点缺字段时按原 learning_rate 复现。第二对只把温度优化器学习率从 1e-4 改为 **0.005**；Actor、Q 的学习率仍为 1e-4，目标熵、奖励和新 terminal 边界均不变。这个量级使 log-alpha 在 500 次持续降温更新中可以变化约 2.5，而旧速率只有约 0.05；实际变化取决于熵反馈，不保证达到指定 alpha。它是缩短短实验内温度适应时间的工程诊断，不是论文创新，也不是动作持续时间熵归一化。

两组仍重新从原 content_20261011_A/final.pth 的 1500 次更新开始，精确新增 500 次至 2000，头保持 300。目录为 temperature_20261011_{A_fifo,AB_learned}，其余预算和终点评估协议与第一对相同。不得从第一对某个有利中途模型继续。第一对是这次温度速率的等预算对照，失败记录保留。预定各五次训练图采样及独立 greedy 复测；若未通过单图，不扩图。

运行命令同上，增加 `--alpha-learning-rate 0.005` 并更改为 temperature 目录。

### 温度加速完整结果

| 指标 | A/FIFO | AB/learned |
| --- | --- | --- |
| 训练 greedy 最低局部覆盖 | 72.7523%，失败 | 95.0331%，失败 |
| 两机显式等待 | 9.78／9.42 s | 0／0 s |
| 五次训练图采样成功数 | 2/5 | 2/5 |
| 两张验证图成功数 | 2/2 | 2/2 |
| 验证 0／1 完成时间 | 144.88／179.30 s | 115.88／150.82 s |
| 终点 alpha | 0.01407 | 0.01244 |
| 终点更新批次熵 | 1.056 | 1.101 |

两组各精确 500 次新更新，初始权重一致，内容头冻结，配置相对 deadline 对照仅温度优化器速率不同。greedy 独立复测一致，全部五次采样完整，无饱和与零梯度。`temperature_20261011_report/audit.json` 和学习曲线保留全部结果；审计命令 `python experiments/deadline_20261011_report/audit.py --prefix temperature`。此诊断有效降低等待，但未改善完成稳定性，因此不采用为默认。

## 第三个单因素诊断：前沿距离势函数（启动前固定）

温度加速组的终点 greedy 都失败：A 最低局部覆盖 72.75%，等待仅 9.78／9.42 秒；AB 团队并集 100%，两机分别 95.03%／100%，等待均为 0。降低 WAIT 不等于完成局部地图任务。因此不把 0.005 设为新的默认，不在该模型上继续；接下来针对穿越已探索区域的稀疏回报做一个共同训练辅助。

对机器人 i，从当前节点沿其**本地图**求到任一 utility>0 可达节点的最短距离 d_i，定义 Phi_i=-2*min(d_i/256,1)。没有可达有用节点时为 0。每次真实持续 tau 的联合动作加 `gamma**(tau/dt)*Phi_i(next)-Phi_i(current)`；成功及任务截止的 next Phi 均置 0，外部训练预算中断不置 0。

这个辅助只使用本机图坐标、边和效用，不用真实队友位置／地图，也不决定动作。沿完整任务的物理时间折扣求和会抵消为 `-Phi_i(initial)`，所以在当前 terminal 任务与折扣下不改变各策略的原回报排序；不把正的靠近奖励当作额外任务成功收益。动作时长和传输仍按原协议。25 项行为检查包括真实环境中成功／超时两类回合的抵消、不读取隐藏状态和奖励分项守恒。

两组重新从原内容阶段 SAC=1500 的同一个模型开始；alpha_learning_rate 回到 1e-4，terminal 保留，仅新增 frontier_shaping_scale=2。同样新增 500 次至 2000、最多 20 回合／9 分钟、头固定，目录 `frontier_20261011_{A_fifo,AB_learned}`。对照使用 deadline 第一对，配置只允许这个尺度不同。训练图 greedy、两张验证图及各五次训练图采样协议相同。单图结果无论正负均保留，不追加到复杂图。

运行命令同第一对，增加 `--frontier-shaping-scale 2`，使用 frontier 目录；不要沿用温度加速参数。

### 前沿辅助完整结果

| 指标 | A/FIFO | AB/learned |
| --- | --- | --- |
| 训练 greedy | 成功，148.52 s | 失败，256 s |
| 最低局部覆盖 | 99.1522% | 97.4731% |
| 两机器人显式等待 | 10.52／0 s | 10.50／17.78 s |
| 五次训练图采样成功数 | 5/5 | 3/5 |
| 两张验证图成功数 | 2/2 | 0/2 |
| 验证最低局部覆盖 | 99.12%／99.68% | 98.65%／97.75% |
| 验证平均覆盖亏损积分↓ | 0.237021 | 0.139757 |

两组精确新增 500 次更新，各 563 个新团队决策。初始化逐张量一致、头与上下文冻结，相对 deadline 仅 frontier_shaping_scale 不同，饱和与零梯度均为 0。尽管 A 单图五次采样均成功，AB 单图与验证均未稳定完成，不能用 AB 较低的验证积分代替成功。势函数保留为默认关闭的诊断选项，不作为统一的新训练配置。

严格回放 `frontier_20261011_AB_q_diagnosis` 与记录完全一致，最终还剩 1222 个自由单元，bbox x=127～176、y=192～239，每台有 17 个 frontier、最大效用 12。192 秒以后当前位置效用始终 0，势函数范围约 -2～-0.3125，表明辅助并非全程为零；等待下降后主要仍是移动绕行和最后区域未覆盖，不能宣称已经解决。

审计：`experiments/frontier_20261011_report/audit.json`。复现 `python experiments/deadline_20261011_report/audit.py --prefix frontier`；四个条件（含上一轮历史组）的比较图为 `frontier_20261011_report/ablation_summary.png/pdf`，验证柱图同时标记成功数，不能只按积分判断优劣。

## 不微调的冻结移动对照（评估前记录）

三个独立配对的初始 greedy 一致：A=147.66 秒成功、AB=197.54 秒成功；最终 AB 均失败。补做源 `content_20261011_A/final.pth` 的只读评估：A/FIFO 与 AB/learned 都保持 SAC=1500、头=300，仅 sender 上限、service 队列、terminal 目标，alpha=1e-4、势函数关闭。各评估同两张开发验证图和五次训练图 sampled，不进行任何梯度更新，目录 `frozen_20261011_{A_fifo,AB_learned}_{validation,sampled}`。

这是用于定位联合微调退化的冻结移动基准，训练预算少于 2000 次终点组，不能混写为等预算方法优势。初始训练图 greedy 已在三个配对起点重复一致，后续只报告当前固定起点下的可行性，仍不保证稳定学习或未见拓扑泛化。此时不再增加新的调参训练。

### 冻结移动结果与可复现模型

| 指标 | A/FIFO | AB/learned |
| --- | --- | --- |
| SAC／监督头更新数 | 1500／300 | 1500／300 |
| 训练图 greedy | 成功，147.66 s | 成功，197.54 s |
| 训练图五次 sampled | 4/5 | 5/5 |
| 两张开发验证图 greedy | 1/2 | 0/2 |
| 验证最低局部覆盖 | 82.90%／99.76% | 90.31%／90.65% |

AB 在单图上的可行性成立，五次采样使用一个训练种子和同一个模型；这不是稳定泛化的证据。它的微调前训练图 greedy 成功在三个配对初始化中均一致，而三个 500 次微调终点都失败；下一步应研究**微调后的能力退化与泛化**，不能继续假设只是 WAIT 多。现有证据尚未区分 Q 估计误差、策略拟合和状态分布变化各自的因果作用。

显式配置的模型保存在 `experiments/frozen_20261011_report/A_fifo.pth` 和 `AB_learned.pth`，逐张量核对网络、alpha、优化器状态与原内容阶段模型完全一致，只写入评估用配置，不新增训练。可以直接加载，不必手动补 variant/content 等参数：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_train.py \
  --stage evaluate --initialize-from experiments/frozen_20261011_report/AB_learned.pth \
  --split train --eval-maps 1 --evaluation-policy greedy --device cuda \
  --run-dir experiments/frozen_AB_train_recheck
```

审计与导出脚本：`python experiments/deadline_20261011_report/frozen_audit.py`，结果为 `frozen_20261011_report/audit.json`。已保留所有六个新训练组、四个冻结评估组及独立 greedy 复测的配置、源码、环境、指标、轨迹与通信日志。25 项行为检查、编译及差异格式检查通过，仍在 research/finite-contact-exploration，未提交／推送模型。

## 本轮结论与下一步约束

保留有限任务 terminal 目标修复；alpha 仍默认 1e-4，势函数默认 0。本轮没有找到让 AB 联合微调稳定收尾的有效设置。单图可行性模型作为冻结参考保留，测试集不使用，暂不扩地图。下一步先把成功率作为可行性门槛，再考察时间与覆盖积分；现有 best_train 仍按积分保存，不能直接当作“已完成任务”的保证。优先用原策略作为参照检查微调偏移，并考虑受约束的小步策略更新／训练回放保留；不直接堆时长，也不把这些工程修复写成通信创新。正式方法优势仍需要独立训练种子与未用于调试的场景。
