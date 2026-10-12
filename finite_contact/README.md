# 两机器人有限通信窗口实验

新对话接手请先阅读[实验计划与新对话交接](EXPERIMENT_HANDOFF.md)，其中包含目标、当前阶段、恢复命令和后续验收计划。

本入口在 `research/finite-contact-exploration` 分支上开展小规模可行性验证。复用 IR² 的 `Graph_generator`、`Node`、LiDAR 传感器、六层图注意力 Actor 与双 Q 网络；不调用原 `Env.single_robot_step()` 的即时地图／位姿／全局图合并。

统一环境：`/home/wyz/anaconda3/envs/ir2_train`。后续运行请用该环境的 Python；入口会核对 `sys.prefix`。

本轮完整改动说明见 [CHANGELOG.md](CHANGELOG.md)，研究设计的仓库内快照见 [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md)。单图验收与物理审计摘要 JSON 保存在 [results/single_map_20261011](results/single_map_20261011/)，原始模型及实验输出保留本机。

后续新增 `simple_rooms_v1` 的 100/20/20 多样简单地图入口，旧单图默认与 checkpoint 保留 legacy 地图族。生成检查、多图轮训和事前固定的开发验证流程见 [MULTIMAP_EXPERIMENT.md](MULTIMAP_EXPERIMENT.md)。

首轮 A/FIFO 已实际轮训全部 100 张，新增加 7395 次 SAC；同一组 20 张开发验证从起点 7/20 提高到 final 18/20。42 项行为检查及 100 条训练／40 条验证轨迹物理检查通过。该结果是单种子热启动的移动阶段改善，后续 AB 通信与适应结果见下文；测试图不执行策略。

多图通信入口 `finite_contact_multimap_content.py` 已实际完成：在全部 100 张训练图采集监督、按地图均衡抽样，固定移动网络完成 5000 次头更新。FIFO／规则／学习在同一开发集成功 18／19／19 张，平均时间 184.348／167.641／175.205 秒；学习固定窗口收益优于规则，但实际方法尚未超过规则。46 项检查通过，该轮只更新通信头。预算、诊断及完整结果见 [MULTIMAP_CONTENT_EXPERIMENT.md](MULTIMAP_CONTENT_EXPERIMENT.md)。

三组等预算移动适应已实际完成：每组完整轮训 100 张图、新增 2000 次 SAC，到固定终点 10895，通信头及上下文冻结。相同 20 张开发图，FIFO／规则／学习成功 18／19／20 张，平均时间 176.503／150.472／125.480 秒；学习比等预算规则快 16.61%。50 项行为检查通过，360 条轨迹／42986 段移动物理审计通过。仍为单种子开发结果，测试策略不执行。入口 `finite_contact_multimap_joint.py`、事前预算、模型及结果见 [MULTIMAP_JOINT_EXPERIMENT.md](MULTIMAP_JOINT_EXPERIMENT.md)。

前一阶段适应种子复现已完成：FIFO 三轮成功 18／19／19，规则 19／18／20，学习 20／17／19。平均探索时间分别 171.970／155.520／136.205 秒；学习三轮均更快，但有两轮比规则少成功一张，尚未稳定全面超过规则。54 项行为检查及 1080 条主结果轨迹物理审计通过。共享预训练起点、资源恢复、全部失败及后续稳定性方向见 [MULTISEED_EXPERIMENT.md](MULTISEED_EXPERIMENT.md)，编排入口 `finite_contact_multiseed.py` 默认最多三个 worker。

前一轮收尾抽样试验已完成：seed=2六组各追加同样2000次SAC，原pooled的FIFO／规则／学习均成功18/20，候选map_tail为19／20／20；学习平均120.694秒，比同为20/20的规则快17.57%。该阶段训练监测中FIFO退步，随后补做的另两种子复现见下段，默认抽样保持pooled。60项行为检查、900条轨迹物理审计通过，测试策略未执行。预算、全部正负结果与模型见[收尾稳定性试验](STABILITY_MULTIMAP_EXPERIMENT.md)。

收尾抽样的固定条件复现已完成：新增seed=1、3共12组，每组保留2000次真实SAC；与先行seed=2合并，候选学习的三个适应种子分别完成20／20／20张（每轮20张）。但规则在确认种子1／3分别由20/20降到19/20、18/20，种子1学习同为20/20却慢7.545秒，候选没有一致优势，默认仍pooled。64项行为检查和2700条主轨迹物理审计通过；OOM后以两并发完整恢复，额外计算单列，测试策略执行0。全部结果、模型与范围见[收尾抽样复现](STABILITY_REPLICATION_EXPERIMENT.md)。

当前已启动1000张简单地图训练：保留原100张、新增900张，1040张训练／开发／测试地图质量审计通过。两机器人，三方法各新增20000次真实SAC，使用无损压缩成功经验和单worker后台队列，先训练学习方案；用户已执行bash run_thousand_maps_foreground.sh恢复前台训练，按学习→规则→FIFO依次运行；接手时先检查实时状态，避免重复启动。通信头沿用原100图版本且冻结，当前尚无新终点结果。计划与进度见[1000图实验](THOUSAND_MAP_EXPERIMENT.md)。

当前已完成从随机权重开始的单图主链路：采用自身成功经验辅助，并在获得能力后把 Actor 学习率降低到 1e-5；Q 继续按 1e-4 学习。两个连续真实训练终点 SAC=3500／4000 均达到每机 99%，十次 sampled 均为 9/10。最终 greedy 耗时 102.624 s，但开发验证为 0/2，仍不能宣称换图泛化。启动和完整结果见 [单图训练指南](RUN_SINGLE_MAP.md)；新编排入口为 `finite_contact_single_map.py --profile actor_decay`，各实际阶段及失败对照见 [Actor 步长记录](ACTOR_RATE_EXPERIMENT.md)。

## 当前实现范围

- 固定 2 台机器人，默认 legacy 地图族为 1 张训练、5 张开发验证及 5 张留出测试地图，最多 10/5/5。新 `--map-family simple_rooms_v1` 可使用 100/20/20，不加载原来的大地图集；数据集种子与训练种子分别保存。
- 256×256 简单布局：legacy 按门的位置划分双房间地图；simple_rooms_v1 从同一分布独立生成两／三房间、不同门／障碍／旋转的三个划分。相邻起点属于任务配置，规范哈希检查旋转／镜像重复；仍不能当作任意新拓扑泛化证据。
- 直接单播、距离连接、固定速率、无随机丢包；地图增量、位姿、ACK 均实际序列化并计费。数据包完整交付才更新接收者地图，断连保留部分包进度。所有组共同跳过已有合法证据证明对方知道的完整待发包，避免把基础去重伪装成选包收益。
- 地图采用静态栅格增量，最多 256 B／包。每台机器人发送和接收分别受同一速率上限约束；默认 2048 B/s 是开发起点，尚未做系统容量标定。
- 以地图单位／秒计运动；每秒以内检查传输，按实际动作时间折扣。显式 WAIT 与移动结束后的同步空闲分别计时。
- B0/A 支持 FIFO 或规则发送；B/AB 使用已训练的选包头。规则和学习评分均只接收发送方本地信息与合法收到的状态。
- 通信头以训练期接收方地图产生的“新增已知单元数”作为监督，图编码上下文在该阶段冻结。微调时保持通信头与其上下文编码器固定，避免输入分布随移动网络同时漂移。
- 保存配置、依赖、源码快照、地图哈希、TensorBoard、逐回合指标、通信事件、轨迹以及可恢复 checkpoint。
- 新建训练默认使用 centered pointer：有效动作原始分数减去均值后再 tanh 裁剪，避免共同偏移导致的动作概率饱和。原 `model.py` 的默认调用仍保持旧模式；历史检查点缺少该配置时也按 legacy 加载，复现旧实验时不能静默换模式。
- 新建训练默认从整个队列 spread 抽取内容标签，将学习预测限制在本地确认信息给出的新增量上限，并在有限窗口估计中优先完成活动续传包。旧检查点缺字段时分别保留 prefix／none／legacy。载入旧模型后要启用修正，应显式设置对应参数并使用新实验目录。

当前是简化的 IR² 适配环境：小地图重建局部已知自由空间图，不使用原大地图层次图合并；网络维度从原 128 调为 64，输入从 6 增为 12，所有核心对照使用同一结构。不能把它的结果直接与原论文数字对照。

## 快速运行

从项目根目录运行：

```bash
source /home/wyz/anaconda3/etc/profile.d/conda.sh
conda activate ir2_train
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
python finite_contact_train.py --smoke --device cuda --variant B0
```

`--smoke` 只执行 8 秒仿真、一个团队回合、最多两次 SAC 更新，验证运行与保存，不代表收敛。明确指定 CUDA 时，如果显卡不可用会报错；`--device auto` 可以选用 CPU，但正式性能测速应明确记录设备。

先跑单图基础对照：

```bash
python finite_contact_train.py --stage movement --variant B0 --train-maps 1 --max-updates 1500 --episodes 150 --max-hours 0.35 --eval-every 10 --eval-maps 2 --eval-train --final-eval --device cuda --run-dir experiments/finite_b0_movement_seed0
```

会合收益模块采用同等预算从头训练：

```bash
python finite_contact_train.py --stage movement --variant A --train-maps 1 --max-updates 1500 --episodes 150 --max-hours 0.35 --eval-every 10 --eval-maps 2 --eval-train --final-eval --device cuda --run-dir experiments/finite_a_movement_seed0
```

阶段二固定移动策略，使用规则发送采集监督样本并训练通信头：

```bash
python finite_contact_train.py --stage content --variant A --initialize-from experiments/finite_a_movement_seed0/latest.pth --label-sampling spread --max-content-updates 300 --episodes 20 --max-hours 0.15 --device cuda --run-dir experiments/finite_a_content_seed0
```

接入学习内容评分并微调会合策略：

```bash
python finite_contact_train.py --stage finetune --variant AB --initialize-from experiments/finite_a_content_seed0/latest.pth --content-score-bound sender --delivery-queue-order service --max-updates 2000 --episodes 20 --max-hours 0.15 --eval-train --eval-maps 2 --final-eval --device cuda --run-dir experiments/finite_ab_finetune_seed0
```

B 的路线对应 B0 移动预训练 → 内容头训练 → `--stage finetune --variant B`。这些命令是阶段执行示例，当前不自动串联。首轮移动检查以 1500 次更新为上限，150 回合／21 分钟用于资源保护；内容头及微调示例的 500 回合和 2 小时也是预算上限，不是收敛承诺。先观察单图是否可学习，再决定是否使用更多简单地图。

公平比较时，B0/A 也必须获得与 B/AB 相同的移动预训练及后续微调预算，例如对 B0 使用 `--stage finetune --variant B0 --content fifo`。通信头的额外监督样本、更新和计算量独立报告；严格计算对照可同样训练头但不用于基础组决策。不能把“500 回合基础线”和“预训练 500 回合再微调 500 回合”的完整方法直接排名。

`--max-updates` 限制检查点中的累计更新数，初始化不会将计数清零。例如载入一个已有 1500 次更新的移动模型后，要额外微调 1500 次，应设置 `--max-updates 3000`，并对各组使用同样的新增预算。

第三轮内容头实验与 A/AB 配对结果见 [CONTENT_EXPERIMENT.md](CONTENT_EXPERIMENT.md)。`--max-content-updates` 独立限制监督更新数，内容训练期间移动网络冻结。按固定轨迹复核内容预测和假设四秒窗口的交付量，可使用 `finite_contact_content_check.py`；该工具的真实接收地图仅用于离线评估，不能当作部署观测。

继续同一阶段：

```bash
python finite_contact_train.py --stage movement --resume experiments/finite_b0_movement_seed0/latest.pth --episodes 1000 --max-hours 2 --device cuda
```

`--episodes` 是该阶段目标总回合数。`--resume` 要求相同阶段和配置，并恢复优化器、回放及 RNG；`--initialize-from` 开始新阶段，载入网络并重建优化器和回放。中断发生于回合中途时，不恢复仿真中途状态，继续时重跑该地图，日志标记该回合为 interrupted。`best.pth` 不含回放，只用于初始化／评估；`latest.pth` 用于恢复。

验证集评估：

```bash
python finite_contact_train.py --stage evaluate --initialize-from experiments/finite_ab_finetune_seed0/latest.pth --split validation --device cuda --run-dir experiments/finite_ab_validation_seed0
```

最终小规模测试把 `--split` 改为 `test`，使用另一个新目录。训练中每 100 个团队回合验证一次，只按验证集覆盖缺口面积选 `best.pth`；不使用测试集选模型。目录非空时不会覆盖旧实验。

## 需要明确的近似与后续工作

1. 运动决策同步：两个机器人分别选邻接节点，较早到达者等待本轮较长动作结束；日志单列 `synchronization_idle_seconds`。显式 WAIT 至少持续一个时间片，也会受本轮同步边界约束。这是首版调度近似，所有组一致，不能混称为完全异步执行。
2. 链路检查要求每个子时间片首尾都在通信范围内，是保守离散近似；默认时间片 1 s，正式比较前需要更小时间片敏感性检查。
3. 会合容量使用固定名义接触时长、已收到速度及位姿年龄估计，尚未训练接触时长预测器；没有读取真实未来轨迹。
4. 会合模块目前把可交付收益沿已知图路径传播，并扣除预计到达成本。等待时间成本由奖励承担；还没有对多种候选停留时长做显式最优化。
5. 通信标签目前衡量接收者新增已知单元，尚未实现任务价值／路线减少监督；能否改善探索效率必须由对照实验判断。
6. 首版直接发送栅格更新，无压缩、多跳转发、信号模型或无线竞争优化。原代码入口 `driver.py` 仍保留原行为；本研究一律使用 `finite_contact_train.py`。

## 行为检查

```bash
python -m unittest discover -s tests -p test_finite_contact.py -v
```

检查内容包括字节上限、断连隔离、部分包续传、确认与去重、隐藏状态隔离、等待、真实 SAC 更新、监督通信头和地图划分。运行结果与学习效果分开记录。

首版检查与短运行结果见 [VALIDATION.md](VALIDATION.md)。

## 稳定性修复与等预算对照

第二轮方案与结果见 [STABILITY_EXPERIMENT.md](STABILITY_EXPERIMENT.md)。`--max-updates 1500` 精确限制累计 SAC 更新数；`--final-eval` 在剩余墙钟预算内评估最终权重并保存 `final.pth`。最终模型的更新数写在评估的 `learner_updates` 字段；中途中断回合不参与完整回合积分统计。主要等预算比较使用 `final.pth`，不能用不同回合长度下选模次数不同的 best 替代。

每次更新的梯度、pointer 原始／裁剪输入幅度、饱和比例与 WAIT 概率写入 `updates.jsonl`。等待另记录断连秒数与没有新增地图交付的秒数；这些日志仅用于离线诊断，不作为 Actor 的额外信息。

新训练默认 `horizon_target=terminal`：256 秒有限任务截止后不估计未来收益，成功终点同样不 bootstrap。外部墙钟／更新预算中断不改成任务终点。旧检查点缺字段时保留历史 bootstrap；从旧权重开始修正实验必须显式加 `--horizon-target terminal`，并使用新的运行目录。每更新的 `deadline_fraction` 记录实际抽到截止转移的比例。修正与等预算复测见 [DEADLINE_EXPERIMENT.md](DEADLINE_EXPERIMENT.md)。

`--alpha-learning-rate` 只控制 SAC 温度优化器，新训练默认 1e-4；旧模型缺字段时沿用其原 learning_rate。诊断用 0.005 需要显式指定，对所有比较组一致使用。它不改变 Actor/Q 学习率，也不保证贪心策略或采样完成率提高。

`--policy-learning-rate` 只控制 Actor，未指定时新配置沿用公共 learning_rate，旧检查点沿用其历史 Actor 速率。要在保留 Adam 动量、回放、成功缓存和 RNG 的条件下改变 Actor 速率，使用 `finite_contact_fork.py --checkpoint <完整finetune状态> --output <新目录> --policy-learning-rate 0.00001`，再从输出的 `fork_source.pth` 严格 resume。该操作只改 Actor LR，其他网络与数据不变，既有运行不会覆盖；普通 resume 仍要求配置完全一致。Actor 的较小步长仍真实更新，不等同冻结。

可选 `--frontier-shaping-scale 2` 为所有方法共用的本地前沿距离势函数训练辅助，默认 0 关闭；必须同时使用 terminal 截止目标。每次真实持续时间 tau 加 `gamma**(tau/dt)*Phi(next)-Phi(current)`，任务结束时 next Phi=0。它不额外提供动作或地图信息，完整任务的折扣奖励抵消为只与初始状态有关的常数。奖励分项记录为 frontier_shaping；此工程诊断不能计作通信方法的创新。

`--coverage-reward local_mean` 将共享探索奖励从联合覆盖增量改为两机平均局部覆盖增量，仍乘相同系数 20。它让另一台机器人已看过的区域在本机首次补扫时也有探索奖励，与“每机 99%”完成目标衔接；真实地图交付、时间／距离成本与终奖保持原设置。默认及旧检查点仍为 union，新实验须显式指定，不导入旧奖励回放。所有比较组须使用同一选项；方案、行为检查与结果见 [LOCAL_COVERAGE_EXPERIMENT.md](LOCAL_COVERAGE_EXPERIMENT.md)。

第四轮已完成三对等预算诊断：截止目标修复保留；温度加速与前沿势函数未稳定改善 AB，仍不启用为默认。未微调的冻结移动 AB 在训练图 greedy 成功（197.54 s）、五次 sampled=5/5，但两张开发验证图 0/2。单图参考模型为 `experiments/frozen_20261011_report/AB_learned.pth`，可直接用于 `--stage evaluate --initialize-from ... --split train`，其 SAC=1500、头=300，导出没有新增训练。不要与 SAC=2000 的微调组混写为等预算优势。当前暂不扩图，下一步检查微调退化与换图表现；best_train 按积分保存，不能保证任务成功。

第五轮加入可选 `--success-imitation-weight 1`：Actor 在普通 SAC 更新之外，学习自己在训练图上真实完成任务的动作。只保留最快的三个完整成功团队回合，失败、预算中断、评估以及内容头阶段不会提供示例；不使用人工路线。默认值 0 保留旧行为。成功经验与优化器、回放和 RNG 一起存入 `latest.pth`，恢复时一并载入。内容头和上下文编码器在移动微调阶段继续冻结。该措施是训练稳定性辅助，必须共同应用于公平比较组，不能单独包装为通信创新。方案与结果见 [SUCCESS_LEARNING_EXPERIMENT.md](SUCCESS_LEARNING_EXPERIMENT.md)。

`best_train_success.pth`／`best_validation_success.pth` 只有完整评估集全部任务成功才保存，按平均完成时间、覆盖积分依次比较。旧的 `best_train.pth`／`best.pth` 仍按积分选取，可能没有完成任务。主要验收继续使用固定更新预算的 `final.pth`。

多图开发可用 `--eval-train --eval-train-maps 10` 固定只监控前 10 张训练图；这时成功检查点仅表示这 10 张全部完成，不表示训练集全部完成。评估范围记在 `invocations.jsonl`，完整多图结果按单独评估记录报告。

对固定终点模型进行只读验收（输出目录必须为空）：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_verify.py \
  --checkpoint experiments/success_20261011_AB_learned/at_2000.pth \
  --samples 10 --sample-seed 101 --validation-maps 2 --device cuda \
  --output experiments/my_single_map_acceptance
```

验收分别报告训练图 greedy、十次 sampled 和两张开发验证图；检查评估没有改动网络与学习计数，并保留每次地图、轨迹及通信日志。单图门槛为 greedy 成功且 sampled 至少 9/10 成功，开发验证结果单独报告。还需从完整检查点继续真实训练并用另一动作噪声序列复核，不能只凭一次通过宣布稳定或跨地图泛化。十次采样不是十个独立训练种子。

`--time-penalty` 可单独控制每秒成本，默认仍为 0.02。提高成本会改变训练目标，必须对所有算法使用相同设置，并比较完成、同步与通信指标，不能只比较训练回报或等待。

创建运行目录内的 `STOP` 文件，或发送 SIGINT/SIGTERM，会在下一次完整动作／更新后记录中断、保存模型并退出。再次恢复前须删除自己创建的 STOP 文件。旧检查点恢复仍使用旧 pointer；要改模式应使用 `--initialize-from ... --pointer-mode centered` 开始新目录，不冒充原实验续训。

## 单图学习评估

单图实验计划与结果见 [SINGLE_MAP_EXPERIMENT.md](SINGLE_MAP_EXPERIMENT.md)。训练加上 `--eval-train`，会在训练前和每次定期验证时，把固定 greedy 策略重新放回训练图，记录 `train_policy.jsonl`；`initial.pth` 保存训练起点，`best_train.pth` 按训练图覆盖亏损积分保存候选。它只用于验证单图能否学会，不能当作未见地图上的效果。

独立复测训练图（随机对照把 `greedy` 改为 `random`，策略采样改为 `sampled`；采样时可加 `--eval-repeats 5`）：

```bash
python finite_contact_train.py --stage evaluate --initialize-from experiments/single_map_20261011_A/best_train.pth --split train --evaluation-policy greedy --device cuda --run-dir experiments/single_map_20261011_A_recheck
```

独立评估另存最终局部地图、轨迹及通信日志。greedy 在固定起点和确定性仿真中复测只是可复现检查，不增加独立样本数。`finite_contact_probe.py` 的人工路线仅用于判断任务能否完成，不给学习策略提供路线，也不是可部署对照。

汇总现有记录并绘图：

```bash
python finite_contact_report.py --runs experiments/single_map_20261011_B0 experiments/single_map_20261011_A --output experiments/single_map_20261011_report
```
