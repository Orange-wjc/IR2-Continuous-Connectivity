# 有限通信下多机器人探索：实验计划与新对话交接

更新时间：2026-10-12，北京时间。进度快照采集于 15:13，训练会继续推进；接手时必须重新读取实际状态。

本文是新对话的优先入口，包含继续当前工作所需的目标、实验配置、结果边界、命令与下一步任务。研究设计详见 [RESEARCH_DESIGN.md](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/finite_contact/RESEARCH_DESIGN.md)，本轮事前方案详见 [THOUSAND_MAP_EXPERIMENT.md](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/finite_contact/THOUSAND_MAP_EXPERIMENT.md)。旧文档中的“等待前台恢复”是历史交接状态，用户已恢复训练。

## 1. 用户目标与工作约定

用户是多机器人科研初学者，研究方向为通信受限下的协同探索，希望在已有论文与源码基础上做有实验证据的方法改进，形成一篇小论文。目标期刊曾考虑 SCI 3／4 区，但不能提前承诺发表或将现有开发结果当成论文已成立。

以 `/home/wyz/wjc_ws/IR2-Continuous-Connectivity` 为主要代码基础。其他论文及源码用于理解、借鉴和后续合理比较；不是直接拼接两个仓库。

用户已要求并正在执行：先两机器人、简单地图验证可学习性，再扩大训练地图数量；当前直接使用 1000 张简单训练地图。用户希望前台监督训练，同时通过 TensorBoard 看曲线。解释应使用通俗中文，区分代码正确、能学习、验证有效和论文贡献成立。

新对话的即时任务：检查当前实验状态，保持已有训练继续完成，核验三方法固定终点及其开发评估，整理结果并据此推进。不要把交接理解成重新从零开始实验。

## 2. 我们研究什么

核心问题：连接不仅会中断，而且即使连接上，短暂接触和有限带宽也未必足够传完地图。机器人需要共同考虑“到哪里会合、优先传什么、等多久、何时离开”。

拟验证两个相互配合的模块：

- 模块 A：按预计可交付的有用信息收益，引导隐式会合与等待／离开决策。
- 模块 B：对地图增量分包，学习优先传递更有用的内容；预计交付收益反馈给移动决策。

支撑环境按字节与物理时间计费，支持排队、部分交付、确认、显式等待。容量模型和训练工程本身不能代替方法创新。当前距离模型不等于已经验证墙体遮挡下的真实无线通信。

目标证据：同样地图、传感器、移动能力、通信资源和训练预算下，改进能提高任务完成率，或在完成率相当时缩短探索与同步时间。部署输入只能来自本地观测与实际收到的信息；不能用免费全图同步、队友真实地图或放宽成功标准获得优势。

参考论文位于 `/home/wyz/wjc_ws/papers/`：

- `Deep Reinforcement Learning-based Large-scale Robot Exploration.pdf`
- `Implicit Rendezvous for Robotic Exploration Teams underSparse Intermittent Connectivity.pdf`
- `Privileged Reinforcement and Communication.pdf`

三者有共同技术背景，但不能简单断言后两篇是前一篇的直接改写。我们的现行 Actor／Critic 使用局部观测，不能描述成已经实现完整地图特权 Critic。文献新颖性仍需后续系统核对。

## 3. 环境、分支和代码状态

| 项目 | 固定位置或配置 |
| --- | --- |
| 工作目录 | `/home/wyz/wjc_ws/IR2-Continuous-Connectivity` |
| 分支 | `research/finite-contact-exploration` |
| 交接建立时 HEAD | `d34855a4`，此前单图版本；接手以 `git log -1` 查看后续提交 |
| Conda 环境 | `/home/wyz/anaconda3/envs/ir2_train` |
| Python | `/home/wyz/anaconda3/envs/ir2_train/bin/python` |
| 软件 | Python 3.8.20、PyTorch 2.3.1+cu121、CUDA runtime 12.1 |
| GPU | NVIDIA GeForce RTX 3090，24 GB |
| 活跃实验目录 | `experiments/thousand_maps_20261012_seed1/` |

交接建立时，100 图后续实验、1000 图入口和 TensorBoard 工具等尚有大量已修改／未跟踪文件。随后用户要求提交并推送本轮源码与文档；接手时读取 `git status` 与 `git log -1` 确认实际版本，不能用旧单图提交代表多图执行源码。实验目录保存执行时的源码快照、diff 与哈希；本轮身份文件为实验根目录的 `source_identity_before.json`，不是 `checks/` 下。保留未提交内容，不清理工作区或切分支覆盖修改。

用户要求只提交并推送本仓库的代码，单图版本已完成；本次后续提交涵盖多图源码、测试、文档及小型证据。仓库外 papers 文档副本不属于该 Git 仓库；`preview_*` 等其他未跟踪文件、模型和原始实验输出保留本机。

## 4. 已完成的阶段与结论

| 阶段 | 已有证据 | 仍有限制 |
| --- | --- | --- |
| 单图可学习性 | 两机器人能在一张简单图完成探索，保存与真实续训可用 | 当时开发验证 0/2，换图问题明显 |
| 100 图移动学习 | 同一 20 张开发图，起点 7/20 → 终点 18/20 | 单种子，只有移动阶段 |
| 100 图通信头学习 | 5000 次通信头更新；固定状态短接触内容收益优于控制 | 冻结移动评估：FIFO／规则／学习成功 18／19／19，学习时间未超过规则 |
| 三组移动适应 | 首个种子：FIFO／规则／学习成功 18／19／20，平均 176.503／150.472／125.480 秒 | 后续种子暴露完成率波动，不能只报首个好结果 |
| 稳定性与抽样复现 | 比较 pooled 与候选 map_tail，所有终点、退步和失败保留 | 候选抽样没有一致优势，默认仍 pooled |

最近完成的 100 图追加适应实验，默认 pooled 的三个适应种子结果如下。每个单元格为“成功张数／20，平均秒”；失败计 256 秒。

| 方法 | seed=1 | seed=2 | seed=3 |
| --- | --- | --- | --- |
| FIFO | 19，159.022 | 18，180.548 | 18，176.603 |
| 规则 | 20，147.114 | 18，147.725 | 20，144.360 |
| 学习选包 | 20，114.473 | 18，123.100 | 20，131.735 |

这些结果说明有探索速度收益的开发迹象，仍不是独立测试或最终创新结论。三个种子共享预训练移动模型和通信头，不是三条从头独立训练的完整链路；同一 20 张开发图多次使用。map_tail 虽在三个学习终点均为 20/20，但确认种子的规则完成率下降、一个学习终点变慢，因此没有切换默认。

该轮主结果 2700 条轨迹、318408 段移动检查通过，穿墙 0；资源中断重做另行统计。详细结果与模型在 [STABILITY_REPLICATION_EXPERIMENT.md](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/finite_contact/STABILITY_REPLICATION_EXPERIMENT.md) 和 `experiments/stability_replication_20261012/`。本轮 1000 图启动前 67 项行为检查通过，另 3 项压缩／实际学习／恢复等价检查通过。

## 5. 当前 1000 图实验的固定计划

本轮问题：扩大简单地图上的移动适应，是否能保持或改善开发表现，并在同样预算下优于对照？

| 项目 | 本轮设定 |
| --- | --- |
| 地图 | `simple_rooms_v1`，256×256，两／三房间，尺寸、门、障碍和方向变化 |
| 数据划分 | 1000 训练／20 开发／20 留出测试，数据种子 20261011 |
| 地图身份 | 原 100 张训练保留，新增 900；原开发／测试身份不变；1040 图质量和去重审计通过 |
| 机器人 | 2 台 |
| 通信 | 距离 65，2048 B/s/发送方，包上限 256 B |
| 成功标准 | 两台机器人各自本地地图覆盖率均 ≥99% |
| 任务时限 | 256 秒；失败按 256 秒计入平均时间 |
| 三方法 | `AB_learned` 学习选包、`A_rule` 规则选包、`A_fifo` 先入先出 |
| 起点 | 共同移动 SAC=8895、通信头更新=5000；继承网络及 Adam，各组清空旧回放和成功缓存 |
| 训练 | 同样乱序遍历 1000 图一次；每图完整 sampled 探索后恰好 20 次真实 SAC，图内不更新 |
| 预算 | 每组新增 20000 次，累计终点 28895；三组共新增 60000 次 |
| 学习率 | Actor 1e-5；Q／温度 1e-4 |
| 其他 | batch=16，embedding=64，replay=12000，成功经验按地图保留 1000 槽，抽样 pooled |
| 资源 | 单 worker，学习 → 规则 → FIFO 顺序运行；每组 12 小时只作异常保护 |
| 保存 | 每 10 回合完整 `latest.pth`，每 50 回合轻量权重，结束保存 `final.pth` |
| 评估 | 每组固定终点评估同一 20 张开发图；不中途按开发分数挑模型；测试策略执行 0 |

**范围必须说明：通信头与上下文编码器冻结，沿用原 100 图训练版本。当前是 1000 图移动适应，不是通信头也在 1000 图上重新训练，更不是全部从随机权重开始。**

100 图与 1000 图实验的更新预算不同，结果差异不能单独归因于地图数量。若以后研究地图规模的单独影响，需要另做更新预算匹配的对照。

二进制邻接掩码在 CPU 回放中无损按位压缩，进入网络前还原；没有减少地图或改变输入数值。原先多 worker 曾导致系统 RAM OOM，所以本轮固定单 worker。显存较少不代表没有用 GPU：已核对训练 PID 3999727 的命令含 `--device cuda`，显存约 1002 MiB；曾观察整卡瞬时利用率 97%，不代表全程平均值。

## 6. 当前进度快照与中断记录

2026-10-12 15:13 北京时间的只读快照：

| 方法 | 状态 | 完整训练回合 | 累计 SAC | 最终开发评估 |
| --- | --- | --- | --- | --- |
| `AB_learned` | 用户前台训练中 | 157／1000 | 12035 | 尚未执行 |
| `A_rule` | 排队 | 0／1000 | 尚未启动 | 尚未执行 |
| `A_fifo` | 排队 | 0／1000 | 尚未启动 | 尚未执行 |

学习组 157 回合中成功 151 次；最近 50 回合成功 49 次，平均仿真时间 129.916 秒。它们是变化策略在不同训练图上的在线记录，不是固定模型验证，也不是训练收敛证明。该时刻尚无 `final.pth`。当前真实进度以日志重新读取为准。

原后台编排器 PID 3884713 已按用户前台监督要求停止，之后用户已执行前台脚本并恢复训练。旧 `background_process.json` 中的死亡 PID 不能用来判断现在没有训练。

切换前台时，完整断点为 episode=50／SAC=9895；原先已执行的 51～56 回合与 120 次更新另行归档，前台从 51 重做，不重复计入保留预算。记录为 `checks/foreground_transfer.json` 与 `foreground_transfer_1791787850506691461/`。新对话最终验收时应核对重做记录的等价性并单列额外计算，尚不能将该项写成已经通过。

PID 是历史定位线索，接手时需要核对 `/proc/<pid>/cmdline` 或宿主机进程列表，不能只按 PID 数字操作。

## 7. 新对话首先执行的检查

先阅读本文，然后执行只读检查：

```bash
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
/home/wyz/anaconda3/envs/ir2_train/bin/python \
  experiments/thousand_maps_20261012_seed1/checks/progress.py
tail -n 3 experiments/thousand_maps_20261012_seed1/AB_learned.log
ps -eo pid,ppid,etime,args | rg 'finite_contact_multimap_joint.py|run_thousand_maps_foreground.sh'
nvidia-smi
```

结合 `<组>/status.json`、`training.jsonl`／`updates.jsonl` 的增长及实际进程判断状态；`pipeline.json` 的 running 标志本身不足以证明进程还在。`checks/progress.py` 的 `checkpoint_or_completed_boundary` 来自状态文件，可能比磁盘 `latest.pth` 新，不能把它当作已保存断点。

在 Codex 沙箱中，`/proc` 可能只能看到沙箱进程，`nvidia-smi` 可能报无法连接驱动。应使用允许的宿主机只读检查核对，不能据此判定 CUDA 环境坏了。不要启动额外训练来“检查 GPU”。

**训练仍在运行时：保留用户前台窗口，不启动第二套训练，不停止它，不修改现有训练源码。** `source_identity_before.json` 记录了以下 11 个文件，恢复和终点审计都会校验：

```text
finite_contact_scale.py
finite_contact_multimap_joint.py
finite_contact_train.py
finite_contact_multiseed.py
finite_contact_stability.py
finite_contact/config.py
finite_contact/learning.py
finite_contact/environment.py
finite_contact/maps.py
finite_contact/sensing.py
finite_contact_dataset.py
```

本次交接检查 11 文件哈希全部匹配。其他实际依赖也不应在训练中静默改动。可以继续只读分析、补文档或添加独立日志工具；需要算法修改时用独立后续实验，并保存当前证据。

只有确认现有训练确实退出后，恢复命令才是：

```bash
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
bash run_thousand_maps_foreground.sh
```

脚本固定解释器、文件锁和源码校验，自动从同组 `latest.pth` 恢复，跳过已完成组，依次运行三组并最终汇总审计。`--dry-run` 仅显示命令。最多重做最后 9 个尚未保存的完整回合；额外计算保留归档。不要直接重新初始化同一输出目录，也不要删除哈希文件绕过检查。

## 8. TensorBoard 与指标解释

用户已在浏览器 `http://localhost:6006` 查看曲线。当前原训练入口写 JSONL；独立工具 `finite_contact_tensorboard.py` 只读转写，每 5 秒轮询，不改变训练。

实时转写进程的历史 PID 为 4043687，元数据与日志：

- `checks/tensorboard_mirror_process.json`
- `checks/tensorboard_mirror.log`

若该工具已退出，确认没有其他转写进程后可在独立终端运行：

```bash
conda activate ir2_train
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
CUDA_VISIBLE_DEVICES="" python -u finite_contact_tensorboard.py \
  --experiment experiments/thousand_maps_20261012_seed1
```

TensorBoard 服务器命令，在另一个终端执行；已有服务器可继续使用：

```bash
conda activate ir2_train
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
tensorboard --logdir experiments/thousand_maps_20261012_seed1/tensorboard_live --port 6006
```

- `<组>/train` 横坐标是本次训练 episode，从 1 开始；第 0 回合没有采样记录。
- `<组>/sac` 横坐标是累计更新次数，从 8896 开始，因为加载模型已有 8895 次历史更新。
- `train/success_rate_last_50` 是最近最多 50 回合的成功比例；前 50 回合之前分母是已有回合数，不是固定 50。
- `train/time_mean_last_50` 是仿真任务时间，失败为 256 秒；不是实际训练墙钟耗时。
- `train/min_robot_coverage` 是回合结束时两机较低覆盖率；成功回合接近 99% 是终止规则决定的，不是过程曲线恒定。
- `train/coverage_deficit` 是探索过程覆盖缺口的时间积分除以 256，越低越好；不是 `1 - 终点覆盖率`。
- `loss/policy` 包含 SAC 项与成功经验辅助项，可能为负；不能按“负数更大／更小”直接判断任务表现。
- 用户截图曾设置 Smoothing=0.999，容易掩盖波动；建议先设 0 看原始值，滚动成功率已平滑过。

日志核对工具测试 2 项通过；曾逐点核对至 95 回合／1900 次更新，20 个事件标签与 JSONL 一致，错值 0、重复 step 0。报告 `checks/tensorboard_audit.json`，诊断图 `checks/tensorboard_audit.png`。这是历史快照，不是最终结果。

## 9. 当前实验完成后的验收顺序

1. 确认三组都完整执行 1000 回合、各新增 20000 次真实更新、累计 SAC=28895。核对所有实际优化器步数、共同初始网络／Adam、地图调度、通信头与上下文冻结、各组自身缓存。
2. 每组完成 20 张固定开发图的 greedy 评估，得到完整 `validation_final/evaluation.jsonl`。训练终点存在 `final.pth` 不等于评估已经结束，需看 status 与记录完整性。
3. 前台脚本最后调用 `finite_contact_scale.py --resume`，运行原汇总审计并保存 `report/summary.json`、`report/matched_adaptation.png`／`.pdf` 与 `source_identity_after.json`。检查总状态 complete 与审计成功，不能只看训练终端最后一行。
4. 从轨迹／原始地图重算覆盖率、穿墙、速度、时限和通信容量。完整预算预期主执行为 3000 条训练轨迹与 60 条开发轨迹；归档重做单列。保存失败地图及探索不足／同步不足的分类。
5. 比较三组成功数、失败计 256 秒的平均时间、覆盖缺口及双方均成功地图上的配对时间差；完成率与效率一起看。报告所有负结果，不能删除失败或只报告成功任务用时。
6. 整理本轮计划、源码、数据身份、测试记录、结果图表与小型证据，更新本文和实验文档，再提出下一轮具体计划。保留权重及原始轨迹，避免把大回放／模型误当成小型报告推送。

当前结果不能提前预测。若学习方法最终优于规则且完成率相当或更好，得到单种子开发支持；若退步或未超过规则，先定位探索、同步、奖励、回放／自模仿、头与移动适配等问题，用训练侧证据设计下一轮公平实验。不要针对开发失败地图写专用路线，也不要无条件加训练次数。

## 10. 后续科研实验路线（本轮之后另定具体预算）

以下是待办研究计划，当前脚本不会自动执行：

1. **确认收益来自哪里。** 同预算消融会合收益反馈、学习内容选择以及二者耦合；保留合法规则基线。现有 `A_fifo/A_rule/AB_learned` 不能独自证明模块 A 的独立贡献，也不能直接视为与原版 IR² 的完整公平比较。
2. **确认通信受限时是否有用。** 事前固定不同通信容量／接触条件，报告完成率、任务时间、有效交付与等待代价，不能改变资源后混合排名。
3. **确认训练稳定性。** 增加从预训练移动、通信头到适应均独立的随机种子；现有共享起点的适应种子不能替代。
4. **确认泛化。** 方法和选模协议固定后使用留出测试；后续研究可增加独立同分布地图和不同拓扑。现有 20 张测试图从未执行策略，保持留出；扩大评估规模需预先写明划分和协议。
5. **扩展问题难度。** 依据两机简单地图证据，再考虑复杂地图、更多机器人、遮挡／信号模型与丢包。用户当前优先简单图验证，不直接启动原版大规模复杂训练。
6. **整理论文。** 对照当前文献核对新颖性；将通信贡献、共同训练辅助和环境工程区分清楚，用独立测试、多种子与消融支持最终主张。

## 11. 关键文件导航

| 内容 | 仓库相对路径 |
| --- | --- |
| 完整研究设计 | `finite_contact/RESEARCH_DESIGN.md`；外部原稿 `/home/wyz/wjc_ws/papers/有限通信窗口下多机器人探索_研究与实验设计.md` |
| 本轮固定实验计划 | `finite_contact/THOUSAND_MAP_EXPERIMENT.md` |
| 当前实验计划与来源 | `experiments/thousand_maps_20261012_seed1/pipeline.json`、`pre_run_design.md`、`source_identity_before.json` |
| 只读进度 | `experiments/thousand_maps_20261012_seed1/checks/progress.py` |
| 训练／恢复入口 | `run_thousand_maps_foreground.sh`；`finite_contact_scale.py` 编排，`finite_contact_multimap_joint.py` 实际 worker |
| 当前模型 | `experiments/thousand_maps_20261012_seed1/<组>/latest.pth`；结束后同组 `final.pth` |
| 日志／物理证据 | 同组 `training.jsonl`、`updates.jsonl`、`trajectories.jsonl`、`communication.jsonl`、`training_map_*.npz` |
| 本轮最终结果 | `experiments/thousand_maps_20261012_seed1/report/summary.json`，完成后才可使用 |
| 100 图移动起点 | `experiments/multimap_20261011_vectorized_seed1/movement_A/latest.pth` |
| 冻结通信头来源 | `experiments/multimap_content_20261012_seed1/head/final.pth` |
| 旧稳定性完整结果 | `finite_contact/STABILITY_REPLICATION_EXPERIMENT.md` 与 `experiments/stability_replication_20261012/report/summary.json` |
| 小型证据 | `finite_contact/results/`，当前 1000 图子目录只是启动证据，不是终点结果 |

## 12. 用户可复制给新对话的话

> 请先阅读 `/home/wyz/wjc_ws/papers/多机器人有限通信探索_实验计划与新对话交接.md`，再检查 `/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/thousand_maps_20261012_seed1/` 的实时状态。我们的目标是在 IR² 基础上研究有限通信窗口下的会合与信息选择，现在正在 ir2_train 环境中做两机器人、1000 张简单地图的三方法等预算移动适应。用户已在前台启动训练，请先确认进程，不要重复启动或修改正在执行的训练源码；接着完成现有实验的监控、固定终点评估、审计和结果分析，再按交接文档推进下一轮。
