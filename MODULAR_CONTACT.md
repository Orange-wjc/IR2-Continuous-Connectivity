# 原IR2训练器上的有限通信模块

当前分支 `research/ir2-modular-contact` 基于已推送的 `baseline/ir2-easy10` / `dcea459d`。
原版运行 `2` 已由用户暂停：1394有效回合、0无效回合、11104次SAC；模型、日志及十图清单保留本地。
v1运行contact_easy10_1已在2036有效回合后退出，近100成功率50%，尚未证明优于有限通信FIFO。当前修复版本为schema_version=6；旧版本检查点不可静默续训为新的决策经验语义。

## 当前目标与边界

先在与运行2逐字相同的10张原easy图上，固定3台机器人，确认改进模块接入原训练器后能学习。
保留原图网络6维输入/128嵌入、原SAC数学更新、LR=1e-5、batch128、replay10000、热身2000、每回合8批更新、196步上限。
继续使用无损压缩回放及RAM保护；没有搬回旧研究的环境、SAC/SIL训练器或地图生成器。

这是以IR2为主体、借鉴BLMRE信息价值思想的第一版迁移，不是完整BLMRE网络复现。
旧研究的正结果不能直接视作本版结果。消息头结构与12项本地特征沿用旧实现，图上下文改为当前IR2的128维冻结编码。
当前实验仅验证学习链路，后续对照必须使用相同通信机制、动作空间和训练预算。

## 当前v6：发送者真实交付奖励与下一决策状态

v5保存停止于527有效/0无效、SAC7880/消息头5036、近100成功64%。冻结轨迹确认发送者在其它机器人时隙的交付奖励遗漏，以及提前封存的下一状态与下次实际决策不一致。v6单独模块contact_module/transitions.py按每机自身决策区间结算全部真实发送收益，下一状态取下一实际决策观察；末条在全部动作/付费交换后封存。原SAC、6维输入、各奖励系数、真实动作/协议/成功判据不变，模块关闭时仍逐项等价。

显式升级工具baseline_checks/upgrade_contact_transition_checkpoint.py保留全部已学权重、优化器、辅助回放及随机状态，新条件计数/验收从0重新开始。运行contact_easy10_v6_20261019；每300回合复查、达到297/300前继续。详见[CONTACT_LEARNING_PLAN.md](CONTACT_LEARNING_PLAN.md)与baseline_checks/contact_transition_verified.json。工程通过不代表99%目标达成。

## 历史v5：本地图效用缓存修复

v4在405有效/0无效回合因确认缓存错误而由助手保存停训：累计SAC3728、消息头2928。300模型配对冻结轨迹中，9999机器人2未访问可达节点(52,156)的缓存效用为0；同一本地地图/前沿重新计算应为7，整张图因此漏掉唯一正探索目标。额外诊断前后588个动作和奖励逐项一致，排除了诊断扰动。

v5在付费收图后的本地图/图更新边界恢复未访问零效用节点，已访问节点保持零；按本地已知地图增长避免重复扫描。仅读自身信息，不修改原版关闭模块路径。Contact/utility_recoveries记录恢复次数，不能当学习效果。运行名contact_easy10_v5_20261019，显式升级工具baseline_checks/upgrade_contact_utility_checkpoint.py保留v4已学状态与辅助RNG，继承3728次SAC/2928次头更新；新语义回合/验收清零，旧日志完整保留、辅助旧回放清空。不是从零重训或公平独立消融；未达到99%目标。完整执行与验收见[CONTACT_LEARNING_PLAN.md](CONTACT_LEARNING_PLAN.md)。

## 历史v4：空等待奖励修复与自动训练复查

执行计划见[CONTACT_LEARNING_PLAN.md](CONTACT_LEARNING_PLAN.md)。用户最新指令已授权自动启动、每300回合复查、必要时保存暂停再修复，直到真实连续300尝试至少297成功。此前“只由用户前台启动”和“不自动暂停”要求在此流程中已更新。

v2保留坐标/路径/预训练修复，但741回合冻结完整轨迹证明：9999出现9次、9995出现7次无新增知识和真实交付仍获正奖励的WAIT。v3恢复原版IR2自身30像素附近会合值清零，其它奖励与原SAC更新保留。配对冻结诊断空等正奖励归零，但旧模型仍失败；不能把工程修复当作已达99%。

v3新增统计：实际并集、最小个体覆盖、空等待正奖励次数和奖励分量。仅供回合末统计，真值不进入观察/消息评分/动作。成功条件仍是每台机器人自身覆盖≥99%。schema_version=4拒绝把v2模型直接resume成新奖励版本。

v3在77回合/552 SAC后发现初始自身位置未同步，9997实际自位置距误用的共同起点约62/55像素，使清零中心偏移。v4在Worker构造与观察边界同步自身已知位置，队友位姿仍只用共同先验/付费消息；增加逐动作异常诊断并遇到空等正奖励立即保存暂停。

当前v4从已停训的v3显式升级：保留actor/双Q/温度、优化器/学习率状态、冻结context、消息头及RNG；累计SAC继承552、头继承1308。新条件回合、各图访问及验收窗口重置，辅助回放清空。旧77回合和全部失败保留，未筛选；不以新旧混合窗口验收。初始权重与优化器逐张量审计见contact_pose_upgrade_verified.json。升级准备工具为baseline_checks/upgrade_contact_pose_checkpoint.py，不允许普通resume静默跨schema。

## 自动训练使用方式

助手将按用户指令自动启动唯一实例，启动命令记录如下；**已有实例运行时不要重复执行**：

```bash
conda activate ir2_train
cd ~/wjc_ws/IR2-Continuous-Connectivity
IR2_METHOD=finite_learned IR2_RUN_NAME=contact_easy10_v6_20261019 IR2_AGENTS=8 \
IR2_RESUME=1 IR2_AUTO_REVIEW=1 python -u driver.py
```

新模型model/contact_easy10_v6_20261019/checkpoint.pth；曲线train/contact_easy10_v6_20261019/；日志experiments/contact_easy10_v6_20261019/training.log。
中断后续训移除两个初始化参数，使用：

```bash
IR2_METHOD=finite_learned IR2_RUN_NAME=contact_easy10_v6_20261019 \
IR2_AGENTS=8 IR2_RESUME=1 IR2_AUTO_REVIEW=1 python -u driver.py
```

目标门槛是连续300真实尝试至少297成功、无无效尝试、10图各至少20次，保存原始窗口证据acceptance.json并保存退出为target_reached。
成功率用真实计数，不用平滑结果；窗口存入检查点，续训保留。此为训练门槛，不等同独立测试泛化。

每300有效回合自动保存检查点、输出reviews/episode_NNNNNN/{README.md,review.json,curves.png}，写入TensorBoard Review/*。首窗比较最初/最后100，之后比较前后300，兼顾各图最近20、本地覆盖、WAIT和异常。±3个百分点仅作描述性提示，不是显著性判断。
IR2_AUTO_REVIEW=1时，如连续3个无效回合、周期发现空等正奖励/无效/非有限记录，或600回合后无明确增长且仍有长期失败/退化地图，会保存退出为paused_for_diagnosis。程序不修改源码或参数；后续由当前Codex活动目标继续诊断与受控修复。达到99%以前不得把这种暂停误报成完成。

```bash
tensorboard --logdir train/contact_easy10_v6_20261019 --host 127.0.0.1 --port 6013
```

重点看Progress/Success Rate 100 Percent、Review/*、Maps/*/Success Last10 Percent、Contact/positive_empty_waits、Contact/min_local_coverage、Contact/union_coverage、Contact/reward_*及Content Loss/Updates。
Perf/Explored Rate始终是平均个体覆盖，不是并集；Perf/Reward是采样回放批次奖励均值，v6按自身决策区间结算交付。连接指标是回合末状态。
Contact/reward_delivery为全部有动作归属的实际首次发送交付收益；own_slot_delivery_reward保留旧时隙口径供排查。reward_delivery×1000加initial_delivery_cells应等于unique_cells，delivery_ledger_residual必须为0。

IR2_INIT_FROM/IR2_CONTENT_INIT只用于新实验，与resume互斥。自动控制开关会随检查点保存，省略环境变量时沿用；显式IR2_AUTO_REVIEW=0关闭自动暂停/达标停机，仍保留300回合报告。原版模式默认不启用自动控制。
移动回放及在途任务仍不存盘，续训先重新收集2000经验；辅助模型/优化器/回放/RNG和验收窗口恢复。

## 独立模块与接入点

| 文件 | 职责 |
|---|---|
| `contact_module/config.py` | 模式与通信参数、检查点兼容性检查 |
| `contact_module/radio.py` | N机器人分包、直接链路、发送/接收预算、位姿、ACK、断连续传 |
| `contact_module/learning.py` | 12维特征＋冻结图上下文消息头、独立辅助优化与恢复 |
| `contact_module/adapter.py` | 本地特征、信息排序、预计交付收益、IR2状态适配 |
| `contact_module/transitions.py` | 自身决策区间、发送者交付奖励守恒、实际下一状态 |

原文件只有明确接口：Env在信息传播处选择原路径/有限传输；Worker采集冻结本地图上下文并设置WAIT候选；
Runner同步辅助权重；driver注册辅助更新与保存；Policy/Q可选允许WAIT；Progress增加通信指标和源码SHA。
关闭扩展时保留原路径，不更改原SAC公式。原版模式与已推送基线的同种子短轨迹，15列经验和全部回合指标完全相同。

## 物理与信息约束

- 保留原信号强度模型判定直接链路。连通组不再免费合并地图、位姿、路由历史或其他队友信念；无免费多跳。
- 每节点总TX和总RX各2048字节/秒，采用全双工预算；三台机器人不是六条边各得到独立完整额度。
- 512字节上限包、16格分片；位姿、ACK和重试都计字节，整包完成才可用。部分包在断连后保留；断连不积攒传输额度。
- 所有链路本次服务完成后才应用收到的数据，因此不存在本时段免费转发。下一时段转发仍须重新排队、付字节。
- 采用保守的串行动作端点模型：移动耗时=地图距离/12，移动期间不按终点链路赚取容量；到达后固定驻留1秒通信。
  WAIT仅支付这1秒。并非真实并行动作仿真或旧实验物理时间的精确重现，因此不混算旧时间结果。
- 所有限通信组按发送者自身决策区间结算真实交付新格数/1000，减0.02×自身动作耗时。保留原探索/会合/距离奖励和成功条件；逐决策gamma仍为原.995。
- 接收位姿保留在通信状态中；只把本地已知自由区域、且存在可用图采样点的位姿交给原图生成器，避免原版整数裁剪/空候选假设失效。
- 决策只读发送方本地地图、已付费位姿和ACK确认知识；仿真真实位置只用于无线服务，接收方完整当前地图只用于辅助监督标签和统计。
- 消息头预测每包对接收方新增格数，采样时按每字节价值排序并以上限裁剪；每次最多64候选，独立采样reservoir每回合256标签，避免辅助数据失控。
  原始包特征的地图坐标按当前地图尺寸归一化；不加载旧两机/64维上下文模型权重。
- 会合估计考虑待发包、未完成包、字节/时间预算、开销及陈旧位姿，沿用原单一会合输入通道。多队友采用保守等份预算估计；真实服务仍逐字节核算。
- 模块的WAIT是候选第0项真实当前节点，移动候选最多29个；Policy/Q共同允许它，图节点编号0也不再被误当有效动作填充。
  全部有限通信对照采用同一动作约束，原版仍保留其原动作规则。

## 工程检查与下一步

`baseline_checks/check_contact_module.py`检查：带宽汇总预算、断连/整包可见性、ACK计费、无同周期转发、原版关闭模式等价、
排序/收益不依赖接收方真值、WAIT、辅助梯度、优化器/RNG续训等价、正式文件SHA保持。
`check_contact_runtime.py`检查：8个真实Ray worker、三机器人原图经验、原batch128×8 GPU更新及内存。
`check_contact_driver.py`检查：真实driver在临时目录启动/辅助更新/SIGINT保存/续载、配置误用拦截和Ray清理。
对应`*_verified.json`记录实际结果；完整196步检查另记`contact_full_runtime_verified.json`。

这些是工程验收，不是性能实验，也不计作正式回合。当前v6按用户最新指令自动训练，每300回合复查，遇实现错误保存停训后修复；达到连续300回合至少297成功时保存退出。没有启动独立测试集周期评估。
下一步先训练一份finite_learned确认成功率/覆盖率上升，再讨论有限通信对照、相同预算比较和独立评估。
当前接入范围是训练入口；原`test_driver.py`尚未适配有限通信。后续评估必须复用同一模块与配置，
不能把有限通信模型直接放进原版免费交换的测试环境并报告为本方法成绩。

实际验收结果（2026-10-18）：全部10张原图在8个Ray worker上完成196步上限的完整回合检查（允许任务提前成功），共5820条真实经验；原GPU batch128连续8次更新、辅助头40次更新均完成。主机可用RAM最低约17.79GiB，GPU张量峰值约17.66GiB。最终源码SHA、原模型/日志SHA及地图一致性见`baseline_checks/contact_final_review.json`。这是2026-10-18的v1接入验收记录，不能替代v2修复后的学习验收。
