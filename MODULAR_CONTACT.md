# 原IR2训练器上的有限通信模块

当前分支 `research/ir2-modular-contact` 基于已推送的 `baseline/ir2-easy10` / `dcea459d`。
原版运行 `2` 已由用户暂停：1394有效回合、0无效回合、11104次SAC；模型、日志及十图清单保留本地。
此分支只完成模块接入与隔离工程检查；尚未开始新的正式训练，也没有证明该接入版优于基线。

## 当前目标与边界

先在与运行2逐字相同的10张原easy图上，固定3台机器人，确认改进模块接入原训练器后能学习。
保留原图网络6维输入/128嵌入、原SAC数学更新、LR=1e-5、batch128、replay10000、热身2000、每回合8批更新、196步上限。
继续使用无损压缩回放及RAM保护；没有搬回旧研究的环境、SAC/SIL训练器或地图生成器。

这是以IR2为主体、借鉴BLMRE信息价值思想的第一版迁移，不是完整BLMRE网络复现。
旧研究的正结果不能直接视作本版结果。消息头结构与12项本地特征沿用旧实现，图上下文改为当前IR2的128维冻结编码。
当前实验仅验证学习链路，后续对照必须使用相同通信机制、动作空间和训练预算。

## 使用方式

先设置conda环境，并进入项目目录。启动第一份改进组（随机初始化，不加载原版模型）：

```bash
conda activate ir2_train
cd ~/wjc_ws/IR2-Continuous-Connectivity
IR2_METHOD=finite_learned IR2_RUN_NAME=contact_easy10_1 IR2_AGENTS=8 python -u driver.py
```

模型 `model/contact_easy10_1/checkpoint.pth`；曲线 `train/contact_easy10_1/`；记录 `experiments/contact_easy10_1/`。
已有同名检查点时拒绝覆盖；中断后继续相同组：

```bash
IR2_METHOD=finite_learned IR2_RUN_NAME=contact_easy10_1 IR2_AGENTS=8 IR2_RESUME=1 python -u driver.py
```

Ctrl+C等待当前优化块完成再保存。沿用原版限制：移动回放和在途任务不存盘，重启须重新采集2000经验；
新增的消息头参数、优化器、辅助回放、冻结编码器、更新计数及辅助抽样RNG会保存。
不允许把原版检查点直接当作finite_learned断点，不允许续训时静默改变通信配置。
原版 `model/2` 不覆盖、不自动加载；如后续需要共享热启动，另行设计两组一致的初始化方案。

```bash
tensorboard --logdir train/contact_easy10_1 --host 127.0.0.1 --port 6011
```

浏览器 http://127.0.0.1:6011 。重点看 `Progress/Success Rate 100 Percent`、`Perf/Explored Rate`、
`Contact/Content Updates`、`Contact/Content Loss`、`Contact/unique_cells` 和 `Contact/wait_actions`。
`Perf/Reward`是采样即时奖励均值；`Perf/Connectivity Rate`是回合末连接状态，不是全程连接率。

仅检查配置、不启动采样/梯度：

```bash
IR2_METHOD=finite_learned IR2_RUN_NAME=contact_easy10_1 IR2_AGENTS=8 python driver.py --check-config
```

`IR2_METHOD`默认仍为`original`，其默认目录也是原版目录。扩展组默认使用独立目录，推荐显式指定运行名。
`finite_fifo`和`finite_rule`可用于后续受控对照；本轮不同时启动两份正式训练。
`IR2_BYTES_PER_SECOND`默认2048；`IR2_CONTACT_COUPLED=0`只关闭收益估计中内容/容量耦合，实际传输仍有限且沿用指定内容策略。
有限通信各组均可WAIT、同样支付时间/真实交付奖励；FIFO使用未耦合的保守待发地图差异，会合信号不使用学习得分。

## 独立模块与接入点

| 文件 | 职责 |
|---|---|
| `contact_module/config.py` | 模式与通信参数、检查点兼容性检查 |
| `contact_module/radio.py` | N机器人分包、直接链路、发送/接收预算、位姿、ACK、断连续传 |
| `contact_module/learning.py` | 12维特征＋冻结图上下文消息头、独立辅助优化与恢复 |
| `contact_module/adapter.py` | 本地特征、信息排序、预计交付收益、IR2状态适配 |

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
- 所有限通信组额外奖励为本次真实交付新格数/1000减0.02×动作耗时。保留原探索/会合/距离奖励和成功条件；逐决策gamma仍为原.995。
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

这些是工程验收，不是性能实验，也不计作正式回合。正式训练仅由用户前台启动；无周期评估、自动停止或后台守护。
下一步先训练一份finite_learned确认成功率/覆盖率上升，再讨论有限通信对照、相同预算比较和独立评估。
当前接入范围是训练入口；原`test_driver.py`尚未适配有限通信。后续评估必须复用同一模块与配置，
不能把有限通信模型直接放进原版免费交换的测试环境并报告为本方法成绩。

实际验收结果（2026-10-18）：全部10张原图在8个Ray worker上完成196步上限的完整回合检查（允许任务提前成功），共5820条真实经验；原GPU batch128连续8次更新、辅助头40次更新均完成。主机可用RAM最低约17.79GiB，GPU张量峰值约17.66GiB。最终源码SHA、原模型/日志SHA及地图一致性见`baseline_checks/contact_final_review.json`。正式训练仍未启动。
