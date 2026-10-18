# 原项目地图迁移排查（2026-10-17）

结论：A、B及耦合的核心逻辑已接入当前实验并实际参与决策；尚未发现开关没开、学习选包退回FIFO、检查点未更新或当前节点/动作索引错位这类直接接线故障。但预实验中验证过的整套学习方案没有等价保留，仓库根入口也未完成整合。因此只能称“原图上的实验迁移可运行”，不能称“只换了地图且已验证等价”。目前还没有证据把性能落差归因到某一个具体差异。

本次状态：训练及监督保持暂停，两组10060回合/80632次SAC。只读核对源码、配置、检查点与既有回放；没有完整地图策略执行、没有梯度更新。允许在内存中运行函数夹具和已有观测的离线前向，未改训练代码、配置或模型文件。核对时间2026-10-17T09:03:34.072913+08:00。

## 已确认正确的部分

| 项目 | 证据与结果 |
|---|---|
| 版本真实 | 原独立种子实际执行快照54文件、当前执行快照60文件逐个SHA一致；基于执行快照对比，未把仓库里未使用的旧文件当作当前训练源码。 |
| A会合价值 | expected_delivery、价值沿图传播的主要逻辑仍接在observation的AB分支；检查点实际variant=AB。 |
| B学习选包 | packet_scores、rank_packet、Learner.score的核心函数AST逐项相同；学习分支通过feature_provider与冻结头打分，未降级FIFO。当前content=learned，头Adam步数5000，头参数相对头训练前已改变。 |
| A/B耦合 | expected_delivery按coupled选择learned/fifo预测，实际coupled=True；固定状态夹具中执行选包与预计交付收益都能随模式变化。 |
| 通信/模型代码 | communication.py、model.py、memory.py、sensing.py与预实验执行版本逐字节相同。模型维度由配置64改128，代码相同不代表权重/容量相同。 |
| 观测动作接口 | 12维输入、WAIT候选、动态补齐与当前节点重排的针对性检查通过；重排后当前节点仍为0、候选索引有效，utility/visited随坐标同步排列。 |
| 模型实际训练 | 当前策略Adam步数均80632、策略参数相对共同起点已变化；头和上下文与common_start完全一致，符合既定冻结安排，没有被意外重置。 |
| A信号进入策略 | 对当前AB回放16条既有观测，仅将会合价值列置零，11/16条的动作分布发生变化，最大概率变化0.953。这是信号存在性检查，不说明A一定有益。 |
| 原图/数据边界 | 当前dataset_manifest与冻结SHA一致；配置读取dungeon_easy原文件，9899训练/100开发，测试保留。沿用已完成的原图解码、几何隔离检查和100图真实评估证据，本轮没有重新执行地图轨迹。 |

10个核心函数AST一致；在共同256地图尺度的固定局部状态下比较B0_fifo、B_learned、A_fifo、AB_learned、AB_uncoupled五配置，预实验与迁移版结果一致。夹具同时检查未共享队友地图/真值改变不影响局部特征、WAIT限制和原图节点重排。夹具不是实际地图策略成功率验证，不能替代已完成的100图评估。

核心代码：[会合/选包接口](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/finite_contact/environment.py:117)、[A收益输入](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/finite_contact/environment.py:248)、[上下文与评分](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/finite_contact/learning.py:140)、[真实worker调用](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/native_worker.py:44)。

## 确认未等价保留的训练机制

| 项目 | 自生成图预实验 | 当前原图训练 | 判断 |
|---|---|---|---|
| 成功轨迹模仿 | 保存每图成功示例，并参与策略损失；抽查seed101 AB有28550次模仿更新、987条成功示例 | 实际模仿更新0、成功示例0 | 整条训练链路未迁移，而不只是参数为0 |
| 移动预训练/冻结图上下文 | 100图×90=9000 SAC后提取 | 32回合、实际216 SAC后冻结 | 前缀减少约41.7倍，上下文此后不随移动策略更新 |
| 消息头训练数据 | 100张训练图，每图最多2048标签 | 16张训练图，每图最多512标签 | 虽然同为5000头更新，数据覆盖和图上下文不同 |
| Q学习率 | 1e-4 | 1e-5 | 采用原项目参数，不能视作不影响效果的工程提速 |
| Actor学习率 | 适应期1e-5 | 1e-5 | 此项未变，不能笼统说所有学习率都下降 |
| 有效batch/每图更新 | 16 / 20 | 128 / 8 | 采用原项目训练参数，样本利用与更新节奏改变 |
| Q梯度裁剪 | 10 | 2000 | 采用原driver设置，更新数值行为改变 |
| 目标网络 | 每步tau=0.01软更新 | 每64更新硬拷贝 | 采用原driver设置，学习动态改变 |
| Actor/Q更新次序 | Q先更新，Actor使用更新后的Q | Actor目标使用当前批更新前Q | 与原driver对齐，有别于预实验 |
| 温度防护 | log_alpha约束[-8,2] | 无该约束 | 当前AB log_alpha=-9.7883，已低于旧下限-8 |
| 目标熵 | 按实际合法动作数量计算 | 按固定neighbors=30计算 | 与原driver一致，掩码较多时语义不同 |
| 采样调度 | 串行一图后更新 | 24图固定权重波次及双方法预取 | 两组相同，但权重滞后、回放时序与旧协议不同 |

关键调用：[当前NativeLearner.update](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/native_learning.py:21)覆盖了继承来的Learner.update，所以旧模仿代码虽还在finite_contact/learning.py里，实际训练不会执行；[commit_episode](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/native_driver.py:110)也没有remember_success。仅将success_imitation_weight改成1无法接通这套机制。

[当前configuration](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/native_driver.py:77)明确设模仿权重0，[图上下文与标签前缀](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/native_driver.py:183)使用216次SAC后的策略、16图标签。原NATIVE_PLAN确实记录了采用原项目式SAC等有意改动，不能把这些差异全称作意外bug；但这些设置的预实验收益等价性没有被验证。若以后恢复成功示例链路，还需设计适配32GB机器的内存上限，不能盲目按9899图各保留长轨迹。

奖励公式、99%双机局部完成条件、有限字节收发、时间与路程成本等关键定义保留；变化主要在原图/层次图、训练器及其协议。没有发现奖励被误接成另一套公式的证据，但相同投递奖励在新地图/图路径/学习协议下可能带来不同的行为取舍，需要后续受控实验验证。

## 入口和训练流程的交付缺口

1. 仓库根[driver.py](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/driver.py:353)仍直接调用旧main，没有--finite-contact分流；根目录也没有native_driver.py。根入口运行的是原6维模型/原流程，不能当作当前A/B训练入口。它还在模块顶层ray.init，本轮只做AST检查，没有导入或执行它，避免误启动Ray。
2. 当前真实训练入口是实验快照[source/driver.py](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/source/driver.py:352)的--finite-contact分支；当前10060回合确实走了这条分支，不能把根入口没合并误说成此前训练跑错代码。
3. 冻结driver.py的NUM_META_AGENT可传入原图训练；LR、batch等在native_driver.configuration构造，改原parameter.py并不会自动改这些值。需要统一入口与参数所有权。
4. 当前CLI只完成easy新训练/恢复；configuration里有difficult配置，但没有正式阶段切换/续训的CLI路径。这是既定“easy验证后再接续”的待办项，不是本轮easy退步的原因。

## 值得跟进的信号及解释边界

会合距离衰减仍为exp(-distance/100)，收益距离成本也保持旧绝对系数；地图从256×256变为原生640×480等尺寸，通信范围仍65。当前AB回放512条抽样中，约47.1%的观测在合法候选（含WAIT）上有正会合价值；旧seed101 AB回放约97.3%。两者地图和状态不同，这只能提示特征尺度/分布变化，不能归因为某个系数错误，也不能单据比例决定调参。

已经完成的固定开发评估是基线56/100、AB59/100，失败罚时均值1410.639→1375.057秒；共同成功56图实际耗时161.855→224.274秒，AB快11图、慢45图。该结果证明当前方案尚未复现旧速度收益，不定位是哪项训练差异导致。需要保留旧正面结论及当前负面效率结果。

## 建议的处理顺序（尚未实施）

先统一driver入口/显式训练配置，并明确“原项目式纯SAC”与“预实验已验证学习配方”的差别。保留原数据、A/B定义、有限通信约束和公平对照，将成功示例训练链路、初始化/头训练前缀及其他学习规则作为明确变量；在隔离目录做有预算上限的小规模配对对照，确认恢复了哪些行为与收益后再开展长训练。一次恢复所有旧参数与只修改一个权重都不足以完成可靠诊断；本轮不擅自选择新配方、不重训、不进入复杂阶段。

## 本轮证据

- audit_summary.json：源码版本、10个核心函数、入口、训练语义、保护文件SHA与零新增训练/评估记账。
- functional_fixture_review.json：五种配置的固定状态逐项输出及原图重排检查。
- checkpoint_review.json：两组原图/旧seed101的模型、优化器、示例缓存、温度及回放信号。
- configuration_comparison.json：全部配置差异；code_reference_lines.json：执行快照函数行号。
- reference_development_results.json：只读引用已完成100图评估，没有重复评估。
- audit.py/audit.log：可复核排查脚本与检查输出。无环境step；16条已有观测共32次actor前向，不采样动作、不执行SAC。

源码、模型、优化器、回放、正式配置和数据在本轮排查前后哈希不变；STOP与监督PAUSE保留。此报告记录“针对性迁移检查通过及已确认不等价项”，不把检查通过表述为算法效果或端到端学习过程等价。
