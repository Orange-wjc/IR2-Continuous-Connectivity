# IR²通信感知探索项目：AI交接文档

> 最后更新：2026-09-15
> 用途：供新AI会话快速恢复项目状态。开始工作前先读本文，再按需阅读 [CONNECTIVITY_AWARE_PAPER_PLAN.md](./CONNECTIVITY_AWARE_PAPER_PLAN.md)。

## 1. 项目目标与边界

用户准备基于官方IR²撰写多机器人探索小论文。目标是：

> 在存在距离损耗和墙体信号衰减的未知室内环境中，提高机器人团队的通信连通率；允许短暂断连，但尽量缩短断连时间、加快重连，并学习动态探索者与中继者行为。

核心设定：

- 使用连续RSSI，由距离和墙体共同决定通信质量。
- 构建动态多机器人通信图，允许多跳连接。
- 优化连通率、断连时间和重连速度。
- 中继者不是固定角色，可以动态交接。
- 不研究低带宽，不要求全程零断连，不用硬动作屏蔽保证绝对连通。
- 暂不加入复杂时延、拥塞、在线墙体材料识别、无线电地图或真实机器人实验。

论文应使用“通信感知探索”“高连通率探索”“弹性连通约束”和“短时断连与快速重连”等表述。不能预先承诺固定95%连通率，也不能把“保持连接”“墙体影响RSSI”单独宣称为首次创新。

相关工作：

- IR²：https://arxiv.org/abs/2409.04730
- PRCL（仅作分布式学习参考）：https://arxiv.org/abs/2407.20203
- 早期持续连接探索：https://doi.org/10.1016/j.conengprac.2006.08.007
- DRL连通导航：https://proceedings.mlr.press/v100/lin20a.html
- PropEM-L：https://www.roboticsproceedings.org/rss18/p014.html
- DCM-RSSI：https://arxiv.org/abs/2410.05798

## 2. 仓库与重要文件

正式仓库：

```text
/home/robot/test/IR2-Continuous-Connectivity/IR2-Multi-Robot-RL-Exploration
```

- GitHub：`https://github.com/Orange-wjc/IR2-Continuous-Connectivity`
- 分支：`wall-aware-connectivity`
- `origin`：`git@github.com:Orange-wjc/IR2-Continuous-Connectivity.git`
- `upstream`：`https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration.git`
- `main` 保留官方基线；开发只在 `wall-aware-connectivity` 进行。
- 旧目录 `/home/robot/test/IR2-Multi-Robot-RL-Exploration` 仅供参考，不再修改。

文档：

- `docs/CONNECTIVITY_AWARE_PAPER_PLAN.md`：完整论文与实验方案。
- `docs/AI_HANDOFF.md`：本交接文档。
- `docs/IDEAL_CONNECTIVITY_MODEL_DEMO.html`：人工设计的概念展示，不是模型输出。
- `docs/hybrid1_full_demo/index.html`：用户确认的最终目标行为参考。使用官方
  `DungeonMaps/test/hybrid/1.png`，展示四机器人分散探索、移动接应、动态交接并
  完成地图探索。该演示由全图规则规划器生成，不是强化学习模型输出，也不能
  作为论文实验结果。
- `docs/build_hybrid1_full_demo.py`、`docs/hybrid1_full_template.html`：最终目标演示
  的生成脚本与页面模板；运行时还依赖 `docs/hybrid1_full_demo/opening_reference.json`。

`CONNECTIVITY_AWARE_PAPER_PLAN.md` 和 `IDEAL_CONNECTIVITY_MODEL_DEMO.html` 是用户原有
未跟踪文件；最终目标演示及生成文件也尚未跟踪。禁止误删、覆盖或随意提交。
任何操作前先运行 `git status --short`。

## 3. 已实现代码与提交

关键提交：

```text
9c6b785 Add continuous connectivity metrics
7635180 Visualize RSSI connectivity in GIFs
9aaf3bd Add connectivity-aware training features
017a3ed Speed up training data collection
b12adff Fix oversized action edge sets and resume config
8af6d7f Reduce single-GPU training synchronization overhead
85c050b Train three-agent policy on hybrid map split
7de27bd Increase hybrid training to 20 CPU actors
2302d18 Resume hybrid training with 22 CPU actors
9885440 Add balanced v3 connectivity fine-tuning
38b879f Add balanced v3.3 connectivity tuning
c6aef5e Implement v3.4 synchronous team rollouts and stay actions
87f5ec6 Run v3.4 evaluation with five parallel workers
45d8b02 Document v3.4 validation and improve evaluation tracking
```

`87f5ec6`是v3.4评估并行配置提交，`45d8b02`是交接文档与评估可追溯性提交。
2026-09-15已确认两者推送到`origin/wall-aware-connectivity`，本地HEAD与远端跟踪引用
均为`45d8b02`；AutoDL服务器也已成功快进更新到该提交。

实现摘要：

- `ss_realistic_model.py`
  - 输出连续RSSI、墙体穿越次数、障碍/自由空间距离和安全余量。
  - RSSI逐像素循环已向量化；200组随机地图/链路等价性检查通过。
- `env.py`
  - 维护时序连通率、断连时长、重连时间和最大连通分量等指标。
  - 加入通信软奖励，并缓存节点坐标索引。
- `multi_robot_worker.py`
  - 观测由官方6维扩展为11维，其中5维是通信特征。
  - 候选动作边限制在 `K_SIZE` 内，修复输入与mask形状不一致。
- `test_multi_robot_worker.py`
  - 测试时记录并可视化RSSI链路，同样限制候选边数量。
- `runner.py`
  - CPU Actor不再预留GPU份额，限制Actor内部线程数。
  - 删除Actor中未使用的Q网络副本及无效Q权重同步。
- `driver.py`
  - 单GPU不再使用无收益的 `DataParallel`。
  - 用独立CPU快照同步策略权重，避免全局模型频繁往返CPU/GPU。
  - 只在真正完成策略更新后刷新Actor权重。

这些性能修改不改变网络结构、奖励、地图分布、机器人数量、更新比例和checkpoint格式。完整episode冒烟测试曾按用户要求省略；语法、接口和权重快照等价性检查已完成。

## 4. 当前训练环境与参数

AutoDL正式训练环境：

- Conda：`mrobot`
- GPU：单卡NVIDIA GeForce RTX 4090
- CPU：25核
- 项目目录：`/root/autodl-tmp/IR2-Continuous-Connectivity`
- 启动：`python -u driver.py`

本地代码验证环境为 `BLMRE_py38`。AutoDL和本地环境名称不同，不要混淆。

`parameter.py` 关键配置：

```python
TRAIN_SET_NAME = "hybrid"
MAX_EPS_STEPS = 196
K_SIZE = 30
NODE_PADDING_SIZE = 360

USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 1
NUM_META_AGENT = 20
SUMMARY_WINDOW = 32
ARCHIVE_CHECKPOINT_EVERY = 320

FOLDER_NAME = 'wall_aware_hybrid3_balanced_v3_2'
MODEL_PATH = 'model/wall_aware_hybrid3_balanced_v3_2/checkpoint.pth'
LOAD_MODEL = False
LOAD_POLICY_ONLY = True
POLICY_PRETRAINED_PATH = 'model/wall_aware_hybrid3_balanced_v2_3520/checkpoint.pth'
CONTINUE_LOG_ALPHA = False
SAVE_TRAINING_GIFS = False

REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
USE_CONNECTIVITY_FEATURES = True
CONNECTIVITY_FEATURE_DIM = 5
INPUT_DIM = 11
INITIAL_LOG_ALPHA = -2.6
POLICY_LR = 3e-6
Q_LR = 1e-5
POLICY_ANCHOR_KL_WEIGHT = 0.5
POLICY_TRANSFER_CRITIC_WARMUP_UPDATES = 1024

CONNECTIVITY_TARGET_RATE = 0.90
CONNECTIVITY_BUDGET_WINDOW = 20
CONNECTIVITY_PRESSURE_RAMP = 0.10
RECONNECT_REWARD_WEIGHT = 0.0
NO_EXPLORATION_PROGRESS_GRACE_STEPS = 5
NO_EXPLORATION_PROGRESS_SATURATION_STEPS = 20
NO_EXPLORATION_PROGRESS_PENALTY_WEIGHT = 0.02
```

第一阶段训练固定使用3台机器人。原`DungeonMaps/test/hybrid`的400张地图由
`map_splits.py`按固定种子`20260907`重新划分为300张训练、50张验证、50张测试；
三个集合互不重叠，`hybrid/1.png`固定保留在测试集。文件仍位于原目录，逻辑划分由
`MAP_FILE_NAMES`控制。批量测试固定3台机器人、遍历50张保留测试地图一次，并默认
关闭GIF。v3.2只迁移balanced v2 episode 3520的策略，不覆盖任何旧实验。训练地图
每300轮按固定种子重新排列，避免每轮重复相同顺序。

v2通信奖励权重：弱信号0.05、断连0.10、断连持续时间0.20、重连0.10；前3步为
断连持续时间宽限期。另加入`5.0 × 平均个体覆盖率增量`的团队探索进度奖励，初始
覆盖率在环境启动后设为基线，避免第一步产生虚假奖励。

当前v3奖励与策略迁移配置见第9节；v2权重仅作为迁移来源和历史基线。

环境仿真主要消耗CPU，网络更新使用GPU。v2使用20个CPU Actor，在吞吐和异步策略
滞后之间折中；GPU利用率曾约65%，显存约19.7/24.6 GiB。不要继续增加Actor或
`BATCH_SIZE`，也不要设置`USE_GPU=True`让采样Actor争抢单块GPU。

## 5. checkpoint与续训

已找回并验证的训练权重：

```text
相对路径：model/wall_aware_stage1/checkpoint.pth
本地路径：/home/robot/test/IR2-Continuous-Connectivity/IR2-Multi-Robot-RL-Exploration/model/wall_aware_stage1/checkpoint.pth
大小：52,199,630 bytes（约50 MB）
episode：928
input_dim：11
connectivity_feature_dim：5
SHA-256：f861e42f5a29461c586e51e782db8eed5d125d11edba16ef06566c7f5d331934
```

checkpoint包含策略网络、两个Q网络、对应优化器、学习率调度器、`log_alpha` 和episode，结构完整，可以直接续训。

已保存的hybrid3 stage1失败基线：

```text
路径：model/wall_aware_hybrid3_stage1/checkpoint.pth
大小：52,199,630 bytes
episode：672
input_dim：11
优化器更新：4960
SHA-256：77638cb934c47b62338ba5cc3e599ff3523ee00adb9d9124872b5a1fbbc58a9e
```

该权重数值完整，但策略过度偏向连通，不能作为最终模型；保留用于失败分析和消融。

已下载并验证的balanced v2当前权重：

```text
路径：model/wall_aware_hybrid3_balanced_v2/checkpoint.pth
大小：52,199,630 bytes
episode：864
input_dim：11
connectivity_feature_dim：5
use_connectivity_features：True
优化器更新：6720
log_alpha：-2.59563732147（alpha约0.07460）
SHA-256：1218764e84a50f0872b142eda1c2dd29cc970508926feea0ea211254bc23d466
```

策略网络、两个Q网络及优化器状态均结构完整，未发现NaN或Inf。该权重是当前第一阶段
的首选评估对象；继续训练前应另行备份，避免覆盖这个可复现节点。

若专门续训历史`wall_aware_stage1`实验，服务器必须将旧权重放到：

```text
/root/autodl-tmp/IR2-Continuous-Connectivity/model/wall_aware_stage1/checkpoint.pth
```

该历史实验续训时才保持`LOAD_MODEL=True`和`CONTINUE_LOG_ALPHA=True`。v2首次启动
必须使用`LOAD_MODEL=False`。旧实验续训成功时会看到：

```text
Loading Model...
curr_episode set to: 928
```

Replay Buffer仅在内存中，不写入checkpoint。每次进程重启后都要重新采集 `MINIMUM_BUFFER_SIZE=2000` 条transition才恢复梯度更新；这是2000条transition，不是2000个episode，也不表示网络权重重置。

checkpoint每32个episode覆盖保存一次，日志显示 `Saving model` / `Saved model`。服务器迁移前必须单独下载 `.pth`；`train/wall_aware_stage1` 只有TensorBoard日志，不能恢复模型。

兼容约束：

- 官方checkpoint是6维输入，测试时保持 `USE_CONNECTIVITY_FEATURES=False`。
- 新模型是11维输入，训练和测试都必须启用通信特征。
- 官方6维checkpoint不能直接加载进11维网络。

## 6. 已知问题与处理原则

### Ray CPU警告

Docker CPU检测警告本身不会阻止训练。15个Actor占用16核时资源接近饱和；若出现持续的 `resource request cannot be scheduled`，先减少Actor，不要给Actor分配GPU。

### 候选边形状不一致

旧错误：

```text
Not (edge_inputs.shape = one.shape == edge_padding_mask.shape). Skipping eps.
```

原因是候选边超过 `K_SIZE=30`，已由 `b12adff` 修复。修复后可继续使用原checkpoint，无需重训。

### 节点数超过padding

偶发日志：

```text
node_coords.shape[0] >= self.node_padding_size (372 >= 360). Skipping eps.
```

这是保护性跳过单个回合，不会停止训练。若占比低于约1%至2%，暂不处理；若超过5%，再统计触发条件。不要随意增大 `NODE_PADDING_SIZE`，图注意力开销近似按节点数平方增长，当前显存余量有限。

### checkpoint找不到

```text
FileNotFoundError: model/wall_aware_stage1/checkpoint.pth
```

表示权重未放到相对于当前工作目录的正确位置，不是模型损坏。搜索命令：

```bash
find /root/autodl-tmp -type f -name "checkpoint.pth" -ls
```

### Ray Actor找不到模块

若出现 `ModuleNotFoundError: No module named 'model'`，确认从项目根目录启动，并在必要时将项目根目录加入 `PYTHONPATH`。Ray Actor会重新导入配置，主进程中的临时参数覆盖不一定会传递给Actor。

### v3首次更新时CUDA显存不足

2026-09-09服务器首次启动v3时，episode 3520的policy已正确加载，critic、
优化器、alpha和episode计数也已按设计重置，但在第一次梯度更新的第二个Q网络
前向计算处出现CUDA OOM。`ray stop --force`后无Ray残留进程，`nvidia-smi`显示
RTX 4090的24564 MiB显存基本全部空闲，因此这是单次SAC更新同时保留多个
注意力计算图导致的峰值显存问题，不是其他进程占用，也不是checkpoint损坏。

`driver.py`已保持`BATCH_SIZE=128`并做以下等价的显存优化：

- critic预热期不更新policy，因此policy前向改为`torch.no_grad()`。
- policy损失所需的计算图在critic训练前释放。
- Q1前向、反向、更新并释放后，再计算Q2，避免两个critic计算图同时驻留。

修复不改变奖励、网络结构、更新次数、checkpoint格式或目标Q值的计算时点。
本地CPU最小更新测试已确认：预热期只更新Q1/Q2，预热后policy、alpha、Q1/Q2
都能正常更新。如果服务器应用此修复后仍在第一次更新OOM，再将
`BATCH_SIZE`降为96并作为独立配置变更；不要在未验证时同时改动更多参数。

## 7. TensorBoard与当前曲线

已验证兼容组合：TensorBoard 2.14.0 + protobuf 4.25.3。protobuf 5.29.6会导致：

```text
TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'
```

启动命令：

```bash
tensorboard --logdir ./train/wall_aware_hybrid3_balanced_v2 --host 0.0.0.0 --port 6006
```

分析时只勾选最新run，并把Smoothing设为约0.2至0.3，避免多次重启和过度平滑干扰判断。

截至约episode 850的早期趋势：

- Agents Connected：约0.72 → 0.99
- Connectivity Rate：约0.42 → 0.99
- Communication Reward：约-0.30 → 接近0
- Explored Rate：约episode 450达到0.96，随后降至约0.74
- Success Rate：峰值约0.70，随后降至约0.40
- Reward：峰值约0.20，随后降至约0.08
- Travel Distance：后期约升至5400

2026-09-06读取TensorBoard最新run
`Sep06_08-38-57_autodl-container-vv2u1fwe7h-67b4e717`，记录范围约episode 985至1561。
应将Smoothing设为0.2至0.3，只选最新run；0.999会掩盖真实波动。部分原始记录点：

| episode | Explored Rate | Success Rate | Connectivity Rate | Travel Distance |
|---:|---:|---:|---:|---:|
| 1497 | 0.7517 | 0.4235 | 0.9208 | 4734.7 |
| 1529 | 0.7419 | 0.5480 | 0.8568 | 4354.9 |
| 1561 | 0.5811 | 0.2496 | 0.8823 | 6131.7 |

这些是训练日志窗口统计，不是独立测试结果；不能根据最后一个点下结论。约1000至
1500轮的总体判断是：连通性较高，但探索率和成功率波动且较早期下降，存在策略
偏向保守连通的风险。Loss未显示明确数值发散，但Loss下降不等于任务效果改善。
`Perf/Reward` 是训练batch的平均单步奖励，`Travel Distance` 是最大单机器人累计路程。
下一步应先做固定地图独立评估，再决定继续原样训练还是新开实验调奖励。

2026-09-07的hybrid3 stage1曲线给出了更明确的失败证据：约episode 320时探索率
达到0.94、成功率约0.47，随后连通率趋近1.0，而探索率跌至约0.56、成功率跌至0，
路程和总奖励同时下降。episode 416权重在保留的`hybrid/1.png`上实现100%连通、
零断连，但196步只达到40.11%平均个体覆盖率；同条件官方模型达到99.99%覆盖，
但连通率仅37.93%。结论是奖励诱导了“抱团少移动”的局部最优，而非数值发散。

2026-09-07的`wall_aware_hybrid3_balanced_v2`曲线记录到约episode 850。探索率在约
episode 250升至0.90以上，此后稳定在约0.97至0.99；成功率从0升至后期约0.90至
0.97。连通率在训练中段两次明显下降后恢复，末期约0.75；Agents Connected末期约
0.90，通信奖励由低谷约-0.11恢复到约-0.02。探索进度奖励持续上升，末期约0.06，
总奖励末期约0.7。训练地图在episode 300和600重排后没有再次出现v1的探索崩塌，
说明奖励重平衡已明显改善“只顾连通、不去探索”的问题。

Loss仍需谨慎解释：Value升至30以上，Q梯度范数后期约150且曾超过200，策略梯度
范数约3.5并有尖峰；但权重有限且没有NaN/Inf，目前没有明确数值发散证据。Value
尺度上升与成功终止奖励和高成功率一致，不能仅凭Loss绝对值判定模型失败。当前不应
继续修改奖励，应先冻结episode 864并完成独立测试。

当前代码的指标口径必须按实现解释：

- `Explored Rate` 是各机器人自身belief覆盖率的平均值，不是融合后的全局地图覆盖率。
- `Success Rate` 只有在每台机器人自己的belief地图都达到99%覆盖时才算成功，因此比
  “团队合并后覆盖完成”更严格。
- 断连机器人相对唯一最大连通分量判定；若多个连通分量并列最大，代码会把它们都
  视为不属于唯一主分量。比较实验时必须沿用相同实现，或先修订并明确新口径。

## 8. 实验设计

建议对比：

1. 原始IR²。
2. 仅按距离判断通信的IR²。
3. 墙体RSSI、无断连时长设计的IR²。
4. 完整方法：连续RSSI + 断连时长约束 + 动态中继。

主要指标：覆盖率、完成步数、路径长度、成功率、回合连通率、断连次数、平均/最长断连时长、平均重连时间、最大连通分量比例、最弱通信树边RSSI和关键中继占用时间。

最终目标是在探索效率没有明显下降的前提下，提高连通率并缩短断连与重连时间。奖励或任务定义发生变化时必须使用新实验名称和目录，保留当前checkpoint作为可复现实验基线。

## 9. 当前状态与下一步

当前状态：

- 分支为 `wall-aware-connectivity`；balanced v2代码已提交并推送至`cf78924`。
- RTX 4090/25核/90GB服务器的v2配置使用20个并行CPU仿真和单GPU学习。
- 候选边形状问题已经修复。
- episode 928的11维checkpoint已找回并验证。
- 历史`wall_aware_stage1`服务器TensorBoard曾观察到约episode 1561，本地保留的
  对应已验证checkpoint为episode 928；两者不能视为同一训练节点。
- hybrid3 stage1已判定为过强通信约束导致的策略退化；episode 672权重已下载并验证，
  必须保留，v2不得续训或覆盖该权重。
- balanced v2最新权重已下载到
  `model/wall_aware_hybrid3_balanced_v2/checkpoint.pth`并验证为episode 864。曲线显示
  探索率长期约0.97至0.99、后期成功率约0.90至0.97，未重现v1的探索崩塌。
- episode 864在保留地图`hybrid/1.png`上的一次贪心实测达到99.795%平均个体覆盖率，
  70个移动步、路径长度3162.02、连通率95.714%、最大连通分量比例98.571%，发生
  2次短暂断连，平均1.5步、最长2步，并成功完成探索。这说明目标行为已在该样例上
  出现，但单张地图不能替代50张保留地图的统计评估。
- balanced v2继续训练至episode 3520，权重保存在
  `model/wall_aware_hybrid3_balanced_v2_3520/checkpoint.pth`。该权重结构完整且无NaN/Inf，
  SHA-256为`ce4ca5e759109d26bdfc2d4097221c399398df49c2572bde7c252f996af03e16`。
  50张保留测试地图一次贪心评估达到50/50成功、平均探索率99.715%、平均72.04步，
  但平均全连通率仅69.88%；23/50张地图最长断连超过10步，说明探索泛化很好但通信
  长尾未达到目标。结果位于
  `wall_aware_hybrid3_balanced_v2_3520_inference/test_results/log/data_2026-09-08_175054.csv`。
- 用户已确认以 `hybrid1_full_demo` 展示的行为作为训练目标：机器人在不同区域
  分散探索，接应位置随探索进度移动，必要时交接中继任务，最终达到地图探索
  完成条件，同时避免长时间断连。演示中的全图信息与硬连通动作检查仅用于构造
  目标参考，不能直接加入论文方法或冒充模型能力。
- 用户决定继续使用原作者提供的训练/测试地图，不再生成新的训练地图。
- 当前第一阶段已收窄为固定3机器人、仅使用原作者hybrid地图分布。400张hybrid地图
  固定划分为300/50/50；训练与测试隔离，`hybrid/1.png`只用于最终测试。完成这一
  阶段并评估效果后，再决定是否加入4机器人训练。
- 最终目标演示使用项目`hybrid/1.png`、原`sensor.py`和固定无噪声RSSI参数；其
  规则轨迹保持简化通信图全程连通，发现107455/107500个自由像素，覆盖率
  99.958%，剩余45像素。共有13885个插值脚本帧；这些数字只验证演示内部一致性。
- 其他由AI生成的临时地图与演示已删除，只保留最终`hybrid1_full_demo`。用户原有
  `IDEAL_CONNECTIVITY_MODEL_DEMO.html`仍保留。
- 用户已将当前工程验收口径简化为两项唯一硬指标：固定保留地图评估的
  平均探索率不低于99%，平均全连通率不低于90%。成功率、完成步数、路径长度、
  断连次数和断连/重连时长继续记录，但当前只作诊断参考，不作为模型达标门槛。
  这两项是模型选择目标，不能在论文中预先宣称已经达到。
- balanced v3已配置为从episode 3520只迁移`policy_model`，不加载旧Q网络、目标Q网络、
  优化器、温度或episode计数。新实验目录是`wall_aware_hybrid3_balanced_v3`，初始
  `log_alpha=-2.6`；Replay Buffer仍需收集2000条transition。随机critic先单独预热
  1024次梯度更新，之后才启用policy和alpha更新，避免随机Q值立即破坏已学探索策略。
- v3保持11维输入不变，但第4个通信特征从二值relay标志改为连续relay/recovery分数；
  它会在断连超过3步后随紧迫度增强，10步时达到最大软恢复强度，不进行硬动作接管。
  因为特征语义和奖励均已改变，v2的完整Q状态不能用于v3续训。
- v3通信奖励使用包含阈值以下边的全队RSSI最大生成树瓶颈，不再只看最大连通分量；
  同时使用瞬时最大连通分量缺口、最大机器人断连时长、断连发生事件和较小的重连奖励。
  权重为弱信号0.10、分量缺口0.25、断连时长0.25、新断连0.10、重连0.03；断连时长
  在3步宽限后增长，到30步才饱和。团队探索进度权重仍为5.0，完成奖励仍为40。
- TensorBoard新增断连次数、平均/最长断连时长、平均重连时间、最大连通分量比例、
  平均分量数和全队瓶颈RSSI。每320轮额外归档`checkpoint_<episode>.pth`，日常
  `checkpoint.pth`仍每32轮更新。
- `test_parameter.py`已指向未来的v3.2权重，50图测试为每个地图使用
  `TEST_RANDOM_SEED + run_index * NUM_TEST + episode_number`固定Python、NumPy和PyTorch
  随机种子，CSV同时记录run、种子及`team_bottleneck_rssi`，以便不同checkpoint使用
  相同RSSI条件复现实验。`NUM_RUN>1`时每轮使用不同但可复现的信道种子。
- v3本地验证已完成：五个相关Python文件AST解析通过，episode 3520策略严格加载通过，
  奖励随5/10/30步断连持续增强，全队RSSI最大生成树与软恢复特征数值检查通过；另有
  一个不保存GIF的完整CPU回合成功完成，15路episode buffer长度一致且新增指标齐全。
- v3在服务器上已确认正确读取v2 episode 3520策略，但洁净的24 GB GPU在
  首次更新时出现峰值显存不足。`driver.py`已改为预热时禁用policy梯度、提前释放
  policy计算图并串行更新Q1/Q2；本地最小训练步已覆盖预热与正常更新两条路径。
- v3训练至约episode 736时已明确退化：训练窗口中全连通率接近99%、断连接近0，
  但探索率降至约69%、成功率降至约6%，再次形成“抱团少探索”的局部最优。
  该运行已判定不宜继续；应保留v3的`checkpoint_640.pth`和最后权重供失败分析。
- v3.1使用明确的90%短窗口通信预算：最近20步全连通率不低于90%时，弱信号、
  分量缺口和新断连的瞬时压力为0；低于90%后在10个百分点内线性增强到全量。
  超过3步的连续断连惩罚始终有效，以防止用一次长断连消耗预算；重连奖励改为0，
  避免主动制造断连并重连的奖励钻空子。
- v3.1的policy学习率从`1e-5`降为`3e-6`，Q网络仍为`1e-5`；奖励基础权重、
  探索进度权重5.0、完成奖励40、预热1024次和11维网络结构均不变。新实验目录为
  `wall_aware_hybrid3_balanced_v3_1`，仍从v2 episode 3520只迁移policy。TensorBoard新增
  `Recent Connectivity Rate`和`Communication Pressure`用于检查预算门控。
- v3.1继续训练至约episode 1400后也已判定失败：全连通率约98%、最近连通率
  接近100%、通信压力接近0，但探索率降至约77%、成功率降至约10%。这证明
  90%预算门控本身正常，但只关闭通信惩罚不足以防止长期微调遗忘v2的探索策略。
  v3.1的归档权重仅作失败分析，不作最终模型。
- v3.2恢复了v2权重训练时使用的二值relay特征语义，不再将该列混合为连续
  relay/recovery分数。训练主策略时同时加载一个冻结的v2 episode 3520参考策略，
  用权重0.5的动作分布KL约束防止新策略快速偏离已验证的探索行为，但不冻结主策略。
- v3.2保留v3.1的90%通信预算和长断连惩罚，另加入轻量探索停滞惩罚：连续5步
  无新探索不处罚，之后在20步内线性增长，每步最多0.02。这一设计直接消除
  “抱团不动且收益为0”的安全局部最优，同时保留穿越已知区域的宽限。新实验目录为
  `wall_aware_hybrid3_balanced_v3_2`，仍从v2 episode 3520重新迁移，不继承v3/v3.1权重。
- v3.2本地验证已确认：主策略与冻结v2参考策略的初始输出逐元素一致，
  KL只对主策略产生梯度，二值relay特征检查通过，停滞惩罚的宽限/饱和/重置均正确。
  一个不保存GIF的完整CPU回合在93个团队步后成功，探索率99.953%、全连通率88.17%、
  最长断连4步；该单回合只是接口冒烟测试，不是独立评估结果。
- v3.2训练至约episode 1300后，探索率稳定在约99.6%，说明二值relay特征、冻结v2
  参考策略KL约束和探索停滞惩罚成功保住了v2的探索能力；但平均全连通率仅约74%，
  最近窗口连通率约65%，仍未达到90%的目标。此时通信压力约0.73、通信奖励约-0.06，
  与探索进度奖励约+0.068处于相近量级，说明低于目标时的即时通信惩罚仍然偏弱。
  v3.2已经判定为“探索达标、通信不达标”，继续原参数长训不太可能稳定达到目标；
  应停止训练并保留现有权重用于分析，不应从v3.2的Q网络和优化器状态继续训练。
- balanced v3.3代码已提交并推送至`38b879f`。新实验目录为
  `wall_aware_hybrid3_balanced_v3_3`；仍从v2 episode 3520只迁移policy并使用权重0.5的
  冻结v2策略KL约束。相对v3.2只将低于90%预算时的即时通信权重提高约2倍：弱信号
  0.20、分量缺口0.50、新断连0.20；断连时长权重保持0.25，其余探索停滞、预算窗口、
  学习率和critic预热设置不变。`driver.py`中的奖励版本标识为
  `balanced_v3_3_anchored_target_90`。
- v3.3训练至约episode 1050时一度出现较好窗口：探索率约99.6%、全连通率约85%至86%、
  最近连通率约78%、最大连通分量约94%至95%，最长断连约6步；但该改善没有稳定保持。
  至约episode 1350，探索率仍约99.7%、成功率接近99%至100%，全连通率却回落至约
  71%至72%，最近连通率约59%，Agents Connected和最大连通分量比例约88%至89%，
  最长断连约12步、平均断连约3.4步、瓶颈RSSI约-76 dBm。Q Value Grad Norm长期升至
  约500至650，Q Loss约14且未见NaN/Inf，反映critic噪声/不稳定性而非简单数值崩溃。
  因此v3.3暂判定为“探索达标、通信短暂改善但无法稳定达标”，不应继续盲目长训。
- balanced v3.4开发已经完成，核心实现提交为`c6aef5e`，验证并行配置提交为`87f5ec6`。
  独立实验目录为`wall_aware_hybrid3_balanced_v3_4`。本版训练与推理共用同步团队rollout：
  所有机器人先基于同一时刻观测选择动作，全部移动和感知后再统一交换信息、计算团队奖励
  并构造下一观测；同时修复独立belief、消息快照、重叠前沿奖励顺序偏差、196步终止语义，
  并将stay作为显式合法动作。完整设计与兼容边界见`docs/V3_4.md`。
- v3.4保留11维actor输入、二值relay特征、v3.3通信奖励、90%短窗口预算、探索停滞惩罚、
  v2 episode 3520策略初始化和移动动作条件KL。它没有实现集中式critic、GRU、持续角色头
  或专家行为克隆，因此只完成了此前建议中的同步转移与stay动作修复，不等于完整v4方案。
- v3.4已训练并保存`checkpoint_320.pth`、`checkpoint_640.pth`、`checkpoint_960.pth`、
  `checkpoint_1280.pth`和`checkpoint_1600.pth`等归档。首轮验证结果已下载到
  `wall_aware_hybrid3_balanced_v3_4_validation/test_results/log/`。旧CSV没有记录模型路径，
  下列对应关系最初来自当时的测试顺序：

  | 推定checkpoint | CSV时间 | 平均探索率 | 平均全连通率 | 成功数 | 判断 |
  |---:|---:|---:|---:|---:|---|
  | 1600 | 15:30 | 99.242% | 87.527% | 48/50 | 探索达标，通信未达标 |
  | 1280 | 15:52 | 98.793% | 84.174% | 47/50 | 两项未同时达标 |
  | 960 | 16:06 | 不完整 | 不完整 | — | 无表头且仅37条，不能使用 |
  | 640 | 16:20 | 99.637% | 86.274% | 50/50 | 探索达标，通信未达标 |
  | 320 | 16:32 | 99.554% | 91.272% | 50/50 | 单轮验证两项达标 |

  15:09的CSV只有9张地图，是中断运行，不能作为完整验证结果。2026-09-15使用新增模型
  身份字段重新运行后，320的`run_0`与旧16:32 CSV在所有共同字段逐行完全一致，因此旧
  16:32结果对应`checkpoint_320.pth`的映射已得到确认。
- 提交`45d8b02`已改进v3.4评估入口：`test_parameter.py`支持通过
  `IR2_EVAL_NUM_RUN`设置正整数run数量，默认仍为1；`test_driver.py`在控制台打印实际
  模型路径和checkpoint episode，并在每条CSV记录中写入`model_path`、`checkpoint_file`
  和`checkpoint_episode`。2026-09-15已在AutoDL服务器用该入口完成320、640、960、1280、
  1600五个checkpoint的三随机种子验证。15个正式CSV各含50张相同验证地图，五个候选的
  `run/eps/map_file/test_seed`键完全一致，共750个成功执行且可配对的回合；另有一个
  `12:05:42 run_0`文件仅1条记录，已排除。
- 五个checkpoint的三run验证汇总如下。`run连通率`按run 0/1/2排列，`run间标准差`是三个
  run均值的样本标准差；探索与连通均值按每个checkpoint的150个回合等权统计：

  | checkpoint | 平均探索率 | 平均全连通率 | run连通率 | run间标准差 | 成功数 | 判断 |
  |---:|---:|---:|---:|---:|---:|---|
  | 320 | 99.597% | 87.452% | 91.272% / 87.227% / 83.858% | 3.71个百分点 | 149/150 | 探索达标，通信不达标 |
  | 640 | 99.428% | 85.848% | 86.274% / 83.375% / 87.893% | 2.29个百分点 | 146/150 | 探索达标，通信不达标 |
  | 960 | 99.329% | 88.294% | 87.886% / 84.670% / 92.325% | 3.84个百分点 | 147/150 | 探索达标，通信不达标 |
  | 1280 | 99.180% | 86.597% | 84.174% / 86.658% / 88.958% | 2.39个百分点 | 143/150 | 探索达标，通信不达标 |
  | 1600 | 99.364% | 87.857% | 87.527% / 87.727% / 88.317% | 0.41个百分点 | 144/150 | 探索达标，通信不达标 |

  960的总体连通率最高，但仍低于90%门槛1.706个百分点，且只有run 2单轮达标；1600的
  跨run波动最小，但三个run均低于90%。配对回合上960相对320只提高0.841个百分点，
  近似95%置信区间跨过0，不能据此宣称960稳定优于320。五个checkpoint没有一个同时满足
  两项模型选择门槛，320也不再是可冻结的达标早停模型。
- 320的150回合中，100/150回合全连通率不低于90%，66/150回合全程连通，42/150回合
  最长断连超过10步；平均最长断连8.83步，最坏为86步。50张地图中只有27张的三run
  平均连通率不低于90%，只有20张在三个run中均达到90%。其总体探索率仍为99.597%，
  说明主要失败仍是随机信道下的通信长尾，而不是探索能力再次崩塌。
- `107.png`、`386.png`、`323.png`、`83.png`、`55.png`和`229.png`在五个checkpoint上
  的三run平均连通率都低于90%，是跨训练阶段反复出现的共同困难地图。部分地图存在极强
  信道敏感性，例如320在`113.png`上的连通率跨种子为17.78%至100%，在`375.png`上为
  30.16%至100%。这说明单次固定种子结果会高估稳定性，下一版必须重点处理少数地图的
  长断连恢复和跨信道泛化。
- v3.4的准确阶段结论更新为：同步团队转移与stay修复、训练以及五候选三种子验证均已完成；
  探索指标整体保持在99%以上，但所有候选的平均全连通率都低于90%，因此v3.4多种子验证
  未达标。最终保留测试集尚未运行，也不应在当前候选上运行；所有现有新CSV的`split`均为
  `validation`，不得表述为最终测试结果或论文最终结论。
- 对v3.4之前实现的代码级复盘表明，瓶颈不只是奖励权重：
  `multi_robot_worker.py::run_episode`按机器人顺序执行动作，每个机器人在自己动作后立即
  保存next observation，但团队奖励要等全部机器人动作完成后统一计算并复制给所有机器人；
  早行动机器人的transition因而包含了next state中看不到的后续机器人行为，破坏奖励与
  转移对齐。与此同时，奖励依赖20步`recent_connectivity_rate`和断连历史，而actor的
  11维观测没有精确包含这些历史量，形成状态混叠/非马尔可夫问题；同一团队奖励复制给
  全体机器人也缺少“谁破坏或恢复了连通”的信用分配。v3.4已经修复联合动作、团队奖励与
  next state的同步对齐，但20步历史的状态混叠和逐机器人信用分配仍未解决。
- v3.4之前的动作空间还通过`model.py`中的`current_mask[:, :, 0] = 1`屏蔽当前位置，
  机器人无法主动停留或驻守；v3.4已经加入显式stay，但独立的一步局部决策仍没有持续的
  探索者/中继者/恢复者角色。
  v2 KL锚点虽然保住探索，却同时锚定了平均连通率仅69.88%的旧策略。v3/v3.1与v3.2/v3.3
  分别落入“高连通低探索”和“高探索低连通”两个端点，说明继续只调标量奖励很难解决
  当前的状态、联合转移、信用分配和时序协调缺陷。
- 已阅读并对照的方法包括：全局连通约束与行为克隆
  （https://arxiv.org/html/2109.08536）、PRCL的训练期特权critic
  （https://arxiv.org/html/2407.20203）、MAAC集中式注意力critic
  （https://arxiv.org/html/1810.02912）、COMA反事实信用分配
  （https://arxiv.org/html/1705.08926）、MACPO/MAPPO-Lagrangian约束多智能体学习
  （https://arxiv.org/html/2110.02793）、MACAAC注意力critic与拉格朗日约束
  （https://ifaamas.org/Proceedings/aamas2021/pdfs/p1616.pdf）以及ROMA持续角色表示
  （https://arxiv.org/html/2003.08039）。其中与本项目最接近的ICRA 2022工作也报告了仅靠
  连通约束会产生原地振荡、难以导航，需用行为克隆使优化可行，这与当前失败模式一致。
- 下一版建议采用`Connectivity-Constrained Privileged MAAC`思路：执行时仍使用分布式
  IR² actor，训练时引入可见全队观测、联合动作、真实RSSI图、连通历史和全局探索状态的
  集中式注意力奖励critic与成本critic；把`c_conn = 1[通信图断开]`及其期望不超过0.10
  作为与90%全连通率直接对应的约束，用自适应拉格朗日乘子代替继续手调固定惩罚，并用
  COMA式反事实优势进行逐机器人信用分配。该方案若在现有off-policy SAC上实现，应明确
  表述为工程适配，不能直接声称继承MACPO的理论保证。
- 原v4建议还包括同步数据与动作建模；其中统一team next state、同步team transition和
  解除stay动作屏蔽已由v3.4完成。尚未实现的部分包括给actor增加局部历史/GRU、RSSI趋势、
  队友信息年龄和角色持续时间，以及简单的`explore/relay/recover`角色头并约每5个团队步
  更新一次。
  用覆盖300张训练地图的特权专家联合轨迹做通信感知行为克隆，替代把低连通v2 KL作为
  唯一锚点；特权全图/RSSI只允许用于训练期teacher和critic，测试时actor仍只能使用局部信息。
  v2可用于初始化原图编码器/指针actor，但新增历史与角色模块需随机初始化，critic、优化器和
  Replay Buffer均不可迁移；建议独立实验名`wall_aware_hybrid3_privileged_cmaac_v1`。
- 2026-09-16已完成12个困难样例的本地CPU轨迹诊断，覆盖`386/113/191/107/83/55.png`
  及表现好坏成对的v3.4 checkpoint。除`107.png checkpoint_1280 run_1`因CPU/GPU动作
  浮点差异只作定性参考外，其余11个回合与服务器关键指标一致。轨迹显示主要问题不是
  探索失败，而是断连后继续探索、缺少明确返回方向，以及通信边界附近反复进出；stay几乎
  没有被使用。750个验证回合中270个低于90%连通率，其中218个最长断连超过10步；连通率
  与最长断连时长相关系数约-0.865，而与stay比例和探索率几乎无关。
- 基于该诊断，已开始实现较小且可归因的`v3.5-A local recovery`，而不是立即一次性引入
  集中式critic、GRU、角色头和行为克隆。独立实验目录为
  `wall_aware_hybrid3_recovery_v3_5_a`，以三run平均连通率最高的v3.4
  `checkpoint_960.pth`只迁移actor；critic、目标critic、优化器、温度、Replay Buffer和
  episode计数全部重新初始化，policy学习率降为`1e-6`。
- v3.5-A保留v3.4前11维输入的顺序并新增5个仅依赖本地可观测信息的恢复特征：30步长断连
  紧迫度、最强可通信链路RSSI趋势、队友位置信息年龄、指向上次全队可达位置的候选恢复分数、
  以及20步本地可达分量比例历史。输入维度由11增至16。旧actor的第一层11列原样复制，
  新5列置零；冻结参考策略同样扩展，因此训练开始时策略输出与v3.4-960逐元素一致。没有把
  全队真实连通状态等训练期特权信息输入actor。
- v3.5-A代码已完成本地静态和接口验证：扩展前后策略输出最大误差为0，10维新特征与旧5维
  兼容模式均为有限值，一个完整同步团队步可成功产生16维下一观测。当前代码尚未提交，
  v3.5-A尚未训练，也没有新的验证结果；这些检查只证明实现可运行，不代表效果达标。

下一步：

1. 不对v3.4任何现有checkpoint运行最终保留测试集，也不将320或960宣布为冻结模型。
   五候选三种子验证已完成且均未达到90%平均全连通率；最终测试集继续保持未使用状态，
   直到新版本在验证集同时达到平均探索率不低于99%、平均全连通率不低于90%。
2. 停止v3.4原参数盲目长训和继续微调标量通信惩罚。保留且不要覆盖320、640、960、1280、
   1600以及现有最终权重，并保留2026-09-15的15个完整CSV作为可复现失败基线；只有1条
   记录的`12:05:42 run_0`以及历史中断CSV不得用于汇总。
3. 先审查并提交当前v3.5-A代码，再在服务器从v3.4 `checkpoint_960.pth`启动一个全新的训练，
   不续接v3.4的episode或优化器。按320 episode归档权重，并周期性运行验证，不能只看训练
   loss或训练窗口判断收敛。
4. v3.5-A训练早期先在相同50张验证地图的固定run 0上筛查320/640/960等归档；只有候选同时
   保持99%探索且连通率有明确改善，才对它运行run 0/1/2完整三种子验证。困难地图与坏种子
   用于解释变化，不单独作为模型选择标准，更不能接触最终测试集。
5. 如果v3.5-A三run总体达到探索率不低于99%、全连通率不低于90%，冻结最佳验证checkpoint
   后才运行一次最终测试。如果局部恢复特征仍不达标，再进入第二阶段：增加独立通信成本critic、
   拉格朗日90%约束和逐机器人信用分配；之后才考虑GRU、专家BC与持续角色，避免一次引入
   太多无法归因的改动。
6. 新候选仍使用相同50张验证地图、相同run/地图种子和贪心动作做至少三run评估，报告每个
   run及150回合总体均值、run间标准差、长断连指标和配对差异。只有总体两项硬指标同时达标
   才能冻结模型；冻结后对50张最终保留测试地图只运行一次，不能再依据测试结果选模或调参。
7. 保存episode 928、hybrid3 stage1 episode 672、balanced v2 episode 864/3520以及
   v3/v3.1/v3.2/v3.3/v3.4归档权重，作为历史基线、失败模式和论文消融对照；3机器人阶段
   获得验证达标模型并完成最终测试后，再决定是否扩展到4机器人。

## 10. 新会话检查清单

1. 确认位于正式仓库和 `wall-aware-connectivity` 分支。
2. 阅读本文和论文方案文档。
3. 运行 `git status --short`，保护用户未跟踪文件。
4. 本地使用 `BLMRE_py38`；AutoDL使用 `mrobot`。
5. 确认GPU可见、checkpoint存在，并核对续训episode。
6. 不重新引入低带宽或硬性零断连目标，除非用户明确改变研究方向。
7. 修改奖励、网络结构或实验定义前先说明对checkpoint兼容性和论文对比的影响。
8. v3.4的320/640/960/1280/1600三随机种子验证已完成，五者探索均值均超过99%，但全连通
   率均值最高的960也只有88.294%；不得运行最终测试集或宣称模型收敛，下一步进入新架构的
   分阶段实现与验证。

后续只增量维护本文件的“当前状态与下一步”；不要重新加入已完成的对话过程和临时输出。
