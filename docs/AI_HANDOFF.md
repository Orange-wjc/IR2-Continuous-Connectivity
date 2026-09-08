# IR²通信感知探索项目：AI交接文档

> 最后更新：2026-09-09
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
```

以上提交均已推送到 `origin/wall-aware-connectivity`。

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

FOLDER_NAME = 'wall_aware_hybrid3_balanced_v3'
MODEL_PATH = 'model/wall_aware_hybrid3_balanced_v3/checkpoint.pth'
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
POLICY_TRANSFER_CRITIC_WARMUP_UPDATES = 1024
```

第一阶段训练固定使用3台机器人。原`DungeonMaps/test/hybrid`的400张地图由
`map_splits.py`按固定种子`20260907`重新划分为300张训练、50张验证、50张测试；
三个集合互不重叠，`hybrid/1.png`固定保留在测试集。文件仍位于原目录，逻辑划分由
`MAP_FILE_NAMES`控制。批量测试固定3台机器人、遍历50张保留测试地图一次，并默认
关闭GIF。v3只迁移balanced v2 episode 3520的策略，不覆盖任何旧实验。训练地图
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
- 用户将工程验收目标设为平均全连通率95%以上、任务成功率99%以上，并要求探索
  步数不要明显增加；这些是模型选择目标，不能在论文中预先宣称已经达到。
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
- `test_parameter.py`已指向未来的v3权重，50图测试为每个地图使用
  `TEST_RANDOM_SEED + run_index * NUM_TEST + episode_number`固定Python、NumPy和PyTorch
  随机种子，CSV同时记录run、种子及`team_bottleneck_rssi`，以便不同checkpoint使用
  相同RSSI条件复现实验。`NUM_RUN>1`时每轮使用不同但可复现的信道种子。
- v3本地验证已完成：五个相关Python文件AST解析通过，episode 3520策略严格加载通过，
  奖励随5/10/30步断连持续增强，全队RSSI最大生成树与软恢复特征数值检查通过；另有
  一个不保存GIF的完整CPU回合成功完成，15路episode buffer长度一致且新增指标齐全。
- v3在服务器上已确认正确读取v2 episode 3520策略，但洁净的24 GB GPU在
  首次更新时出现峰值显存不足。`driver.py`已改为预热时禁用policy梯度、提前释放
  policy计算图并串行更新Q1/Q2；本地最小训练步已覆盖预热与正常更新两条路径。

下一步：

1. 服务器先拉取包含显存优化的最新提交，确认
   `model/wall_aware_hybrid3_balanced_v2_3520/checkpoint.pth`存在，执行
   `ray stop --force`后再运行`python -u driver.py`。启动日志必须显示从episode 3520
   加载policy、critic等状态重新初始化；v3的episode从0开始，不能出现
   `curr_episode set to: 3520`。
2. 等待2000条transition填满Replay Buffer及1024次critic预热完成。日志出现
   `Critic warmup complete; enabling policy and alpha updates.`后，策略才开始微调。
3. 重点监控成功率、探索率、完成步数、连通率和新增的最长断连/重连曲线。若探索率
   或成功率明显退化，不要盲目长训；保留每320轮归档权重并在验证集上选择候选模型。
4. 调参和模型选择只使用固定验证集，不再根据50张保留测试地图逐图调整；达到候选
   状态后才用固定随机种子运行最终测试，避免进一步测试集泄漏。
5. 保存episode 928、hybrid3 stage1 episode 672、balanced v2 episode 864和3520权重，
   作为历史基线和失败/成功对照。v3验证达标后再决定是否进入4机器人阶段。

## 10. 新会话检查清单

1. 确认位于正式仓库和 `wall-aware-connectivity` 分支。
2. 阅读本文和论文方案文档。
3. 运行 `git status --short`，保护用户未跟踪文件。
4. 本地使用 `BLMRE_py38`；AutoDL使用 `mrobot`。
5. 确认GPU可见、checkpoint存在，并核对续训episode。
6. 不重新引入低带宽或硬性零断连目标，除非用户明确改变研究方向。
7. 修改奖励、网络结构或实验定义前先说明对checkpoint兼容性和论文对比的影响。

后续只增量维护本文件的“当前状态与下一步”；不要重新加入已完成的对话过程和临时输出。
