# IR²通信感知探索项目：AI交接文档

> 最后更新：2026-09-07
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

FOLDER_NAME = 'wall_aware_hybrid3_stage1'
MODEL_PATH = 'model/wall_aware_hybrid3_stage1/checkpoint.pth'
LOAD_MODEL = False
CONTINUE_LOG_ALPHA = True
SAVE_TRAINING_GIFS = False

REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
USE_CONNECTIVITY_FEATURES = True
CONNECTIVITY_FEATURE_DIM = 5
INPUT_DIM = 11
```

第一阶段训练固定使用3台机器人。原`DungeonMaps/test/hybrid`的400张地图由
`map_splits.py`按固定种子`20260907`重新划分为300张训练、50张验证、50张测试；
三个集合互不重叠，`hybrid/1.png`固定保留在测试集。文件仍位于原目录，逻辑划分由
`MAP_FILE_NAMES`控制。批量测试固定3台机器人、遍历50张保留测试地图一次，并默认
关闭GIF。新实验从头训练，不覆盖或续接旧`wall_aware_stage1`曲线。

通信奖励权重：弱信号0.2、断连0.5、断连持续时间0.5、重连奖励0.5。

环境仿真主要消耗CPU，网络更新使用GPU。当前25核服务器先使用20个CPU Actor，
为学习器、Ray和系统保留约5核；GPU利用率可能在0%至100%之间脉冲变化。不要直接
把Actor开满25个，也不要设置`USE_GPU=True`让采样Actor争抢单块GPU。应比较15与20
Actor各约100轮的episode吞吐；20个稳定且CPU仍有余量时才试22个。若出现Ray资源
警告、CPU长期100%或吞吐下降，则退回18至20个。`BATCH_SIZE=128`暂不增加。

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

服务器必须将其放到运行 `driver.py` 的项目根目录下：

```text
/root/autodl-tmp/IR2-Continuous-Connectivity/model/wall_aware_stage1/checkpoint.pth
```

保持 `LOAD_MODEL=True` 和 `CONTINUE_LOG_ALPHA=True`。启动成功应看到：

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

## 7. TensorBoard与当前曲线

已验证兼容组合：TensorBoard 2.14.0 + protobuf 4.25.3。protobuf 5.29.6会导致：

```text
TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'
```

启动命令：

```bash
tensorboard --logdir ./train/wall_aware_stage1 --host 0.0.0.0 --port 6006
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

- 分支为 `wall-aware-connectivity`；三机器人hybrid划分基线提交为`85c050b`。
- RTX 4090/25核/90GB服务器配置为20个并行CPU仿真和单GPU训练；该值需通过约
  100轮吞吐测试与15个Actor对照验证，不能仅凭硬件规格认定一定更快。
- 候选边形状问题已经修复。
- episode 928的11维checkpoint已找回并验证。
- 服务器TensorBoard最后观察到约episode 1561；本地仓库的已验证checkpoint仍是
  episode 928。不要把服务器曲线轮数当成本地权重轮数；应下载并核验服务器最新
  checkpoint的`episode`字段。
- 当前连通性显著改善，但探索率、成功率和奖励后期波动并下降，尚未形成最终结论。
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
- 当前代码中的“动态中继”主要由候选节点的relay特征和奖励间接学习；尚无中继
  占用时间统计。`emergency_reconnect_required`目前只是标志位，没有动作接管规则。
- `test_parameter.py`已指向`wall_aware_hybrid3_stage1`的11维checkpoint，并固定使用
  3台机器人和50张保留测试地图；新模型尚未训练完成前运行测试会找不到checkpoint。

下一步：

1. 将当前代码部署到服务器，确认`DungeonMaps/test/hybrid`包含完整的1至400号地图。
2. 从头启动`wall_aware_hybrid3_stage1`，先跑约100轮冒烟和计时测试，确认无大量
   padding跳过、checkpoint正常保存、TensorBoard指标正常，再决定正式轮数。
3. 训练后用`test_parameter.py`加载新11维checkpoint，对50张保留地图各测试一次。
   至少报告探索率、成功率、步骤/路径、连通率、断连次数、平均/最长断连时长和
   重连时间；训练曲线不能替代该评估。
4. 在保留的`hybrid/1.png`上输出真实模型轨迹，与目标演示对照是否分散探索、是否
   长期抱团、接应位置是否移动、是否发生中继交接以及是否完成探索。
5. 保存旧episode 928和服务器1500+模型作为历史基线，不覆盖其checkpoint和曲线。
6. 根据三机器人独立测试结果决定第二阶段：若效果达到要求，再引入4机器人微调；
   若只提高连接却降低探索，则在新实验目录调整奖励，不在同一run中途改目标。

## 10. 新会话检查清单

1. 确认位于正式仓库和 `wall-aware-connectivity` 分支。
2. 阅读本文和论文方案文档。
3. 运行 `git status --short`，保护用户未跟踪文件。
4. 本地使用 `BLMRE_py38`；AutoDL使用 `mrobot`。
5. 确认GPU可见、checkpoint存在，并核对续训episode。
6. 不重新引入低带宽或硬性零断连目标，除非用户明确改变研究方向。
7. 修改奖励、网络结构或实验定义前先说明对checkpoint兼容性和论文对比的影响。

后续只增量维护本文件的“当前状态与下一步”；不要重新加入已完成的对话过程和临时输出。
