# IR²通信感知探索项目：AI交接文档

> 最后更新：2026-10-08
> 用途：新会话先读本文恢复当前状态；论文细节见
> [CONNECTIVITY_AWARE_PAPER_PLAN.md](./CONNECTIVITY_AWARE_PAPER_PLAN.md)。

> 当前用户决定：继续v3.5-A训练，暂缓验证和最终测试，不要自行启动评估。
> 最新下载快照：checkpoint为4032回合，TensorBoard为4029回合；服务器实时进度需另行确认。
> 1280权重已经通过150回合验证集的平均指标，必须保留，但尚未冻结为最终模型。

## 1. 当前目标与验收标准

项目基于官方IR²研究未知室内环境中的多机器人通信感知探索。通信质量由距离和墙体共同决定，
使用连续RSSI和动态多跳通信图。允许短暂断连，但要减少长断连并加快重连；探索者、中继者和
恢复者应随任务进度动态交接，而不是固定角色。

当前做的是允许短暂断连的高连通率维护与主动恢复，不是严格持续连通保证，也不是固定周期
集合通信；连续RSSI指信号强度为连续数值，不表示物理运动全过程始终连通。
底层仍沿用IR²注意力图策略和SAC，动态角色是期望行为，尚不能宣称已学出角色分工。

第一阶段固定3台机器人，只使用原作者hybrid地图。当前唯一硬指标是在固定50张验证地图、
3个可复现通信随机种子（共150回合）上同时满足：

| 指标 | 门槛 |
|---|---:|
| 平均个体探索率 | ≥99% |
| 平均全连通率 | ≥90% |

成功率、完成步数、路径长度、断连次数、平均/最长断连和重连时间继续记录，但当前只作诊断。
验证达标后才能冻结checkpoint，并对50张最终测试地图运行一次；不得依据测试结果继续选模。

理想策略应表现为：

1. 三台机器人分散探索，不以长期抱团换取通信率。
2. 接近门口、墙角或弱信号边界时，个别机器人能减速、短暂停留或承担移动中继。
3. 发生断连后停止盲目深入，利用断连时长、RSSI趋势、信息年龄和上次全连通位置快速返回。
4. 重连后避免立即再次越界，随后继续探索并在需要时交接中继任务。
5. 行为仅依赖分布式局部观测；不使用全图真实位置、硬动作屏蔽或固定角色保证连通。

v3.5-A没有显式角色头、GRU或集中式critic，上述角色行为需要自然涌现；若仍不达标，再进入
通信成本critic、拉格朗日约束和逐机器人信用分配阶段。

## 2. 仓库、分支与受保护文件

正式仓库：

```text
/home/robot/test/IR2-Continuous-Connectivity/IR2-Multi-Robot-RL-Exploration
```

- 分支：`wall-aware-connectivity`
- GitHub：`git@github.com:Orange-wjc/IR2-Continuous-Connectivity.git`
- `upstream`：`https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration.git`
- 远端最新提交：`1b2938c Update v3.5-A handoff status`
- 2026-10-08本地检查：HEAD为`adec663 切换main分支`，仍在`wall-aware-connectivity`；
  本地远端引用为`1b2938c`，更新本文前工作区干净。2026-10-07服务器曾确认HEAD为`1b2938c`。
- `main`保留官方基线；旧目录`/home/robot/test/IR2-Multi-Robot-RL-Exploration`不再修改。

以下用户文件/目录历史上列为受保护内容；不要因当前Git状态干净就误删、覆盖或随意提交：

```text
docs/CONNECTIVITY_AWARE_PAPER_PLAN.md
docs/IDEAL_CONNECTIVITY_MODEL_DEMO.html
docs/build_hybrid1_full_demo.py
docs/hybrid1_full_demo/
docs/hybrid1_full_template.html
wall_aware_hybrid3_balanced_v3_4_validation/
wall_aware_hybrid3_recovery_v3_5_a_validation/
```

`docs/hybrid1_full_demo/index.html`是用户确认的理想行为参考，但由全图规则规划器生成，不是模型
输出，也不能作为论文实验结果。任何操作前先运行`git status --short`。

## 3. 数据、环境与指标口径

- 400张`DungeonMaps/test/hybrid`地图按种子`20260907`固定划分为300训练、50验证、50测试。
- 三个集合互不重叠，`hybrid/1.png`只在最终测试集；不再生成新训练地图。
- 这里的不重叠是文件名层面的划分，不等于地图几何独立。此前审计400张hybrid仅有105个
  D4旋转/镜像归一化几何组；当前300/50/50划分的跨集合几何重叠仍需单独审计，不宣称独立布局泛化。
- 本地环境：`BLMRE_py38`；AutoDL环境：`mrobot`。
- 本地只负责开发和读取下载结果，真实训练在AutoDL；不要在本地启动训练。
- AutoDL项目路径`/root/autodl-tmp/IR2-Continuous-Connectivity`。
- 2026-10-07服务器实测：RTX 4090 24GB；cgroup CPU配额`2500000/100000`，相当于25个
  逻辑核计算时间；内存上限`96636764160`字节，即90GiB。
- `nproc=224`、`free`显示约754GiB是宿主机可见资源，不是容器独占配额。
- 训练时一次采样GPU占用8117MiB、利用率39%；容器内存39.01GiB。这些是瞬时值，不是峰值。
- 训练入口为`driver.py`；环境仿真使用CPU，网络更新使用单GPU。续训不能直接使用默认
  `python driver.py`，否则默认policy-only配置会重新初始化，详见第7节。

指标必须按实现解释：

- `Explored Rate`是每台机器人自身belief覆盖率的平均值，不是团队融合地图覆盖率。
- `Success Rate`要求每台机器人自身belief都达到99%，比团队融合完成更严格。
- `Connectivity Rate`是整个团队通信图处于单一连通分量的团队步比例。
- 断连身份相对唯一最大连通分量判定；最大分量并列时所有机器人都记为断连。
- 测试种子为`TEST_RANDOM_SEED + run_index * NUM_TEST + episode_number`；不同checkpoint必须
  使用相同`run/地图/种子`键做配对比较。

## 4. 当前代码与v3.5-A配置

关键提交：

```text
c6aef5e Implement v3.4 synchronous team rollouts and stay actions
45d8b02 Document v3.4 validation and improve evaluation tracking
8db4c69 Add v3.5-A local recovery observations
5058b3c Harden v3.5-A recovery training
1b2938c Update v3.5-A handoff status
```

v3.4已实现同步团队rollout、统一团队奖励/next state、独立belief快照和显式stay动作。v3.5-A
在此基础上做局部恢复增强，独立实验目录为`wall_aware_hybrid3_recovery_v3_5_a`。

### 观测与迁移

- actor输入由11维扩展为16维：官方6维 + 10个通信特征。
- 前5个通信特征保持v3.4顺序和语义。
- 新5维为：30步长断连紧迫度、本地连续RSSI估计趋势、队友位置最大信息年龄、指向上次
  全队可达位置的候选恢复分数、20步本地可达分量比例历史。
- RSSI趋势使用本地地图、本机位置和队友最后已知位置，包含门限以下估计值；完全失联后仍能
  判断估计信号改善或恶化，同时由信息年龄表示估计可靠性。
- v3.4-960 actor第一层的前11列原样复制，新5列置零；初始化策略输出误差为0。

### 训练配置

```text
NUM_META_AGENT=20       BATCH_SIZE=128       REPLAY_SIZE=10000
MINIMUM_BUFFER_SIZE=2000                    MAX_EPS_STEPS=196
K_SIZE=30               NODE_PADDING_SIZE=360
INPUT_DIM=16            POLICY_LR=1e-6       Q_LR=1e-5
POLICY_ANCHOR_KL_WEIGHT=0.5                 critic warmup=1024 updates
```

- 从`model/wall_aware_hybrid3_balanced_v3_4/checkpoint_960.pth`只迁移actor。
- 冻结参考策略也是v3.4-960，用移动动作条件KL抑制探索能力遗忘。
- critic、目标critic、优化器、温度、Replay Buffer和episode计数全部重新初始化。
- 日常checkpoint每32 episode覆盖保存；每320 episode归档`checkpoint_<episode>.pth`。
- 不要增加Actor或batch，也不要让采样Actor使用单块GPU。

当前奖励保持v3.3/v3.4定义：20步窗口目标连通率0.90；弱信号、分量缺口、新断连权重分别为
0.20、0.50、0.20，低于预算时按压力启用；断连持续权重0.25始终有效，重连奖励为0。
探索进度权重5.0，完成奖励40；探索停滞宽限5步、20步饱和、每步最大惩罚0.02。

### 审查、训练与验证状态

`5058b3c`已完成以下修正：

- 断连期间RSSI趋势持续更新，且不引入全局真实队友位置。
- v3.5-A完整续训严格核对输入、观测、奖励、转移、动作、Replay字段、KL锚点及温度状态。
- 推理默认指向v3.5-A的16维配置；checkpoint身份写入CSV，错误配置在生成CSV前失败。
- `tests/test_v35_recovery.py`含4个回归测试，当前全部通过。
- 实际同步团队步已验证当前/下一观测均为16维，新特征有限。

2026-09-23已在AutoDL服务器首次启动v3.5-A训练。服务器上的4个回归测试全部通过，启动日志
确认从v3.4 episode 960只加载actor，并重新初始化critic、优化器、alpha和episode计数。
2026-10-07服务器再次运行4个回归测试，全部通过；当时出现的`OMP_NUM_THREADS`无效警告，
通过启动命令显式设置线程数为1消除。随后完成第6节所列验证。

2026-10-07用户明确要求暂停测试、继续训练；从1536回合checkpoint恢复。
2026-10-08读取下载文件确认最新checkpoint为4032回合，梯度更新31800次；续训前备份为
1536回合、12064次更新。两者均16维输入、相同v3.5-A合同，actor学习率仍为1e-6，Q为1e-5。

TensorBoard现有两个run：

- `Sep23_08-51-36_...`：首次训练，47个窗口，step 60至1532。
- `Oct07_18-23-50_...`：续训，77个窗口，step 1597至4029。

最后10个窗口的均值（不是固定地图配对比较）：

| 指标 | 首次训练末段 | 续训末段 |
|---|---:|---:|
| 探索率 | 99.269% | 99.669% |
| 成功率 | 96.841% | 99.581% |
| 全连通率 | 76.699% | 81.155% |
| 最长断连时长均值 | 13.577步 | 10.338步 |
| 断连次数均值 | 6.646 | 5.434 |
| 平均重连时间 | 4.297步 | 4.024步 |
| stay比例 | 0.132% | 0.074% |

续训按1537--2240、2241--2880、2881--3520、3521--4029分段，平均连通率为
78.812%、78.975%、79.600%、79.781%，改善缓慢，接近平台；相应最长断连均值为
12.498、11.675、10.977、10.743步。最新窗口连通率85.372%，不能单凭最后一点判断收敛。
两run所有已读取标量均无NaN/Inf；续训末10窗口Q loss约3.288、Q梯度范数182.685、
策略锚点KL约0.037。stay很少，不能认定已形成明确停留中继行为。

训练窗口仅用于诊断，不能与固定验证集90.96%直接比较，也不能证明4032比1280更好。
本地文件是下载快照，不能据此推断服务器已经停止训练。当前建议保持参数到约5000回合，
再检查日志趋势；5000是观察节点，不是自动停止条件或收敛保证。

## 5. 关键checkpoint与兼容性

| 用途 | 路径/状态 |
|---|---|
| 历史stage1 | `model/wall_aware_stage1/checkpoint.pth`，episode 928，SHA-256 `f861e42f5a29461c586e51e782db8eed5d125d11edba16ef06566c7f5d331934` |
| 失败基线 | `model/wall_aware_hybrid3_stage1/checkpoint.pth`，episode 672，SHA-256 `77638cb934c47b62338ba5cc3e599ff3523ee00adb9d9124872b5a1fbbc58a9e` |
| balanced v2早期 | `model/wall_aware_hybrid3_balanced_v2/checkpoint.pth`，episode 864，SHA-256 `1218764e84a50f0872b142eda1c2dd29cc970508926feea0ea211254bc23d466` |
| v2探索基线 | `model/wall_aware_hybrid3_balanced_v2_3520/checkpoint.pth`，SHA-256 `ce4ca5e759109d26bdfc2d4097221c399398df49c2572bde7c252f996af03e16` |
| v3.4归档 | `checkpoint_320/640/960/1280/1600.pth`，全部保留 |
| v3.5-A初始化 | v3.4 `checkpoint_960.pth`，episode 960、11维输入 |
| v3.5-A验证候选 | `checkpoint_1280.pth`已完成150回合验证并通过平均门槛，保留但未最终冻结 |
| v3.5-A最新下载 | `checkpoint.pth`为4032回合；TensorBoard到4029；320至3840每320回合归档均可读取 |
| v3.5-A验证前备份 | `model/wall_aware_hybrid3_recovery_v3_5_a/backup_before_validation_20261007/`，含1536及320/640/960/1280 |
| v3.5-A续训前备份 | `model/wall_aware_hybrid3_recovery_v3_5_a/backup_before_resume_20261007/checkpoint.pth`，1536回合 |

兼容边界：

- 官方checkpoint为6维，只能在`USE_CONNECTIVITY_FEATURES=False`时使用。
- v2/v3/v3.4为11维；v3.5-A为16维，不能直接完整续训11维checkpoint。
- v3.5-A第一次启动：`LOAD_MODEL=False`、`LOAD_POLICY_ONLY=True`、
  `POLICY_PRETRAINED_PATH`指向v3.4-960。
- v3.5-A自身完整续训：`LOAD_MODEL=True`、`LOAD_POLICY_ONLY=False`、
  `CONTINUE_LOG_ALPHA=True`，且checkpoint合同必须完全匹配。
- Replay Buffer只在内存中；重启后需重新收集2000条transition，不表示模型权重重置。
- 当前checkpoint不保存目标Q和完整随机数/采样状态；恢复时目标Q从当前Q复制。因此这是恢复
  已保存训练状态的续训，不应称为逐位一致的无中断重放。

## 6. v3.4验证结论与失败原因

五个v3.4 checkpoint均在相同50张验证地图上完成3个随机种子，共750个可配对回合：

| checkpoint | 平均探索率 | 平均全连通率 | 三个run连通率 | 成功数 |
|---:|---:|---:|---|---:|
| 320 | 99.597% | 87.452% | 91.272 / 87.227 / 83.858% | 149/150 |
| 640 | 99.428% | 85.848% | 86.274 / 83.375 / 87.893% | 146/150 |
| 960 | 99.329% | **88.294%** | 87.886 / 84.670 / 92.325% | 147/150 |
| 1280 | 99.180% | 86.597% | 84.174 / 86.658 / 88.958% | 143/150 |
| 1600 | 99.364% | 87.857% | 87.527 / 87.727 / 88.317% | 144/150 |

结论：探索全部达标，但全连通率全部低于90%。960均值最高，仍差1.706个百分点；它相对320
只提高0.841个百分点，配对近似95%置信区间跨0，不能宣称稳定更优。单种子320曾达到
91.272%，但三run均值只有87.452%，不得把单run结果当成收敛或达标证据。

轨迹和CSV诊断：

- 750回合中270个低于90%连通率，其中218个最长断连超过10步。
- 连通率与最长断连相关系数约-0.865；与stay比例和探索率几乎无关。
- 主要失败是断连后继续探索、缺少返回方向、在通信边界反复进出；stay几乎未使用。
- `107/386/323/83/55/229.png`是跨checkpoint反复出现的困难地图。
- v3.4最终测试集尚未运行，所有现有CSV的`split`都必须保持为`validation`解释。

历史版本只保留结论：v1/v3/v3.1形成“高连通、低探索”；v2探索强但连通率69.88%；v3.2
探索约99.6%但连通约74%；v3.3通信曾短暂到85%至86%后回落。继续只调标量奖励很难解决
长断连恢复、状态历史和信用分配问题，因此当前先验证可归因的v3.5-A局部恢复特征。

### v3.5-A固定验证结果（2026-10-07完成，已下载核对）

目录：`wall_aware_hybrid3_recovery_v3_5_a_validation/test_results/log/`，共8份CSV、400回合。
320/640只做run 0各50回合；960/1280分别完成run 0/1/2各150回合。
所有文件均为完整50张验证地图，地图、种子、checkpoint身份匹配，无执行错误、缺失、重复
或非有限数值。超时/探索失败保留在统计中，不是执行错误。

| 权重 | 样本数 | 平均探索率 | 平均全连通率 | 成功数 | 平均步数 | 最长断连均值 |
|---|---:|---:|---:|---:|---:|---:|
| v3.5-A 320（仅run 0） | 50 | 98.508% | 90.154% | 47/50 | 114.40 | 6.90步 |
| v3.5-A 640（仅run 0） | 50 | 99.483% | 87.144% | 49/50 | 104.44 | 10.18步 |
| v3.5-A 960 | 150 | 99.511% | 88.601% | 145/150 | 103.26 | 8.58步 |
| v3.5-A 1280 | 150 | **99.428%** | **90.956%** | 146/150 | 106.77 | 6.387步 |
| v3.4 960基线 | 150 | 99.329% | 88.294% | 147/150 | 105.71 | 7.793步 |

- v3.5-A 960三个run连通率：90.577%、88.599%、86.627%，合并未达90%。
- v3.5-A 1280三个run连通率：91.316%、89.350%、92.204%，合并通过平均门槛，但不是每run都通过。
- 1280相对v3.4-960连通率增加2.663个百分点；断连次数均值2.42降至1.86，平均步数增加1.053。
- 对相同地图/run/seed配对，按50张地图分组bootstrap，组内保留三个run，20000次重采样、
  种子20261007：1280连通率差的95%区间为[-0.456, +6.033]个百分点，仍跨0；960为
  [-2.771, +3.455]。1280连通率本身的同类区间为[88.042%, 93.620%]。
- 1280有37/150回合连通率低于90%，最长断连55步；960对应46/150、94步。
  1280的run 1、35.png连通率仅3.28%，属于需后续分析的异常困难案例，不能删除。
- 结论：1280通过预设样本均值工程门槛，但尚不能宣称统计上稳定优于基线、所有任务高连通，
  更不能称为持续连通保证。新1600及以后权重目前未验证。

曾建议冻结1280并进行最终测试，但用户随后明确选择继续训练、先不测试。因此当前未最终
冻结模型，也未运行最终测试；保留1280作为已验证的候选和后续对照。

## 7. 当前执行顺序与服务器命令

1. 遵从用户最新决定：继续训练，暂不启动验证或最终测试，不因旧交接步骤自动改回评估。
2. 若服务器正在训练，不重复启动。当前计划保持20个CPU采样Actor、单GPU learner、batch128，
   不改变奖励、学习率、网络或地图划分；到约5000回合检查训练趋势。
3. 持续保留所有归档、备份、1280验证候选和v3.4-960参考权重。不得用最新checkpoint覆盖
   历史候选或把训练曲线当验证结果。已超过先前3200观察节点，无须倒退训练。
4. 若未来恢复验证，先用验证集比较新归档和1280，不要直接在最终测试集选模；冻结后才开展
   预先规定的最终测试和匹配基线比较。后续消融验证局部恢复特征的贡献。
5. 若训练/验证长期无明显改善，再讨论独立通信成本critic、拉格朗日约束和逐机器人信用分配。
   这些尚未实现，不要未经用户同意在本轮同时引入，也不能保证会解决问题。

### 断点续训（仅在没有训练进程、确实需要恢复时运行）

服务器环境`mrobot`，工作目录`~/autodl-tmp/IR2-Continuous-Connectivity`。运行前备份当时的
`checkpoint.pth`到新的备份目录；原1536备份不覆盖。下列命令通过内存覆盖配置后执行入口，
不修改`parameter.py`，下一次直接执行`driver.py`仍会使用原始policy-only默认配置。

```bash
conda activate mrobot
cd ~/autodl-tmp/IR2-Continuous-Connectivity
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -u - <<'PY'
import runpy
import parameter
parameter.LOAD_MODEL = True
parameter.LOAD_POLICY_ONLY = False
parameter.CONTINUE_LOG_ALPHA = True
runpy.run_path('driver.py', run_name='__main__')
PY
```

启动应显示`Loading Model...`及`curr_episode set to:`实际恢复回合；若合同不兼容，查原因，
不要删除校验或改成policy-only绕过。上述命令不会在5000自动停止，终端连接需保持。

### TensorBoard

另开服务器终端，不能关闭训练终端：

```bash
conda activate mrobot
cd ~/autodl-tmp/IR2-Continuous-Connectivity
tensorboard --logdir train/wall_aware_hybrid3_recovery_v3_5_a --host 127.0.0.1 --port 6006
```

VS Code“端口”面板转发6006，在浏览器打开；分析只勾选最新续训run，平滑0.2--0.3。
关注探索、成功、全队连通、最长断连、重连和stay，不把`Agents Connected [%]`当全队连通率。

### 将来恢复评估时的固定设置

本次推理使用8个并行评估Actor（最早320使用5个），与训练20个CPU Actor是不同配置。
固定`MPLBACKEND=Agg`及BLAS/OMP/MKL线程数1，`IR2_EVAL_SPLIT=validation`，
`IR2_EVAL_MAP_FILES=''`保证完整50图，`IR2_EVAL_SAVE_GIFS=0`，`IR2_EVAL_USE_GPU=1`，
v3.5-A通信特征维度10。run 0筛选后用`IR2_EVAL_RUN_INDICES=1,2`补测，不重复run 0。
结果目录由实验名和split组成；不同权重用时间戳CSV及checkpoint身份区分。
进程退出码为0仍须检查50行、地图唯一性、种子和`status/error`，不可只凭退出码认定完整成功。

v3.5-A当前是默认推理配置。回看v3.4必须显式设置：

```bash
IR2_EVAL_EXPERIMENT_NAME=wall_aware_hybrid3_balanced_v3_4 \
IR2_EVAL_CONNECTIVITY_FEATURE_DIM=5 \
IR2_EVAL_MODEL_PATH=model/wall_aware_hybrid3_balanced_v3_4/checkpoint_960.pth \
python -u test_driver.py
```

## 8. 常见问题与禁止事项

- `node_coords.shape[0] >= 360`偶发时会跳过单回合；低于1%至2%暂不处理，超过5%再统计。
- 候选边超过`K_SIZE=30`的形状错误已由`b12adff`修复，不要撤销。
- v3首次更新的24 GB CUDA OOM已通过预热禁用policy梯度、提前释放图和串行更新Q1/Q2缓解；
  若v3.5-A仍在首次更新OOM，单独将batch从128降至96，不要同时改其他参数。
- TensorBoard已验证组合为2.14.0 + protobuf 4.25.3；分析只选最新run，平滑约0.2至0.3。
- 不运行最终测试集选模，不宣称固定95%连通率，不把演示当模型结果。
- 不重新引入低带宽、硬性零断连、硬动作接管、全图actor输入或固定中继角色，除非用户明确
  改变研究方向。
- 修改奖励、网络、观测语义或实验定义时必须使用新实验目录，并说明checkpoint兼容性。
- 保留所有历史checkpoint和2026-09-15的15个完整v3.4验证CSV；中断/不完整CSV不得汇总。

## 9. 新会话最短检查清单

1. `git status --short --branch`，保护未跟踪文件。
2. 确认分支和远端提交；服务器代码必须包含`5058b3c`。
3. 当前任务是继续v3.5-A训练、暂缓测试。最新下载权重4032、日志4029，服务器实时状态另查。
4. 核对16维配置；自身续训必须full resume三标志，不能用policy-only重新初始化。
5. 1280已通过150回合平均门槛，但改善区间跨0、长断连仍存在；保留候选，尚未最终冻结。
6. 新归档尚未验证，不能以loss或最新权重代表更好；将来恢复评估先验证选模，再冻结最终测试。

后续只增量维护“当前状态、验证结果、下一步”和必要的兼容性信息；不要重新加入完整对话过程。
