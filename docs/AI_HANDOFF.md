# IR²通信感知探索项目：AI交接文档

> 最后更新：2026-09-23
> 用途：新会话先读本文恢复当前状态；论文细节见
> [CONNECTIVITY_AWARE_PAPER_PLAN.md](./CONNECTIVITY_AWARE_PAPER_PLAN.md)。

## 1. 当前目标与验收标准

项目基于官方IR²研究未知室内环境中的多机器人通信感知探索。通信质量由距离和墙体共同决定，
使用连续RSSI和动态多跳通信图。允许短暂断连，但要减少长断连并加快重连；探索者、中继者和
恢复者应随任务进度动态交接，而不是固定角色。

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
- `main`保留官方基线；旧目录`/home/robot/test/IR2-Multi-Robot-RL-Exploration`不再修改。

以下用户文件/目录当前未跟踪，禁止误删、覆盖或随意提交：

```text
docs/CONNECTIVITY_AWARE_PAPER_PLAN.md
docs/IDEAL_CONNECTIVITY_MODEL_DEMO.html
docs/build_hybrid1_full_demo.py
docs/hybrid1_full_demo/
docs/hybrid1_full_template.html
wall_aware_hybrid3_balanced_v3_4_validation/
```

`docs/hybrid1_full_demo/index.html`是用户确认的理想行为参考，但由全图规则规划器生成，不是模型
输出，也不能作为论文实验结果。任何操作前先运行`git status --short`。

## 3. 数据、环境与指标口径

- 400张`DungeonMaps/test/hybrid`地图按种子`20260907`固定划分为300训练、50验证、50测试。
- 三个集合互不重叠，`hybrid/1.png`只在最终测试集；不再生成新训练地图。
- 本地环境：`BLMRE_py38`；AutoDL环境：`mrobot`。
- AutoDL：RTX 4090、25核CPU，项目路径`/root/autodl-tmp/IR2-Continuous-Connectivity`。
- 训练入口：`python -u driver.py`；环境仿真使用CPU，网络更新使用单GPU。

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
已下载的TensorBoard快照来自唯一run，记录到episode 1468，共45个统计窗口：

- 全程训练窗口平均探索率99.530%、成功率98.236%、全连通率77.841%。
- 最近10个窗口平均探索率99.287%、成功率97.305%、全连通率78.228%；最新窗口分别为
  99.715%、100%和70.918%。
- 按episode 60--316、348--636、668--956、988--1276、1308--1468分段，平均全连通率依次为
  78.821%、75.468%、78.663%、78.227%、78.313%，没有持续上升趋势。
- 最近10个窗口的平均最长断连为12.53步、平均断连次数为6.46；最新窗口最长断连21.03步。
  最近10个窗口stay比例仅0.115%，长断连恢复和主动停留/中继行为尚未明显形成。
- 数值训练未崩溃且没有NaN；最近10个窗口Q loss均值约5.09。Q梯度范数升至317.6，但仍远低于
  代码中的2000裁剪上限；移动动作策略锚点KL最近约0.022，策略仍接近v3.4初始化策略。

这些数值来自随机训练地图的滑动窗口，只能用于诊断，不能与固定验证集指标直接等同，也不能
据此选模或宣称未达标。当前尚无v3.5-A验证结果；必须对归档checkpoint运行固定验证协议。

## 5. 关键checkpoint与兼容性

| 用途 | 路径/状态 |
|---|---|
| 历史stage1 | `model/wall_aware_stage1/checkpoint.pth`，episode 928，SHA-256 `f861e42f5a29461c586e51e782db8eed5d125d11edba16ef06566c7f5d331934` |
| 失败基线 | `model/wall_aware_hybrid3_stage1/checkpoint.pth`，episode 672，SHA-256 `77638cb934c47b62338ba5cc3e599ff3523ee00adb9d9124872b5a1fbbc58a9e` |
| balanced v2早期 | `model/wall_aware_hybrid3_balanced_v2/checkpoint.pth`，episode 864，SHA-256 `1218764e84a50f0872b142eda1c2dd29cc970508926feea0ea211254bc23d466` |
| v2探索基线 | `model/wall_aware_hybrid3_balanced_v2_3520/checkpoint.pth`，SHA-256 `ce4ca5e759109d26bdfc2d4097221c399398df49c2572bde7c252f996af03e16` |
| v3.4归档 | `checkpoint_320/640/960/1280/1600.pth`，全部保留 |
| v3.5-A初始化 | v3.4 `checkpoint_960.pth`，episode 960、11维输入 |
| v3.5-A训练 | TensorBoard已记录到episode 1468；服务器应已产生320/640/960/1280归档，验证前必须逐个核对并备份 |

兼容边界：

- 官方checkpoint为6维，只能在`USE_CONNECTIVITY_FEATURES=False`时使用。
- v2/v3/v3.4为11维；v3.5-A为16维，不能直接完整续训11维checkpoint。
- v3.5-A第一次启动：`LOAD_MODEL=False`、`LOAD_POLICY_ONLY=True`、
  `POLICY_PRETRAINED_PATH`指向v3.4-960。
- v3.5-A自身完整续训：`LOAD_MODEL=True`、`LOAD_POLICY_ONLY=False`、
  `CONTINUE_LOG_ALPHA=True`，且checkpoint合同必须完全匹配。
- Replay Buffer只在内存中；重启后需重新收集2000条transition，不表示模型权重重置。

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

## 7. 下一步执行顺序

1. AutoDL服务器进入正式仓库，确认`wall-aware-connectivity`为`1b2938c`或更新提交，并确认当前
   v3.5-A训练进程及最新episode状态。
2. 首次启动检查已经完成：4个回归测试通过，v3.4源权重存在且启动日志确认了policy-only迁移。
   继续保留源权重：

   ```text
   model/wall_aware_hybrid3_balanced_v3_4/checkpoint_960.pth
   ```

3. 在服务器逐个核对并备份v3.5-A的`checkpoint_320/640/960/1280.pth`；训练到1600时继续保留
   `checkpoint_1600.pth`。不能只保留会被覆盖的`checkpoint.pth`，也不能用TensorBoard文件
   代替模型权重。
4. 尽快在相同50张验证地图的固定run 0筛选320/640/960/1280；训练曲线未显示通信率持续改善，
   因此不要只依据loss、最后窗口或最后checkpoint继续盲目长训。
5. 只有探索率保持≥99%且连通率相对v3.4有明确改善的候选，才运行run 0/1/2完整三种子验证。
   训练窗口指标不能作为90%门槛的判定依据。
6. 三run总体同时达到99%探索和90%全连通后冻结最佳checkpoint，再运行一次最终测试。
7. 若v3.5-A归档经验证仍未达标，下一阶段增加独立通信成本critic、拉格朗日0.10断连成本约束和
   逐机器人信用分配；之后才考虑GRU、专家BC和持续角色头。

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
3. 确认当前任务是v3.5-A归档核对和验证；训练日志已记录到episode 1468，但尚无固定验证结果。
4. 核对源checkpoint路径、输入维度16和policy-only迁移配置。
5. 训练只看验证checkpoint，不以loss或最后checkpoint代表收敛。
6. 只有三run验证同时达到99%探索、90%全连通，才能冻结并进入一次最终测试。

后续只增量维护“当前状态、验证结果、下一步”和必要的兼容性信息；不要重新加入完整对话过程。
