# IR²通信感知探索项目：AI交接文档

> 最后更新：2026-09-02
> 用途：供新的AI会话快速恢复项目上下文。开始工作前，先阅读本文，再按需阅读 [CONNECTIVITY_AWARE_PAPER_PLAN.md](./CONNECTIVITY_AWARE_PAPER_PLAN.md)。

## 1. 用户目标

用户准备写一篇多机器人探索小论文，并在官方 IR² 代码基础上实现改进。

当前研究目标是：

> 在存在距离损耗和墙体信号衰减的未知室内环境中，通过强化学习提高机器人团队的通信连通率，允许短暂断连，但尽量缩短断连时间、加快重连，并学习动态探索者与中继者行为。

通俗比喻是“弹力通信绳”：

- 信号良好时，机器人可以自由分散。
- 信号变弱时，策略逐渐增加通信约束。
- 接近断连时，部分机器人可在门口、转角或走廊充当中继。
- 允许短暂断连，但断连越久，重连优先级越高。
- 中继者不是固定角色，可以动态交接。

## 2. 已确定的研究决策

### 保留

1. 距离和墙体共同决定通信质量。
2. 使用连续 RSSI，而不只使用“连接/断开”二值状态。
3. 构建动态多机器人通信图，允许多跳连接。
4. 记录并优化连通率、断连时间和重连速度。
5. 让策略学习动态探索者/中继者行为。
6. 以 IR² 的图强化学习和重连机制为主要基础。

### 不采用或暂缓

1. **不把低带宽作为当前论文重点。**
2. **不要求全程零断连。**
3. 不使用“所有可能断连的动作都被硬屏蔽”的严格持续连通方案。
4. 第一版暂不加入复杂网络时延、拥塞、在线墙体材料识别、高斯过程无线电地图或真实机器人实验。

### 研究表述

不再使用“保证持续连通”作为主要表述，应使用：

- 通信感知探索（connectivity-aware exploration）
- 高连通率探索
- 弹性连通约束
- 短时断连与快速重连

不能预先声称固定的 95% 连通率，也不能声称任意地图均可完成全部探索。最终数值应由基线实验决定。

## 3. 与已有工作的关系

### IR²

- 论文：IR²: Implicit Rendezvous for Robotic Exploration Teams under Sparse Intermittent Connectivity
- 重点：学习何时分开探索、何时重新连接和共享信息。
- 本项目区别：使用连续 RSSI、墙体衰减、断连持续时间和动态中继倾向。
- 链接：https://arxiv.org/abs/2409.04730

### PRCL

- 论文：Privileged Reinforcement and Communication Learning for Distributed, Bandwidth-limited Multi-robot Exploration
- 重点：用固定长度学习消息减少通信量。
- 当前项目不研究低带宽；PRCL仅作为分布式学习、图注意力和特权训练参考。
- 链接：https://arxiv.org/abs/2407.20203

### 必须注意的相近工作

- 2007年已有“持续保持无线网络连接的多机器人探索”：https://doi.org/10.1016/j.conengprac.2006.08.007
- 2020年已有DRL保证多机器人导航连通：https://proceedings.mlr.press/v100/lin20a.html
- PropEM-L研究环境/墙体对RSSI预测的影响：https://www.roboticsproceedings.org/rss18/p014.html
- DCM-RSSI研究在线RSSI学习与全局连通控制：https://arxiv.org/abs/2410.05798

因此不能把“持续连接”“强化学习保持连接”或“墙体影响信号”单独宣称为首次创新。创新必须落在这些因素与未知环境探索、断连时间和动态中继的具体结合方式上。

## 4. 仓库与分支

### 正式开发仓库

```text
/home/robot/test/IR2-Continuous-Connectivity/IR2-Multi-Robot-RL-Exploration
```

GitHub：

```text
https://github.com/Orange-wjc/IR2-Continuous-Connectivity
```

当前分支：

```text
wall-aware-connectivity
```

远程关系：

- `origin`：`git@github.com:Orange-wjc/IR2-Continuous-Connectivity.git`
- `upstream`：`https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration.git`
- `main`：保留官方 IR² 基线。
- `wall-aware-connectivity`：论文开发分支。

仓库已配置专用SSH认证。不要在文档或对话中输出私钥内容。

### 不再作为正式实现的旧项目

```text
/home/robot/test/IR2-Multi-Robot-RL-Exploration
```

该目录曾被多次修改，仅可用于参考，后续实现不要在这里进行。

### PRCL参考项目

```text
/home/robot/test/Bandwidth-Limited-Multi-Robot-Exploration
```

当前不以它作为主实验平台。

## 5. 文档与生成物

### 新仓库文档

- `docs/CONNECTIVITY_AWARE_PAPER_PLAN.md`
  - 完整研究方案。
  - 已更新为允许短暂断连的通信感知方案。
- `docs/AI_HANDOFF.md`
  - 本交接文档。
- `docs/IDEAL_CONNECTIVITY_MODEL_DEMO.html`
  - 自包含的交互式概念展示页，用于说明理想模型的探索者/中继者行为。
  - 这是设计目标示意，不是训练模型实际输出。

`AI_HANDOFF.md` 随本阶段代码一并维护；其余文档是否提交应以 `git status` 为准，不要误删或覆盖未跟踪文件。

### 已删除的旧文档

```text
/home/robot/test/IR2-Multi-Robot-RL-Exploration/docs/CONTINUOUS_CONNECTIVITY_PAPER_PLAN.md
```

旧文档强调严格持续连接，已经删除。

### 理想模型交互展示

```text
docs/IDEAL_CONNECTIVITY_MODEL_DEMO.html
```

这是人工设计的交互式概念页面，不是训练模型输出。它用于直观展示墙体衰减、动态多跳链路、短暂断连、快速重连和动态中继角色。

## 6. Conda、CUDA与硬件

使用的Conda环境：

```bash
conda activate BLMRE_py38
```

已验证环境：

- Python：3.8.20
- PyTorch：2.3.1+cu121
- Ray：2.10.0
- SciPy：1.10.1
- scikit-learn：1.3.2
- scikit-image：0.21.0
- Matplotlib：3.7.5
- pandas：2.0.3

官方README声明的主要版本较旧：

- PyTorch 1.10.0
- Ray 1.10.0
- scikit-image 0.19.3
- scikit-learn 1.2.1
- Matplotlib 3.6.3

当前较新环境已经成功完成一次官方模型推理，但存在Ray工作目录兼容问题，见后文。

GPU：

- NVIDIA GeForce RTX 4060 Laptop GPU
- 显存：8188 MiB
- 驱动：580.173.02
- 驱动支持CUDA：13.0
- PyTorch编译CUDA：12.1
- `torch.cuda.is_available() == True`
- GPU数量：1

注意：普通沙箱内执行时曾错误显示GPU不可用；需要允许访问宿主GPU后才可正确检测和运行。

## 7. 官方IR²基线冒烟测试

### 测试配置

- 官方 `model/stage2/checkpoint.pth`
- `hybrid` 测试集，第0张地图
- 4台机器人
- GPU推理
- 1个回合
- 保存GIF
- 关闭图边可视化以缩短绘图时间
- 未修改任何项目代码或参数文件

### 测试结果

- 测试成功：是
- 探索步数：48（终端内部零起始显示为Step 47）
- 地图覆盖率：0.9997123595505618，约99.971%
- 最大机器人路径长度：2206.865001633915
- 图节点数：234
- 跳过回合：0
- CSV中的 `connectivity=False`

### 指标解释

现有 `env.py` 中：

```python
self.agents_connected_percentage = 1 - (len(self.agents_comms_broken) / self.n_agent)
self.connectivity_rate = (len(self.agents_comms_broken) == 0)
```

因此当前 `connectivity_rate` 实际是“当前/最终时刻是否所有机器人均连接”的布尔值，不是整个回合的连通率。CSV中的 `False` 不能解释为“全程连通率为0”。

下一阶段首先应补充真正的时序指标：

- 每一步是否全局连通。
- 回合全程连通步数比例。
- 每台机器人平均/最长连续断连步数。
- 平均重连时间。
- 最大连通分量比例。
- 最弱通信边RSSI。

### GIF位置

全局：

```text
mar_inference/test_results/gifs/merged/eps0_merged_explored_rate_0.9997.gif
```

单机器人：

```text
mar_inference/test_results/gifs/robot_1/eps0_robot1_explored_rate_0.9997.gif
mar_inference/test_results/gifs/robot_2/eps0_robot2_explored_rate_0.9997.gif
mar_inference/test_results/gifs/robot_3/eps0_robot3_explored_rate_0.9997.gif
mar_inference/test_results/gifs/robot_4/eps0_robot4_explored_rate_0.9997.gif
```

`mar_inference/test_results/*` 被 `.gitignore` 忽略，不会出现在普通Git状态中。

CSV：

```text
/tmp/ir2_smoke_test/log/data_2026-09-02_135612.csv
```

另有一次失败启动产生的仅含表头文件：

```text
/tmp/ir2_smoke_test/log/data_2026-09-02_135537.csv
```

## 8. 已知运行问题

### Ray工作进程找不到项目模块

第一次使用Ray 2.10运行时，Actor创建失败：

```text
ModuleNotFoundError: No module named 'model'
```

原因：Ray工作进程没有继承项目目录作为Python模块搜索路径。运行时显式设置：

```bash
PYTHONPATH=/home/robot/test/IR2-Continuous-Connectivity/IR2-Multi-Robot-RL-Exploration
```

后测试成功。

### 运行时参数覆盖对Ray Actor不完全生效

主进程通过内存覆盖把日志路径设置到 `/tmp/ir2_smoke_test`，但Ray Actor重新导入了默认 `test_parameter.py`，所以GIF仍写入默认：

```text
mar_inference/test_results/gifs
```

后续应使用正式的小型测试配置文件或让Ray `runtime_env`显式传递环境，避免主进程与Actor参数不一致。

## 9. 已完成的代码实现

当前分支已有以下4个提交：

```text
9c6b785 Add continuous connectivity metrics
7635180 Visualize RSSI connectivity in GIFs
9aaf3bd Add connectivity-aware training features
017a3ed Speed up training data collection
```

主要实现如下：

- `ss_realistic_model.py`
  - 输出连续RSSI、墙体穿越次数、障碍/自由空间距离和通信安全余量。
  - 将逐像素Python循环改为NumPy向量化，保持原公式不变。
- `env.py`
  - 维护通信质量、时序连通率、断连时长、重连时间和最大连通分量等指标。
  - 加入通信相关软奖励，并缓存节点坐标到索引的精确查找。
- `multi_robot_worker.py`
  - 训练观测加入5个通信特征，训练输入由官方6维扩展为11维。
  - 删除无用深拷贝并优化图掩码生成。
- `test_multi_robot_worker.py`
  - 记录并可视化RSSI链路；测试绘图仅在 `SAVE_GIFS=True` 时执行。
- `parameter.py`
  - 训练默认启用通信特征，使用新的训练目录 `wall_aware_stage1`。
  - `SAVE_TRAINING_GIFS=False`，正式训练默认不生成GIF。
- `test_parameter.py`
  - 默认关闭新增通信输入，保持官方6维预训练模型兼容。
- `runner.py`
  - CPU训练Actor不再预留GPU份额，并限制每个Actor内部计算线程数，减少资源争抢。

兼容关系必须注意：

- 官方 checkpoint：6维输入，应保持测试配置 `USE_CONNECTIVITY_FEATURES=False`。
- 新训练模型：11维输入，应在训练和测试中启用通信特征，并加载新 checkpoint。
- 官方6维 checkpoint 不能直接加载到11维网络中。

## 10. 实验设计摘要

建议主要对比：

1. 原始 IR²。
2. 只按距离判断通信的 IR²。
3. 考虑墙体 RSSI、但没有断连时长设计的 IR²。
4. 完整方法：连续 RSSI + 断连时长约束 + 动态中继。

主要指标：

- 覆盖率、完成步数、总路径长度和成功率。
- 真正的回合连通率。
- 断连次数、平均/最长断连时长。
- 平均重连时间。
- 最大连通分量比例。
- 最弱通信树边RSSI。
- 关键中继占用时间。

最终目标是在探索效率没有明显下降的前提下，提高连通率并缩短断连与重连时间。

## 11. 当前状态

- 正式代码已完成连续RSSI指标、通信可视化、通信感知状态/奖励和训练加速修改。
- 官方预训练模型已在修改后的环境中成功完成GPU推理测试。
- 用户随后直接运行 `python test_driver.py`，GIF已正常生成。
- 该测试组合是“官方6维预训练模型 + 修改后的环境”，不是新11维通信感知模型。
- 训练加速修改不改变网络结构、奖励权重、地图分布、机器人数量、更新比例或训练轮数。
- RSSI新旧实现经过200组随机地图/链路等价性检查，结果一致。
- 节点索引缓存和图掩码经过新旧实现等价性检查，结果一致。
- 3机器人单步训练数据采集测试通过，且未生成训练图片。
- 2500条链路的局部微基准中，RSSI子程序由约0.108秒降至0.039秒，约快2.79倍；这不是整体训练速度倍数。
- 当前开发分支：`wall-aware-connectivity`。
- 最新代码提交：`017a3ed Speed up training data collection`。
- 测试GIF是被忽略文件，不属于待提交代码。
- 方案目标为弹性、高连通率探索，允许短暂断连但要求尽快重连。

## 12. 下一步任务

按以下顺序继续：

1. 将 `wall-aware-connectivity` 分支部署到GPU服务器。
2. 创建与本地一致的 `BLMRE_py38` 环境并确认CUDA、Ray和项目 `PYTHONPATH`。
3. 检查 `parameter.py`：`USE_CONNECTIVITY_FEATURES=True`、`INPUT_DIM=11`、`SAVE_TRAINING_GIFS=False`、`LOAD_MODEL=False`。
4. 先运行少量训练回合，确认经验采集、梯度更新、checkpoint和TensorBoard日志正常。
5. 冒烟测试通过后再启动正式训练：`python driver.py`。
6. 定期保存checkpoint，但训练阶段保持GIF关闭；需要观察轨迹时单独运行 `python test_driver.py`。
7. 新模型训练完成后，在 `test_parameter.py` 启用通信特征并指向11维新checkpoint。
8. 使用固定地图和随机种子比较官方IR²与新模型的探索和连通性指标。

服务器训练前不要使用官方6维 checkpoint 继续训练11维网络；除非额外实现并验证部分权重迁移。

## 13. 新AI开始工作时的检查清单

1. 确认当前目录是正式新仓库，而不是旧修改版。
2. 阅读本文和 `docs/CONNECTIVITY_AWARE_PAPER_PLAN.md`。
3. 确认分支为 `wall-aware-connectivity`。
4. 激活 `BLMRE_py38`。
5. 在GPU相关命令中确认宿主GPU可见。
6. 运行Ray时显式设置项目 `PYTHONPATH`。
7. 先检查工作区，保护用户已有改动。
8. 任何新指标先写小范围测试，再修改训练逻辑。
9. 不重新引入低带宽主线或硬性零断连目标，除非用户明确改变决定。

## 14. 交接信息维护规则

后续每完成一个阶段，应增量更新本文以下部分：

- “文件与生成物”
- “测试结果”
- “已知运行问题”
- “当前状态”
- “下一步任务”

不要每次重写整份文档，避免历史决策和精确路径在多次摘要中丢失。
