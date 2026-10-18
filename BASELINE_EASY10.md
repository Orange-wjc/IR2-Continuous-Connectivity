# 原版IR²：10张原easy地图重复训练基线

当前分支 `baseline/ir2-easy10`，从 main 的 `1ef6d224ae18503f4763c3cec20e05a737f95783` 创建。目标是先验证原版训练链路和固定10图学习能力，之后再讨论接入改进模块。它不是此前有限容量框架的10图实验，也不等于完整规模论文复现。用户首次前台训练已在18个有效回合/120次SAC更新后因RAM保护退出；本节末记录两项工程修复。修复后仍等待用户前台续训，不后台自动启动。

## 开始训练

```bash
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
conda activate ir2_train
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
IR2_RESUME=1 python -u driver.py 2>&1 | tee -a experiments/ir2_baseline_easy10_20261018/foreground.log
```

入口仍为原项目 `driver.py`。当前已有检查点，直接续训可用 `IR2_RESUME=1 python driver.py`；新运行从零训练时才直接 `python driver.py`。前面的线程设置用于限制CPU数学库并行。不要加旧研究入口的 `--finite-contact`。默认4个原Ray采样worker、单GPU学习；无后台启动服务或自动评估。

在同一项目目录只检查地图和配置、不启动Ray或训练：

```bash
python driver.py --check-config
```

默认输出名称 `ir2_baseline_easy10_20261018`；已有检查点时禁止误覆盖。需要完全独立的新运行，可设置 `IR2_RUN_NAME=ir2_baseline_easy10_run2`；每个运行有独立的地图清单、模型和日志。新运行的终端日志应写到已存在的目录，或直接用 `python -u driver.py`。

## 原版保留与设备适配

- `env.py`、`model.py`、`multi_robot_worker.py`、`sensor.py`、`ss_realistic_model.py`逐文件SHA与main相同。`graph_generator.py`修复零剩余邻居预算仍追加第31候选的边界，`robot.py`仅无损压缩保存掩码；这些是显式记录的基线工程修复，后续所有对照共同使用，不计为方法创新。
- 原SAC损失、Actor/Q/温度更新顺序、梯度裁剪、固定目标熵、硬目标更新与每次8批更新保持；GPU更新数学代码块与main逐字一致，仅增加统计计数。没有加入SIL、消息头、WAIT、有限字节传输或新的奖励。
- 按用户最新要求，本轮easy固定3台机器人（NUM_ROBOTS_MIN=NUM_ROBOTS_MAX=3，原版easy默认为随机3～5台）。保留6维输入、128维嵌入、196步、原信号强度通信与原完成规则。后续改进组也须使用相同三机设置进行公平比较。
- 学习率1e-5、batch128、回放10000、开始学习前至少2000条经验、gamma .995及原目标更新方式不变。回放未满时前台 `sac_updates=0` 是收集阶段，不表示学习故障。
- 两份360×360邻接掩码在worker保存时按位无损压缩，采样时还原原始浮点张量再stack；非二值数据使用原字节无损回退，不量化、不裁剪、不改回放索引或有效batch。原版两份稠密掩码10000条约9.66GiB，本修复已检查真实样本的逐值一致性和全部原SAC/Adam状态等价。
- 本机改为NUM_GPU=1、NUM_META_AGENT默认4，worker Torch1线程、learner2线程、Ray对象存储1GiB，关闭非必要训练GIF。可以在启动时显式 `IR2_AGENTS=8` 改采样数，但当前仅默认4的调度与短检查通过，不宣称8/12/24个worker在长训练时都内存安全。
- 启动要求主机至少6GiB可用RAM，循环中可用RAM低于4GiB则保存并退出；不自动改变学习算法或无限重启。较长训练会逐步填满原版稠密回放，短检查不代表已验证全程峰值。

## 固定10图与计数

选原 `DungeonMaps/train/easy` 反向文件名排序的前10张：`img_9999.png`至`img_9990.png`。不按成绩挑图、不生成、不缩放、不改原PNG。

`baseline_support.prepare_maps` 在 `experiments/<运行名>/maps` 创建仅含这10张的原文件符号链接，并写 `map_manifest.json` 保存源路径及SHA。原Env仍直接执行 `episode_number % map_count`，因此任务1读`img_9998.png`，任务9读`img_9990.png`，任务10读`img_9999.png`，任务11再次读`img_9998.png`。只限制池大小，不限制总回合，持续循环直到用户Ctrl+C。

终端每10个有效回合输出一次汇总；无效回合警告、报错和模型保存提示保留。逐回合JSON继续写progress.jsonl，TensorBoard逐回合更新；worker的启动/完成等详细输出写到experiments同名目录下worker_logs/worker_*.log。运行中的进程保持已加载版本，此日志调整在下次启动时生效，不为更新日志自动重启或中断训练。

原Ray异步派发会使任务完成顺序不同；不要把任务编号或 `completed/10` 直接解释为所有图恰好训练了相同遍数。前台JSON含：

- `episode`：原项目分配的任务编号；`completed`：已返回的有效回合数。
- `task_success`：是否按原环境完成探索，与“有效采样返回”分开。
- `invalid`：原代码因图/A*等问题丢弃的采样回合数，单独报告，不能隐瞒。
- `recent_success_percent`、`window_size`：最近至多100个有效回合的成功率与实际样本数，初期不足100按已有样本计。
- `replay`、`sac_updates`：经验量与实际SAC更新次数；达到2000条经验后开始每轮8次更新。
- `min_map_visits`：10图中最少被有效训练的次数；逐图访问和最近10次成功率另见status.json及TensorBoard。

原 `Perf/Success Rate` 统计仍保留；新增 `Progress/Success Rate 100 Percent` 是直接按有效完成回合统计的0～100百分比。图/A*无效回合单独计数，主成功率分母不含这些计算失败，沿用原采样接受规则。原测试集不用于这轮训练或检查，没有自动插入开发评估。

## TensorBoard

另开终端：

```bash
conda activate ir2_train
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
tensorboard --logdir train/ir2_baseline_easy10_20261018 --host 127.0.0.1 --port 6010
```

浏览器 `http://127.0.0.1:6010`，选择 `Progress/Success Rate 100 Percent`、`Progress/SAC Updates`、`Maps/`逐图成功率及原 `Losses/`。建议Smoothing=0先看窗口原值。机器人数量、连通图和动作带随机性，某一遍10/10不等于稳定收敛。

## 保存与续训

Ctrl+C发送停止请求，主进程完成当前8批更新块（若正在学习）后原子保存检查点，关闭自己的Ray worker和日志。原每32个派发编号保存也保留。模型位于 `model/ir2_baseline_easy10_20261018/checkpoint.pth`；进度在 `experiments/ir2_baseline_easy10_20261018/status.json`、progress.jsonl、setup.json；曲线在train同名目录。

继续同一个模型：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
IR2_RESUME=1 python -u driver.py 2>&1 | tee -a experiments/ir2_baseline_easy10_20261018/foreground.log
```

续训沿用原版检查点机制：恢复模型、优化器和温度，额外恢复累计计数/窗口/逐图统计；**原回放和在途采样不存入检查点**，重启后需重新收集2000条经验，目标网络按原版从当前Q初始化。因此这不是精确全状态断点恢复，统计与比较需记录重启次数。长期正式协议后续再明确，当前不悄悄引入另一套完整恢复训练器。

## 已验证与尚未验证

`baseline_checks/main_identity.json`保存main源码SHA；config_verified.json保存准备配置；runtime_verified.json记录真实原PNG解码/循环、两个短2步Ray采样、原batch128连续8次CUDA更新、每个优化器真实8步、有限损失和参数变化，显存峰值约17.7GiB。GPU检查重复短轨迹只用于工程内存/梯度检查，不证明学习效果。

`foreground_verified.json`记录独立临时目录的真实driver前台SIGINT保存、Ray退出、检查点恢复和TensorBoard事件；原回放门槛不变，短采样不足2000经验，所以SAC=0。临时模型已移除，正式运行没有模型或progress.jsonl。检查脚本在baseline_checks，可复核，不作为训练结果。

上述运行检查完成于最初的3～5台配置；后续按用户要求固定3台，仅重新核验配置与地图，未重跑采样或训练。最新config_verified.json及正式启动时的setup.json记录固定三机设置。

尚未验证：这10张图多遍训练后的学习曲线、充分收敛表现、其他地图泛化和全量数据的资源峰值。用户前台启动后再观察，不提前宣称原版复现成功或保证99%。旧研究实验及STOP保持。

## 2026-10-18：候选尺寸与主机RAM修复

首次运行记录：18有效回合、1无效回合、120次SAC，派发编号23，保存checkpoint.pth后主机可用RAM低于4GiB触发受控退出，Ray已清理。截图中的候选错误发生于任务21/机器人2/步46。原日志未保留该时刻的图/RNG，不能精确重放同一轨迹；隔离构图夹具复现了同一错误分支。

根因与修复：

1. `find_k_neighbor`和`find_k_neighbor_custom`选满30个全局邻居后本地预算为0，仍进入for循环加入一个节点，产生31候选而掩码固定30。现在预算≤0立即返回，不随意截断候选，也不放大K。原本合法的0/5/29个全局邻居场景逐值保持。
2. 原Robot保存当前/下一观测的稠密浮点edge_mask，两份每条约1MiB；10000条掩码9.66GiB，再叠加4个长回合、Ray序列化/对象传输和其他张量，导致主机RAM余量耗尽。不是GPU显存报错，也不是系统强杀。`baseline_replay.py`用位打包+zlib无损保存，driver仅在原采样位置解码；完整原SAC数学块保持。

原失败检查点和来源资料已复制并记录SHA到experiments同名目录/before_memory_graph_repair_20261018，正式checkpoint保持原字节。检查结果在baseline_checks/recovery_20261018：graph_failure_reproduced.json、repairs_verified.json、cuda_equivalence_verified.json及后续运行/续载检查。10000条独立存储的真实样本回放内存增长约427MiB；密集随机二值掩码也有≤16.3KB/个的载荷上界。GPU比较同batch128×8更新后全部网络、Adam与温度张量逐位相同，最大误差0。短采样重复用于工程压力/等价检查，不是学习效果。

内存保护保留，不降batch、不缩回放、不调整学习率/奖励。`--check-config`现在写独立config_check.json，避免覆盖历史setup；真正启动前保留旧setup历史，每条进度记录RAM；退出原因另写run_state.json。Ctrl+C和自动资源退出仍保存模型、关闭自己的Ray。

续训命令为`IR2_RESUME=1 python driver.py`。回放仍按原版不存盘，恢复后重新积累2000经验；累计120更新不会清零。当前修复的验证范围是已复现边界和有限工程运行，长期曲线与其他未知边界仍需正式续训观察。
