# 两机器人、单张简单地图训练入口

本流程只做当前单图学习检查。运行环境固定为 `/home/wyz/anaconda3/envs/ir2_train`，代码分支 `research/finite-contact-exploration`。所有动作仍由学习网络输出，不使用人工路线；不增加地图、不使用测试集。

## 本轮已完成的验收

从 seed=1 随机权重开始，经 1500 移动更新、300 通信头更新和实际 AB 学习，再使用小 Actor 步长续训，两个连续固定预算终点均达到门槛：

| 累计 SAC／head | 训练图 greedy | 十次 sampled | 两张开发验证 |
| --- | --- | --- | --- |
| 3500／300 | 成功 160.204 s | 9/10，seed505 | 2/2 |
| 4000／300 | 成功 102.624 s | 9/10，seed606 | 0/2 |

Actor 与 Q 在后两段确实更新 1000 次，头／上下文逐张量不变，数值与协议检查通过，33 项行为检查通过。这是本轮两机单图可以学习并真实续训的证据；最终开发验证 0/2，不能称已经解决泛化或证明通信创新更优。

可直接评估的模型：`experiments/actor_rate_20261011_AB_seed1/final.pth`；继续训练用同目录 `latest.pth`。验收为 `experiments/actor_rate_20261011_AB_seed1_verify{505,606}/acceptance.json`，审计和图为 `experiments/actor_rate_20261011_report/`。具体失败条件与同噪声控制见 [Actor 步长记录](ACTOR_RATE_EXPERIMENT.md)。

## 从随机权重启动

选择新的输出目录：

```bash
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_single_map.py \
  --profile actor_decay --seed 1 --output experiments/my_single_map_run_01
```

仅查看即将执行的命令，不创建实验、不启动训练：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_single_map.py \
  --profile actor_decay --seed 1 --output experiments/my_single_map_run_01 --dry-run
```

编排入口已做命令干跑与编译检查；各阶段命令分别实际使用 GPU 运行过，具体结果和已知失败见 [实验记录](ACTOR_RATE_EXPERIMENT.md)。编排入口尚未另外重复一整套训练，不能把干跑当作新实验结果。

## 实际阶段与输出

1. `movement_A/`：单图从零学习，1500 次 SAC；自身成功经验辅助 weight=1，union 探索奖励，terminal 截止。
2. `content/`：冻结移动网络，300 次监督更新通信评分头，标签只来自训练图。
3. `finetune_AB/`：接入内容选择与有限窗口会合，按 2000、2500、3000 累计 SAC 更新分段训练，保留每个固定预算模型和完整状态。
4. `verify_3000/`：greedy 与十次 sampled 复核，动作噪声 seed303；开发验证两张另行报告。
5. `actor_decay_AB/`：从完整 3000 状态建立独立实验目录，**只把 Actor 学习率降至 1e-5**，Q 和温度仍为 1e-4；保留优化器动量、训练回放、成功经验和 RNG。继续实际学习至 3500、4000；不是冻结 Actor，也没有从评估中提取训练动作。
6. `verify_3500/`、`verify_4000/`：使用事前固定的动作噪声 seed505／606，各十次采样，模型和计数只读核对。

`--max-updates` 均指累计 SAC 更新，不是新增次数。1500→4000 总计 4000 次 SAC，另外 300 次通信头监督；没有使用复杂地图课程。后半程建立的是实验目录，代码仍在同一 Git 开发分支。

整体状态写在 `pipeline.json`，每个步骤的 stdout／错误写在相应 `.log`。各阶段另外保留配置、地图 SHA、源码、依赖、训练曲线、通信事件和 checkpoint。若达到预算，`final.pth` 是该预算终点，`latest.pth` 是包含回放及 RNG 的可恢复状态；`at_<更新数>.pth` 和 `at_<更新数>_latest.pth` 保留分段终点。

## 如何判断结果

单图工程门槛为每台机器人局部覆盖≥99%的 greedy 成功，且同一终点十次 sampled 至少 9 次成功，继续实际更新后再次通过。两张开发验证的完成率独立报告；验证失败不会被隐藏，也不能用单图成功宣称跨地图收敛。

脚本不会因最早 AB 终点失败就拿 best 替换 final。阶段未达到预算、评估不完整或最终门槛未通过会记录并返回非零退出码；完整数据仍保留。十次动作采样不是十个训练种子，单个通过序列也不是统计收敛证明。

既有目录不会覆盖。中断后按 [README 的 resume 说明](README.md) 从当前阶段 `latest.pth` 继续，保留相同配置；不要重新从第一步覆盖目录。要改 Actor 速率同时保留全部训练状态，用 `finite_contact_fork.py` 建立新实验，不能把轻量 best 当作完整恢复点。

`--profile union`／`--profile local` 用于复现已有诊断条件，它们的某些终点已记录采样失败，不能因 greedy 成功就当作稳定方案。相同的成功辅助、学习率安排与训练预算应共同用于后续各消融组；这些训练措施不属于有限通信窗口与内容选择的研究创新。
