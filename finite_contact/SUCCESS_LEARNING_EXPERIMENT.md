# 第五轮：保留成功经验，验证真实训练稳定性

日期：2026-10-11。环境 ir2_train；分支 research/finite-contact-exploration；两机器人、单张简单训练图、99% 每机局部覆盖、256 秒期限不变。所有旧结果保留。

## 问题与修改

第四轮发现源 AB 在训练图能完成，三个 500 次联合微调终点反而失败。deadline 运行的回放中 A 有 4 个、AB 有 5 个完整成功回合，但普通 SAC 未特别保留这些路线；截止转移只占更新批次样本约 0.41%／0.43%。这些事实支持检查目标回报稀疏与已获得动作能力丢失，不证明单一原因。

加入自身成功经历的辅助学习：只在 movement/finetune 的训练回合真正成功后保存其本地观测与实际动作，最多保留最快的 3 个完整团队回合；验证、内容头训练、失败、预算中断都不能加入。普通 SAC 的 Q 目标和回放采样不变，Actor 损失增加 `weight * mean(-log pi(a_success|o_success))`。这仍是学习网络选动作，没有人工路线或真实远端状态输入。辅助权重本轮固定为 1，默认仍为 0 以保留旧行为。内容头、上下文继续冻结。

完整 latest 保存成功回合和累计辅助更新次数，resume 恢复后继续训练；轻量 final/best 只用于初始化或评估。另新增 best_train_success／best_validation_success：仅完整评估集每张图都成功才可保存，先比平均完成时间，再比覆盖积分；旧的积分 best 保留，不能冒充成功验收。

## 启动前预算与验收

- A/FIFO 与 AB/learned 从相同 content_20261011_A/final.pth（SAC=1500、head=300）开始，新优化器与回放；不导入之前失败运行的数据。
- terminal 目标、alpha 学习率 1e-4、势函数关闭、sender 上限与 service 顺序；仅新增 success_imitation_weight=1。
- 首对各新增 500 次至 2000，最多 30 回合／12 分钟，每 5 回合评估训练图和同两张开发验证图，固定预算终点是主要结果。
- 训练正常的最低门槛：最终 greedy 训练图每机≥99%，同模型 10 次 sampled 至少 9 次成功；继续真实训练后仍可达到该门槛，网络权重确实更新，数值稳定、传输预算正确，resume 可复现。
- 十次采样预先固定动作噪声 seed=101，继续训练后的复核改用 seed=202。地图生成不依赖该 seed，仍为同一张训练图；它们是独立于训练噪声的评估序列，不是独立训练种子。
- 若通过，进一步用新随机初始化跑完整分阶段训练，确认不是仅靠保留旧模型。单图正常训练与未见图泛化分开判断，测试集继续不用，不扩大地图。
- 若未通过，不用 best 或冻结模型替代失败 final；继续根据失败状态调整，不宣布问题解决。

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_train.py \
  --stage finetune --variant AB --content learned \
  --initialize-from experiments/content_20261011_A/final.pth \
  --content-score-bound sender --delivery-queue-order service --horizon-target terminal \
  --alpha-learning-rate 0.0001 --frontier-shaping-scale 0 --success-imitation-weight 1 \
  --max-updates 2000 --episodes 30 --max-hours 0.2 \
  --eval-every 5 --eval-maps 2 --eval-train --final-eval --device cuda \
  --run-dir experiments/success_20261011_AB_learned
```

A 对照替换 variant=A、content=fifo，目录 success_20261011_A_fifo。

## 独立初始化的端到端复核（启动前固定）

与已有权重的配对诊断并行，从 seed=1 随机初始化 A 移动 Actor／Q 开始，使用同一单图、成功经验辅助 weight=1。初始移动预算固定为 1500 次更新、最多 100 回合／24 分钟，每 10 回合评估，目录 fresh_success_20261011_A_seed1。成功后再以该移动模型训练 300 次通信头，随后 AB 进行真实联合微调与继续训练，所有阶段分别记录更新与输出。尚未通过时不跳过失败阶段。该链路不与旧 seed=0 的 500 次微调组做创新性能排名。

## 首对固定预算终点（已完成）

以 `at_2000.pth` 保留首轮 final，以 `at_2000_latest.pth` 保留其完整状态，之后在原目录恢复到 2500 次更新；不覆盖首轮证据。独立只读验收的动作噪声 seed=101，逐次记录与权重不变检查均完成。

| 真实训练终点 | SAC／head | 训练图 greedy | sampled | 两张开发验证 greedy |
| --- | --- | --- | --- | --- |
| A／FIFO | 2000／300 | 成功，136.043 s | 9/10 | 0/2 |
| AB／learned | 2000／300 | 成功，104.267 s | 9/10 | 1/2 |

两组均新增 500 次真实 SAC 和 500 次辅助更新，而非冻结导出或按 best 选模。单图首轮达到最低门槛；继续训练后的稳定性另行验收。此结果不能证明 AB 在新地图上更优，也不能替代完整消融和多训练种子统计。

验收证据：`experiments/success_20261011_{A_fifo,AB_learned}_verify101/acceptance.json`，相应目录保留 greedy／sampled／validation 的轨迹、最终地图、通信事件与源码快照。

新 seed=1 的初始 greedy 最低单机覆盖 30.651%，1500 次真实更新后的 final greedy 在 100.791 s 成功。后续通信头和 AB 链路继续单独记录；这一训练种子不等于十次动作采样。

## 续训入口修复与代码验证

实际 A/FIFO 续训发现 CLI 默认值将保存的 FIFO 改成 learned，从而触发配置不一致拒绝。已修复为：只有新阶段初始化应用默认内容方式，resume 不隐式改变已有方式；content 阶段同样修正。30 项行为检查全部通过，其中新增真实 CLI 恢复 finetune 与 content，并核对配置逐项相同、更新计数实际增加。测试日志 `/tmp/ir2_success_resume_tests.log`。

## 续训与分阶段权重审计（已完成）

首对恢复完整状态后各再更新 500 次，累计 SAC=2500、head=300、辅助更新=1000。A 最终 greedy 成功 101.640 s，AB 成功 70.731 s；新的十次采样 seed=202 另存验收目录。两组新增更新日志恰为 1501～2500 连续计数，全部数值有限，Actor、双 Q、目标 Q 均改变，内容头与上下文逐张量不变。完整成功的训练回合 A=13/16、AB=22/22；预算中途截止回合另行记录，不冒充任务失败／成功。

新 seed=1 移动阶段实际完成 1500 次 SAC、1162 次辅助更新，完整训练回合成功 10/21。该中间 final 的 greedy 成功，十次 sampled 只有 6/10，开发验证 0/2，**未达到稳定单图验收**。保留这一失败结果，后续仍需检查整个 AB 训练链路的最终模型，不能用中途 greedy 成功替代最终验收。

新链路内容头完成 300 次监督更新；源 Actor、双 Q、目标 Q、温度逐张量不变，上下文严格复制源 Actor，只有通信头学习。其六个完整训练轨迹均来自同一训练图，没有使用验证标签。审计脚本和当前记录为 `experiments/success_20261011_report/audit.py`、`training_audit.json`、`fresh_stages_audit.json`；图为 `paired_curves/learning_curves.{png,pdf}`。

首对续训的只读复核（seed=202）得到：A 的 sampled 降为 **7/10，未通过稳定性门槛**；AB 的 sampled 为 **10/10，通过**，两张开发验证 greedy 为 2/2。AB 在两个连续真实训练终点达到门槛，A 没有。该区别应保留，不能只列双方 greedy 成功或用旧 best 替换 A 的失败 final。

## 新初始化 AB 的失败终点与诊断

新 seed=1 链路在 SAC=2000 的 final greedy 失败，局部覆盖为 67.450%／99.998%，union=100%。保存为 `fresh_success_20261011_AB_seed1/at_2000.pth` 及完整 `at_2000_latest.pth`，在原目录恢复到 2500；首轮失败验收仍保留，不用周期性成功或旧 best 替换。

其固定轨迹严格回放全部 83 次决策的位置、覆盖、时间和交付量均相同，argmax 全部匹配。联合覆盖在 35.887 s 达到门槛，最终机器人 0 仍有 131 个局部前沿、最大 utility=40，机器人 1 则已完成；显式等待仅 6.58／14.95 s。因而不是单纯 WAIT 过多或物理不可达，完成一台机器人的地图并未保证所有机器人完成。原 union 探索奖励在联合覆盖完成后不再奖励本机补扫，只剩实际交付和稀疏团队终奖；这可能增加后半程学习难度，当前并未据此改动奖励或放宽完成条件。

只读梯度检查在两个 2000 次完整检查点各抽十批真实回放和成功示例。旧 seed=0 的 SAC／辅助梯度平均范数为 3.30／28.15，新 seed=1 为 3.32／37.56；两者都有 5 批原始梯度方向夹角为负，但加入 weight=1 后，十批总梯度与辅助梯度的内积均为正。检查不更新权重，也没有计算 Adam 预条件后的实际步长。因此这些样本不支持“辅助太弱、被 SAC 完全压倒”的猜测，不据此盲目增大权重。文件为 `gradient_check.py`、`gradient_check.json`、`fresh_2000_diagnosis/`。
