# 第七轮：学会后的 Actor 步长与真实续训稳定性

日期：2026-10-11。两机、同一训练图、ir2_train，测试集不使用。

## 直接依据与单一改动

新 seed=1 从零开始的 union 奖励链路在 SAC=3000 的 final，greedy 成功 168.908 s、十次 sampled=10/10。但相同学习率真实续训 500 次后，3500 的 greedy 虽成功 192.478 s，sampled 降至 7/10。新 local_mean 条件在 SAC=2000 的 final 虽 greedy 成功 61.480 s，sampled 仍只有 5/10。因此继续加奖励或仅凭最快 greedy 选模不能解决稳定性问题。

从已通过的 **完整 at_3000_latest.pth** 新建独立实验，保留 Actor、双 Q、目标 Q、内容头、上下文、Adam 动量、普通回放、成功缓存、全部 RNG 与累计计数。只把 Actor 的学习率从 1e-4 降至 **1e-5**，Q 与温度仍为 1e-4。没有冻结 Actor，也不导入评估数据或重置训练经验。旧 3500 的失败运行保留。此项是能力获得后的共同微调安排，不能计作通信创新。

新增 `policy_learning_rate`；新配置缺省 None 时沿用公共 learning_rate，旧检查点迁移保留其历史 Actor 速率。`finite_contact_fork.py` 只能对完整 finetune 状态建立新目录并修改 Actor LR，不能改变奖励／物理设置或覆盖源运行。之后仍用严格的 `--resume` 恢复；fork_source.pth 永久保留源状态与变更记录。新实验的墙钟计时单独开始，学习计数继续累计。

## 启动前固定的预算与验收

- 源为 `fresh_success_20261011_AB_seed1/at_3000_latest.pth`，SAC=3000、head=300、success weight=1、union 奖励；其他配置不变。
- 首次真实续训 500 次至 3500，完整状态再续训 500 次至 4000；每段最多 100 回合／18 分钟，每 5 回合评估训练图和两张开发验证。
- 第一个新终点十次 sampled 使用事前指定 seed=505，第二个使用 seed=606。固定预算 final 是主要结果，不用 best 替代。
- 两个新终点均需 greedy 每机≥99%，十次 sampled≥9/10；核对 Actor 与 Q 确实改变、头和上下文冻结、数值有限、协议守恒与完整恢复。开发验证结果独立报告，不替代单图门槛。
- 如果失败，继续保留失败并诊断，不通过切换动作噪声、冻结 Actor 或回退 best 假装正常训练。

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_fork.py \
  --checkpoint experiments/fresh_success_20261011_AB_seed1/at_3000_latest.pth \
  --output experiments/actor_rate_20261011_AB_seed1 --policy-learning-rate 0.00001

/home/wyz/anaconda3/envs/ir2_train/bin/python finite_contact_train.py \
  --stage finetune --resume experiments/actor_rate_20261011_AB_seed1/fork_source.pth \
  --max-updates 3500 --episodes 100 --max-hours 0.3 \
  --eval-every 5 --eval-maps 2 --eval-train --final-eval --device cuda
```

## 行为验证

新增检查确认：完整 fork 的权重、回放、成功回合与 RNG 不变；相同回放和随机状态下，下一步 Q／目标 Q／温度与原速率逐张量一致，Actor 真实变化约为原步长的十分之一；CLI 能恢复新速率和训练缓存，学习计数实际增加。测试日志 `/tmp/ir2_actor_rate_tests.log`。

33 项行为检查全部通过（19.884 s）。实际源 checkpoint 的全量权重、所有优化器的非 LR 状态、回放、成功轨迹、四类 RNG 与 episode 计数逐项比对通过，记录为 `actor_rate_20261011_AB_seed1/fork_audit.json`。首个小步长终点 greedy 成功 160.204 s；只读采样和下一段真实训练继续执行，尚不能据此判定稳定。

## 同采样设置的补充控制

为分开检查噪声波动，3000 次源模型额外使用 seed=404 复测，结果为 **6/10**，而原速率 3500 次模型在 seed=404 为 **7/10**。因此之前不同序列下的 10/10（seed303）→7/10（seed404）不能单独证明续训退化，更不能断言“大步长是唯一原因”。小步长仍按事前固定的预算和新噪声完成检验，但须作为工程假设报告。控制输出为 `fresh_success_20261011_AB_seed1_verify404_control/acceptance.json`，无网络或训练缓存更新。原源模型的采样稳定性尚不充分，不能称其已经统计收敛。

## 首个小步长终点（已完成）

固定 SAC=3500、head=300，Actor 真正更新 500 次、Q 同步学习，未冻结部署策略。训练图 greedy 成功 160.204 s；seed=505 的十次 sampled 为 **9/10**，开发验证 greedy 为 **2/2**，只读权重与学习计数不变，单图门槛通过。输出 `actor_rate_20261011_AB_seed1_verify505/acceptance.json`。

按同配置续训到 SAC=4000 后，最终 greedy 成功 102.624 s；预先指定 seed=606 的十次采样另行验收，不能仅凭这个 greedy 就宣布续训稳定。

## 最终续训验收（已完成）

4000 次终点的 seed=606 十次 sampled 为 **9/10**，只读权重和计数检查通过，开发验证 **0/2**。两个连续小步长终点均通过本轮单图工程门槛。最终模型与完整状态为 `actor_rate_20261011_AB_seed1/final.pth`／`latest.pth`，3500／4000 的轻量和完整分段模型保留为 at_3500／at_4000 系列。

后两段恰好新增 1000 次 SAC，Actor 权重变化 L2=0.623827，双 Q=3.471062，目标 Q=3.410748；头和上下文为 0。完整训练回合成功 10/14，中途更新预算截止单独记录；训练中策略持续变化，不能把这个比例等同于固定 final 的采样成功率。所有更新数值有限、协议字节守恒，成功缓存只含自身真实完成的训练回合。完整 fork 与恢复检查通过。

结论为：**当前两机单图流程可以从随机权重学习并在两段真实续训后保持本轮门槛**。十次采样与单训练种子不证明统计收敛；最终未见图 0/2，尚未解决泛化，也未证明小步长单独造成改善或通信创新优势。union/local_mean 的失败终点和同噪声源模型控制全部保留，不用 best 替换失败。

复现入口与阶段命令见 [RUN_SINGLE_MAP.md](RUN_SINGLE_MAP.md)。`finite_contact_single_map.py --profile actor_decay` 编排这些实际分别运行过的阶段；编排本身另做干跑和编译检查，没有把命令干跑计作新训练。最终审计为 `experiments/actor_rate_20261011_report/final_audit.json`，阶段曲线及最终路线图为该目录 `full_pipeline/`。

物理复核覆盖两个终点的全部 26 次评估：共 3000 段移动，穿墙数为 0；覆盖率从真值与各机局部地图重算后与日志完全一致，连接时长对应的通信总量上限与速度对应的总路程上限检查通过。记录为 `experiments/actor_rate_20261011_report/physical_audit.json`。
