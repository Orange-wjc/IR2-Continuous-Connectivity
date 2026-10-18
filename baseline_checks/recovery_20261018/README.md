# 原版十图基线：候选上限与主机内存修复

首次前台运行停止状态：有效18回合、无效1回合、120次SAC，派发编号23。正式模型及各Adam参数步数已读取核对，模型没有丢失。触发的是主机RAM可用量<4GiB的保护，GPU未报OOM，Ray退出后内存回到约22GiB可用。

原checkpoint和修复前工作树源码/状态备份位于experiments/ir2_baseline_easy10_20261018/before_memory_graph_repair_20261018，SHA在before_repair.json。该模型始终没有被测试或修复过程覆盖。本次实际运行的启动源码另已完整保存在source_before_quiet_logging_20261018；后续静音代码当时尚未被进程加载。

## 候选数量

原find_k_neighbor及custom版本选满K=30个全局邻居后，本地剩余预算为0，却继续进入循环再加入一个节点，得到31个候选。graph_failure_reproduced.json记录修复前两个函数均31。夹具还调用原Worker.get_observations，复现与截图相同的尺寸检查跳过分支。原截图时刻未保存图/RNG，不宣称完整复现同一历史轨迹。

修复为预算≤0立即返回；不是截断某个任意候选，也未提高K。修复后30个全局候选保持，mask尺寸一致。原合法0、5、29个全局邻居的返回索引与边集合顺序都不变。真实四个196步采样均有效。

## 回放存储

原每条transition含当前/下一张360×360 float32 mask，仅掩码10000条9.656GiB，总张量约9.880GiB，加上四个长采样、Ray传输复制/驻留导致主机RAM不足。原Ray日志已记录约2388MiB spill，不能把对象存储1GiB误当全部Ray/回放的内存上限。

baseline_replay.PackedMask按位+zlib无损存储，保留shape/dtype，非二值/负零等走原字节回退；robot仅改变两个mask列存储，driver仅在采样的两处stack前还原。其余13列、索引、随机采样、batch128、replay10000、热身2000、8批更新与原数学公式都保持。

验证证据：

- repairs_verified.json：全部15列真实原图样本逐值一致、dtype一致，原零预算错误复现与回归，10000个独立存储槽（无共享别名伪装）的真实样本回放峰值RSS增长426.68MiB；payload+张量约0.236GiB。随机密集二值mask载荷也仅约16.2KB/个，有保守上界。
- cuda_equivalence_verified.json：同样128样本连续8次原CUDA更新，所有网络、Adam和温度最终张量逐位相同，最大差0，峰值显存约17.66GiB。
- full_runtime_verified.json：四个三机器人原图采样均完整196步/588transition，随后满10000独立回放，主机可用RAM最低19.249GiB；父进程峰值RSS约1.065GiB，worker峰值约595～610MiB，耗时156秒。这里没有learner更新。
- saved_resume_verified.json：把真实120-update模型复制到临时目录，加载后只运行2步采样并SIGINT保存，更新保持120，网络/Adam/温度逐位不变，正式模型SHA不变，Ray已清理。

这些是工程边界/等价/资源检查，重复短轨迹用于容量压力和GPU对照，不代表多遍学习或收敛效果。只验证了列明范围，长期训练、其他地图/图边界仍需继续观察。

## 恢复

仍由用户前台运行：

```bash
conda activate ir2_train
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
IR2_RESUME=1 python driver.py
```

原版不存回放，恢复后重新积累2000经验；模型、Adam、温度、累计有效18/无效1/SAC120及窗口统计继承。正式结果不能与隔离验证产生的临时20回合数字混淆。终端已改每10有效回合汇总，详细日志和TensorBoard照常写。

没有自动启动正式训练，不删除旧研究STOP/PAUSE，不发飞书。graph和replay修复须作为公共基线工程修复对所有后续方法一致应用，不能计入方法创新或冒称完全未经修改的原作者代码。
