# 单适应种子稳定性试验的小型证据

六组固定终点均完成：seed=2，pooled／map_tail × FIFO／规则／学习。每组在自己的原完整状态上新增2000次SAC，累计12895；通信头5000且冻结。开发成功数分别为18／18／18与19／20／20，学习map_tail平均120.694秒。测试策略执行为0。

`summary.json` 是原始汇总的完整复制，包含训练监测退步；`failure_analysis.json` 保留全部开发失败及配对得失。`pair_integrity.json` 核对执行源码、成对首次轨迹、地图调度、所有Actor／双Q／温度Adam参数的真实新增步数。`manifest.json` 给出复制文件的SHA256；`plan.json` 和事前设计保留实验范围及初始化中断记录。

`source_identity_before.json` 对应首次初始化中断前；`source_identity_resume.json` 对应增加初始化归档恢复后实际执行版本；`source_identity_after.json` 与后者相同。中断发生在任何新训练地图／SAC更新之前，旧目录归档，没有重做更新。

训练诊断与frontier探测是描述性证据，不能当作失败因果证明。前10张训练图监测也不是独立泛化样本；成对初始监测是相同状态重复执行。尚需另外两个种子同预算复现，不默认启用map_tail。

文中本机路径指向未纳入版本控制的模型、原始地图及日志；本目录不能单独代替完整模型与原始运行数据。完整说明见[实验记录](../../STABILITY_MULTIMAP_EXPERIMENT.md)。
