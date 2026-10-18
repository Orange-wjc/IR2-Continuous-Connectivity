# 完整独立三种子实验：已完成并复核

训练与自动评估于北京时间2026年10月15日14:15完成。用户随后授权结果分析，全部15组已从原始记录复核；当前没有启动下一轮。飞书主动通知保持关闭。

主要入口：[完整分析与限制](HUMAN_REVIEW.md)、[论文结论草稿](PAPER_CONCLUSIONS.md)、[下一步建议（未启动）](NEXT_STEPS.md)。

| 方法 | 三种子平均任务秒 ± 种子样本标准差 | 合计成功 |
| --- | ---: | ---: |
| B0_fifo | 130.133 ± 33.183 | 56/60 |
| A_fifo | 108.928 ± 17.098 | 60/60 |
| B_learned | 95.810 ± 11.513 | 60/60 |
| AB_learned | 75.410 ± 6.271 | 60/60 |
| AB_uncoupled | 83.795 ± 7.010 | 60/60 |

完整AB相对基线缩短42.05%，相对去耦合缩短10.01%；三个种子均值方向一致，但逐图不一致，seed101耦合增益对最大收益图敏感。A单独收益随种子变化，2×2交互符号不一致，不支持稳定超加性结论。均为重复使用的20张开发图，三个独立训练种子，测试策略执行0；不能将60条种子×地图记录视为60个独立种子。失败计256秒。

固定计划见[PLAN.md](PLAN.md)。每种子从随机初始化做中性B0/FIFO预训练9000次SAC、训练消息头5000步，再配对训练五组各新增20000次SAC、固定终点29000；不能与旧A/FIFO起点协议混合排名。

证据文件：

- [原自动汇总](pipeline_summary.json)、[原自动报告](pipeline_RESULTS_ANALYSIS.md)、[最终流水线状态](pipeline_final.json)。原件仍在experiments/independent_seeds_20261013/report，不覆盖。
- [补充分析JSON](supplemental_analysis.json)、[逐图指标CSV](endpoint_metrics.csv)、[逐种子终点CSV](endpoint_by_seed.csv)、[全部配对差CSV](paired_differences.csv)、[训练分块CSV](training_blocks.csv)。
- [只读复核](final_readonly_review.json)、[输入SHA256](analysis_input_hashes.json)、[复现入口](analysis_reproduction.json)、manifest.json。完整模型与大体积轨迹仍在原实验目录，本目录为小型结果包。
- [各种子终点图](independent_seed_endpoints.pdf)、[逐图配对图](paired_map_savings.pdf)、[机制图](mechanism_measures.pdf)、[训练曲线](training_progress.pdf)、[全部基线失败案例](baseline_failure_cases.pdf)，均另有PNG。
- [几何审计修复记录](PHYSICAL_AUDIT_RECOVERY_20261014.md)：保留原源码快照，活动身份为source_identity_audit_resume_20261014.json。本次复核统一重审600条共同阶段与300条开发轨迹，其他适应轨迹使用已完成的流水线全量审计。

合计15900条保留轨迹、1349336段移动，进入障碍格内部0，边界/角点接触1008。几何为连续点对像素内部，不等于有限半径机器人安全性。训练进程峰值最高约3.04GiB，未再OOM；曾因几何审计误判停机，修复并补审后完成。
