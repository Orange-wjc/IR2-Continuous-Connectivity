# 24进程试用与选择

## 最新并行试用：16对24进程，最终采用24（2026-10-16）

用户希望利用内存余量试用24。机器Ryzen 9 9900X为12物理核/24逻辑线程，不能按内存余量推断速度。先PAUSE监督，正式训练完整保存到{'IR2_FIFO': 5380, 'AB_learned': 5380}后正常退出；同一AB冻结权重、同一48张原easy训练图/逻辑种子、完整终止和运动/通信审计规则比较16与24，不与正式训练争用CPU。共96个工程CPU回合，SAC/开发/测试策略执行均0，不进入正式回放。

16进程用时210.975秒、采样吞吐13.651回合/分钟；24用时201.596秒、吞吐14.286。24/16吞吐比1.0465，耗时降低4.45%。最低可用RAM分别17.15/14.62GiB，新增系统换出0/23页。48回合经验哈希逐对一致，几何/通信审计通过。结论：实测采样提速且内存安全，保留24进程。这是一次固定模型CPU采样比较，含进程启动，不保证整轮CPU+GPU训练同比例提速，不与历史不同地图测速连乘。

资源判断说明：原探针把任意新增系统换出视为不安全，因此23页（92KiB）触发了保守否决；人工核查14.62GiB可用RAM及所有任务正常后，不据这点主机全局换出认定采样进程内存不足。改用“可用RAM>8GiB、整臂全局换出≤1MiB、任务及审计通过”的工程保护标准。原判定capacity_review_strict_zero_swap.json、原始日志和计数完整保留，修订依据写入capacity_review.json的memory_safety_review；没有重跑或改动经验、耗时和审计结果。

正式冻结source/driver.py的NUM_META_AGENT默认常量改为24，另外59文件不变；旧60文件归档source_versions/pre_meta_agent_24_20261016/。 两个训练锁空闲后才切换和启动唯一协调器。共享/初始检查点及原latest恢复边界SHA核验通过；完整恢复副本各组checkpoints/parallelism24_对应回合.pth保留。模型、目标网络、Adam、回放、RNG完整继续。入口默认值及pipeline workers/NUM_META_AGENT均为24，显式续训`--num-meta-agent 24`；省略则沿用保存配置。两组共同改变波次宽度，长期策略滞后会改变，不声称整段长训练逐位相同。

实际CPU+GPU复核通过：2026-10-16T21:00:53.779035+08:00，保存进度{'IR2_FIFO': 5404, 'AB_learned': 5404}，均完成新完整波次。最低可用RAM 13.42GiB、最高占用17.03GiB、显存峰值4.42GiB；新增系统换出306页（主机全局计数，旧swap不清空）。60源码身份/数据清单/恢复副本核验通过，实际采样进程数见runtime_processes.json，常驻监督healthy。有效批量128、微批量64、每完成回合8次SAC、每图2机器人不变。

自动评估仍关闭，不自动进入复杂阶段，不发飞书。4项参数/续训配置检查通过，监督既有正整数workers及按数量伸缩的内存保护无需改码。实时进度仍以coordinator/pipeline/status及真实进程为准。本条覆盖下方历史16配置。

证据：`research_work/native_workers_24_20261016/`；正式目录`checks/parallelism_24_20261016/`（capacity_review、switch_review、post_switch_review、runtime_memory_samples和runtime_processes）。

