# 原easy累计1000张续训（2026-10-18）

用户同意下一步从500扩大到累计1000张原easy地图，作为进入原训练集全量训练前的中途评估节点。每方法新增500张/10000 SAC，累计1000张/29000 SAC（9000共享移动+20000适应）。不从头重训，不把1000当最终固定数据规模，不自动超过1000或进入复杂图/原测试，不发飞书。

正式输出`experiments/native_easy_1000_20261018_seed101`，父实验`experiments/native_easy_500_20261017_seed101`。两组各自继承500图的完整网络、Q/目标、Adam/温度、头/上下文、RNG、12000条压缩回放和成功示例（基线228图、AB245图）。原始字节检查点备份inherited/，转换后initial/latest只修改清单路径、train_maps500→1000、success_episode_limit500→1000，保留状态语义SHA核对完全相同。共享9000和头5000不重跑。Actor/Q学习率、损失、奖励、时限、采样节奏、图/模型结构均不改；13关键文件与父执行快照逐字一致。该轮为同一seed101的延续，不是新独立种子。

地图取现有原easy训练清单前1000文件，前500与父清单完全一致，新episode501→map_index500，连续到episode1000/map_index999；不重新生成或按开发成绩筛图。原100开发图不变，与训练几何组隔离，原PNG逐个SHA核验。原测试0使用。固定18小时单次墙钟保护，无周期评估，只在累计1000端点各做一次100图开发贪心评估（共200），生成RESULTS.md、全部逐图、comparison.json、round_progress.json、PNG/SVG后停止。

启动前核对：基线100→500成功53→49，成功转失败6图、失败转成功2图；AB55→57，成功转失败1图、失败转成功3图。500图端点所有失败均196决策上限停止，基线4图/AB9图联合覆盖≥99%但局部未同步达标；其余还有探索不足。未见继承、预算、冻结、运动/通信审计异常。保留这些失败，不重置基线、不提高决策上限、不剔除难图；终点现象不能定位因果。证据FAILURE_REVIEW.md/failure_review.json，实际新策略执行及梯度更新均0。

内存：30.5GiB RAM/RTX3090；500图检查点375.5/317.8MiB，成功缓存压缩载荷266.1/210.9MiB。以当前每条平均外推1000个成功图的压缩载荷约1.14/0.84GiB（不含Python/图增长/回放，不能当硬上限）。有效batch16，按节点平方预算2097152做微批，回放12000压缩、每图成功缓存上限1000。CPU池上限24、训练最多2、开发按可用RAM预留6GiB自动分配，阶段结束关闭池。训练器可用RAM<3GiB时保存边界停止；独立守护每2秒查RAM，<4GiB写STOP并终止本轮进程组保留最近完整检查点。本机用户级cgroup v1硬内存限额不可靠，不依赖systemd属性。守护22小时有限运行，启动计数文件将自动训练尝试限制为首次+至多2次重试，避免只依赖systemd时间窗限速。

末次固定开发结果与父500图端点比较：主指标保留全部失败，报告成功数、失败罚时均值、共同成功时间/快慢地图、配对bootstrap区间；另存四模型共同成功交集，防止不同子集间直接比较条件时间。关注AB自身成功率/耗时，而不只看基线退步后的相对差距。单种子、重复查看开发集不足以确认稳定泛化，不承诺训练更久必然改善，不挑检查点。

根入口（后台运行时勿重复启动）：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
  --output experiments/native_easy_1000_20261018_seed101 \
  --from-run experiments/native_easy_500_20261017_seed101 \
  --dataset experiments/native_easy_20261015_seed101/dataset/native_dataset.json \
  --train-maps 1000 --pilot-episodes 1000 --num-meta-agent 24 --microbatch 16 --max-hours 18
```

后台训练`ir2-native-easy1000-20261018`，守护/交接收尾`ir2-native-easy1000-review-20261018`。实时训练状态pipeline.json、本目录training.log与monitor_status.json；TensorBoard在本轮tensorboard/。暂停touch新实验/STOP，等回合完整保存退出；恢复前确认后台已停止，按用户意愿移除新STOP后根入口--output新实验 --resume。旧10060正式实验STOP及旧监督PAUSE继续保留。完成时独立守护校验历史保护SHA、写COMPLETION.md并更新三个交接文档；不发消息。

工程证据：preflight_checks.json（独立6→8原图短续训，1000原图清单，8决策/60秒），验证继承、下一地图、STOP/恢复、完成后恢复无额外更新、不重跑前缀/头、预算与几何/通信。预检不能用于效果结论。正式inheritance_audit.json核验原样状态继承；launch_checks.json核验首张新地图501/index500和19000→19020、代码身份及历史结果不变；protected_runs.json涵盖旧100、500、10060回合结果。内存估算memory_review.json；正式资源以运行监测为准。
