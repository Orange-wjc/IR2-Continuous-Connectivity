# 原easy累计500张续训（2026-10-17）

用户授权“下轮训练500张，再看看有没有效果”。按上一轮讨论的100→500解释：每组从自己的100图正式端点继续，新增400张原easy训练图，累计500张；不是额外500/累计600，也不重做前缀或消息头。旧100图结果完整保留。复杂地图、原测试集及全量9899训练不启动，不发飞书。

正式输出：`experiments/native_easy_500_20261017_seed101`。父实验：`experiments/native_easy_repair_100_20261017_seed101`。维护入口：根driver.py --finite-contact，新增--from-run。代码仍先冻结再执行，--resume只执行自己的source/，不修改正在执行或历史快照。

本轮唯一学习条件变化为原easy训练地图范围扩大。每方法保留自身网络、Q/目标、Adam/温度、冻结上下文/消息头、12000条压缩回放、成功自示例及完整RNG、回合/梯度计数；不重置回放、成功记忆或优化器。仅修改三项配置：dataset_manifest改为本轮清单、train_maps100→500、success_episode_limit100→500（继续每图最快成功轨迹，不因扩图丢旧示例）。导入前后完整保留状态语义哈希逐项验证，原字节检查点放inherited/，本轮initial.pth是100图状态的配置迁移版，latest从其续训。

原图：沿现有全量冻结清单easy训练前500文件，前100与父实验一致，新增索引100～499；episode101对应map_index100。原100张开发图完全不变且训练几何隔离，原PNG逐个SHA验证。不生成/缩放地图，不根据开发表现选训练图，原测试不使用。

预算：每组新增400地图回合×20=8000 SAC；总回合累计500，总SAC累计19000=9000共享+10000适应。共享移动100图/9000 SAC、头采样100图/5000监督更新完全复用，不再执行。每组自己的checkpoint续训，不把AB权重复制给基线。种子仍101，是一次轨迹延续而非新的独立重复。只在500图端点贪心评估原100开发图/方法，无周期评估。

公平性：环境、图、模型、NativeLearner、worker、通信等13个关键文件与父实验逐字相同；仅扩展driver的续训准备/有限预算控制，分析器支持累计计数及100→500端点对照。仍每方法一图后20更新，双方法CPU采样与另一组GPU更新交错，不改变权重刷新节奏。有效batch16、Q LR1e-4、Actor1e-5、SIL权重1、Q先更新/clip10、Actor clip5、合法动作目标熵、alpha[-8,2]、tau.01。图196决策/3000物理秒的停止条件不变。

硬件保护：CPU池上限24，训练最多2进程，开发按RAM预留6GiB及每进程0.85GiB自动下调；阶段关闭池。微批16上限/节点平方2097152预算，回放12000无损压缩；成功记忆最多500张/方法（200余成功时预计仍有充足余量，以实际资源监视为准）。可用RAM低于3GiB训练器停止；独立monitor每2秒检查、低于4GiB写STOP并终止本轮进程组，留最近完整检查点。当前用户级cgroup v1硬内存限制不可靠，不把systemd属性当成生效硬保护。每图完整保存；失败最多有限重试2次，不无限循环。14小时单次墙钟保护，守护最多18小时。

结束后自动生成RESULTS.md、analysis/comparison.json、paired_maps.json、round_progress.json和PNG/SVG，随后退出。主指标保留全部100开发图及失败罚时；比较本轮基线/AB，同时与父100图端点比较。共同成功集合会随检查点变化，另存四模型共同成功交集，避免把不同子集的条件耗时直接作纵向结论。配对bootstrap区间与失败/退步地图均保留；同一开发集反复查看，不能当独立最终测试。达到均值/中位数小试筛选条件也不等于稳定泛化，不保证训练更多必然有效。不自动超500、不自动进入复杂地图。

启动命令（后台服务运行时不要重复执行）：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
  --output experiments/native_easy_500_20261017_seed101 \
  --from-run experiments/native_easy_repair_100_20261017_seed101 \
  --dataset experiments/native_easy_20261015_seed101/dataset/native_dataset.json \
  --train-maps 500 --pilot-episodes 500 --num-meta-agent 24 --microbatch 16 --max-hours 14
```

后台训练服务：ir2-native-easy500-20261017；守护/交接收尾：ir2-native-easy500-review-20261017。日志：本目录training.log；实时阶段：实验pipeline.json与本目录monitor_status.json；TensorBoard：实验tensorboard/。暂停：touch新实验/STOP，等完整保存退出；恢复：确认后台不在运行，按用户意愿移除STOP后通过根入口--output新实验 --resume。旧10060实验STOP和旧监督PAUSE继续保留。

验证：preflight_checks.json来自独立工程续训（继承4→累计6，500张原图清单、8决策/60秒），检查完整状态继承、无前缀/头重训、正确下张地图、STOP/恢复、完成后恢复无新增SAC、端点运动/通信审计与父文件不变。工程预检不用于算法性能结论；正式启动后另核验episode101对应索引100、两组11000→11020，确保真实续训。继承审计在正式实验inheritance_audit.json；历史保护SHA在protected_runs.json。
