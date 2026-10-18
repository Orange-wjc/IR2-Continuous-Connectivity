# 原图100张修复试验（2026-10-17）

用户授权：修复迁移后重训，先100张原easy地图验证，再考虑大规模。旧原图10060回合/80632 SAC不继续、不覆盖，STOP和旧监督PAUSE保留，不发飞书。

维护代码：根driver.py统一入口，旧driver原样保留为driver_legacy.py；原图改进实现位于native_training/，从已经通过迁移审计的旧执行快照复制，修复native_driver.py/native_learning.py/native_worker.py，新增端点native_analysis.py。保留其原图环境/分层图及历史图恢复修复。其余历史finite_contact模块和实验工具不改动。native_pilot.py把维护源码冻结到实验source后exec，恢复仍用自身快照，绝不对正在运行的源码打补丁。

正式输出：`experiments/native_easy_repair_100_20261017_seed101`。先工程预检，后正式小试；预检为8决策/60秒的短流程，不能用于算法效果判断。数值检查使用合成数组夹具，不是生成新训练地图。

预算固定：现有9899 easy训练条目中按原清单顺序前100文件，0开发/测试训练；共享100图×90=9000 SAC，复制冻结上下文；100训练图每图至多2048标签、5000次头更新；两组从同一共同起点各100图×20=2000适应SAC；相同原100开发图各做一次完整196决策/3000秒贪心端点评估。不同训练图100张；移动100+头采样100+适应200+开发200，共600条地图轨迹。不是只执行100次环境回合，也不从旧10060回合权重初始化。

学习修复：成功轨迹入每图缓存并实际加入Actor损失；有效batch16、Q学习率1e-4、Actor适应1e-5、Q先更新/裁剪10、合法动作目标熵、alpha边界[-8,2]、tau=.01软更新。保持原图/分层图/128维模型/奖励/通信时间成本，未根据开发结果改会合距离尺度。9000共享和头训练恢复的是旧成功方案，不承诺100图适应足以复现旧结果。预实验训练1000适应图而本轮仅100张，这仍是预算差异。

硬件：训练按每方法一图后更新，两方法间采样与GPU更新交错；标签/开发CPU池最多24，按RAM预留6GiB及每进程0.85GiB自动下调，阶段结束关闭池；有效batch16内按图节点平方预算微批，保留一次完整优化器更新。RAM可用低于3GiB中止到已有完整保存边界，不跳图。进程级训练锁；每图保存Adam/RNG/成功示例/压缩回放。后台服务配置22GiB RAM/256MiB swap，但本机用户级cgroup v1没有独立memory控制组（MemoryCurrent未设），这些硬限额未生效，不能依赖。实际保护为内存预留/微批/压缩缓存/低内存停止，加独立监视每2秒读MemAvailable，低于4GiB就写STOP并终止本轮进程组，保留最近完整检查点；失败重试最多2次；绝不重启旧正式实验。

端点生成RESULTS.md、analysis/comparison.json、逐图差值和PNG/SVG。预先筛选条件为审计/相同预算/模仿实际参与均通过，AB成功数不下降、失败罚时均值降低、共同成功平均及中位耗时均降低。保留全部地图结果及配对bootstrap95%区间。条件通过也只是单种子小试信号，不能宣称已证明普遍有效；原测试集仍保留。两组达到100后不再训练，不自动开启大规模/difficult；结果不明确则报告并继续定位。

启动（不要与后台服务重复执行）：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
  --output experiments/native_easy_repair_100_20261017_seed101 \
  --dataset experiments/native_easy_20261015_seed101/dataset/native_dataset.json \
  --train-maps 100 --pilot-episodes 100 --num-meta-agent 24 --microbatch 16
```

恢复用同一根入口 `--output ... --resume`。暂停：在本轮输出目录touch STOP，等完整回合保存退出；STOP存在时后台服务不重启。日志在本工作目录training.log，实时状态在实验pipeline.json（共享移动→头采集→头拟合→双组适应→开发评估→pilot_complete）。TensorBoard位于新实验tensorboard/。

检查证据：unit_checks.json（原训练器数值对比、SIL链路、压缩观测恢复、alpha）；lifecycle_checks.json（根入口、STOP/恢复、有限预算、结束后恢复不额外训练、几何/通信审计）；old_run_protected.json（旧正式文件SHA）。原图工程预检轨迹与失败不当作效果证据，不隐去预检数据。

独立监视/收尾服务 `ir2-native-repair100-review-20261017` 只监督本轮和写报告：每2秒读RAM，20秒写monitor_status.json；成功完成后核对旧文件SHA、生成COMPLETION.md并更新三个交接文档；运行失败则写needs_review.json，不修改训练源码、不扩大预算、不发消息。监视最多16小时后退出。
