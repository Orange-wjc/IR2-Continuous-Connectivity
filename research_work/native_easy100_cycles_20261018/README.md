# 固定100张原easy图多轮训练（2026-10-18）

用户最新指令：取消1000张扩图路线；固定100张地图反复训练，不因地图遍历一遍或时间预算暂停；用户亲自看TensorBoard，成功率接近99%后由用户决定何时评估。99%仅为显示目标，不自动停止或评估。取消旧1000路线的STOP已写，守护已停，旧训练完整保存退出：基线511回合/19220 SAC，AB510回合/19200 SAC；取消后的源、模型、数据身份均保留。

正式输出：`experiments/native_easy100_cycles_20261018_seed101`。起点选择纯100图正式端点`experiments/native_easy_repair_100_20261017_seed101`，不用已经见过500多张图的模型冒充固定100图训练。两组各自继承episode100/SAC11000的网络、Q/目标、Adam/温度、冻结上下文/消息头、RNG、12000条压缩回放及成功缓存（基线43张、AB48张）；只改dataset_manifest路径，训练地图数/成功缓存容量仍100。共享预训练9000和头5000不重跑，保留完整状态语义SHA核对。仅重新设计循环控制/训练统计，13个环境、模型、图、通信、worker和SAC/SIL核心文件与父快照逐字相同。

地图直接沿父清单100张原easy训练图，不生成或缩放、不按表现剔除难图。episode101→map_index0，199→98，200→99，201→0；每100回合为一遍。两组预算始终对齐：每方法完成一图后20 SAC，有效batch16，Q1e-4/Actor1e-5、SIL权重1、Q先更新/clip10、Actor clip5、合法动作目标熵、alpha[-8,2]、tau.01，196决策/3000物理秒限制均不改。训练回合数和整轮数不限，整段墙钟时间不限，没有99%自动停机。原100开发图不变且与训练几何隔离，原测试未用；无周期或终点评估，用户明确请求才评估。

进度解释：回合计数包含已有第1遍100张，故新训练从101/第2遍开始；总SAC=9000+20×累计回合。两组可能临时相差一回合，检查点逐图保存。既有100条训练及2000条适应更新日志复制到新目录并补写TensorBoard，后续曲线连续，旧结果不改。

为利用CPU并缩短等待，两方法的独立环境采样并行（最多2个CPU进程），GPU更新按原顺序执行；每组下一张图使用该组上张完成更新后的权重，不增加同组权重滞后的采样波次。原图短CLI预检对比优化前后，基线与AB在相同回合的完整保留学习状态（参数/优化器/回放/示例/RNG）语义SHA逐字一致，轨迹指标一致；不是对算法效果的证明。证据parallel_equivalence.json。

## TensorBoard

训练这台电脑浏览器打开 **http://127.0.0.1:6008**。这里只显示新路线IR2_FIFO与AB_learned；旧6007仍是旧实验，不用它看本轮。推荐Scalars选择：

- `train/success_rate_100_percent`：最近100回合训练成功率，纵轴0～100，99表示100回合成功99次；起点继承完整第1遍100条。
- `cycle/success_rate_percent`：每整遍100张图的训练成功率，横轴是完成的遍数；比尚未跑完一遍的比例更适合检查是否稳定。
- `train/target_success_rate_percent`：99的参考线，仅显示，不触发动作。
- `train/mean_capped_time_100`、`train/mean_min_local_coverage_100`、`train/mean_union_coverage_100`、`train/mean_return_100`：完成效率/覆盖/训练奖励辅助曲线。
- `losses/` 下策略、Q、模仿损失和温度用于排查数值问题；不能只凭loss平稳认定成功率收敛。

上述成功率本身已经按窗口平均，建议先将TensorBoard的Smoothing设为0看原始窗口曲线，勾选两组对照。`train/success_rate_20_percent`可看短期变化，`train/current_cycle_success_percent`是当前未完成一遍的临时比例，不能把开头一两张成功形成的100%当整轮收敛。训练中的策略不断变化且动作带探索，训练99%不代表独立开发/最终测试也99%；达到目标后仍训练，等用户发出评估指令。

## 后台和恢复

根driver.py --finite-contact新增`--continuous`；新目录先冻结源码后执行，绝不修改正在运行的source。启动（后台已运行勿重复执行）：

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
  --output experiments/native_easy100_cycles_20261018_seed101 \
  --from-run experiments/native_easy_repair_100_20261017_seed101 \
  --dataset experiments/native_easy_repair_100_20261017_seed101/dataset/native_dataset.json \
  --continuous --train-maps 100 --num-meta-agent 24 --microbatch 16 --no-auto-evaluation
```

训练服务`ir2-native-easy100-cycles-20261018`；守护`ir2-native-easy100-cycles-review-20261018`；TensorBoard `ir2-native-easy100-tensorboard-20261018`。新实验pipeline.json、本工作目录training.log/monitor_status.json记录实时状态；没有计划性轮数/时间停机。异常服务从完整检查点恢复，若连续3次在相同保存进度失败则记录needs_review.json，避免无意义反复崩溃；进度一旦推进恢复计数窗口重新开始。CPU上限24但训练只需两个独立进程，Torch主进程2线程/worker1线程，微批节点平方预算2097152、压缩回放12000、成功示例最多100图，保持设备余量。

内存保护保留：训练器RAM可用<3GiB中止到已保存边界；持久守护每2秒查看主机RAM，<4GiB时用RESOURCE_PAUSE停止本轮，恢复到6GiB后自动从完整检查点继续，不触发评估、不改变训练地图。守护没有22小时或其他生命周期期限。用户STOP始终优先，不会被自动恢复。本机用户级cgroup v1硬限制不可靠，不依赖systemd伪限额。其他不可恢复错误需结合日志修复，不承诺任何代码错误都能自动修复。

用户需要停止/评估时，可让助手执行；手工暂停可 `touch experiments/native_easy100_cycles_20261018_seed101/STOP`，等两组完成当前回合/保存退出。守护检测用户停止后退出，不自动重启；TensorBoard继续运行供查看。恢复先确认后台没有训练进程，按用户意愿移除STOP，再通过根入口`--output本轮 --resume`或重新启动训练/守护服务。不移除旧1000路线STOP，不恢复旧10060路线监督。

工程预检位于独立preflight/v2目录，8决策/60秒不用于算法效果结论。核对了原样继承、序号98/99→0/1循环、无回合/墙钟上限、99%不停止、无自动评估、TensorBoard历史/新数据、STOP/恢复和父结果不变。证据preflight_checks.json/parallel_equivalence.json。正式启动后launch_checks.json核对episode101/index0、两组11000→11020、真实TensorBoard HTTP曲线及历史保护SHA。
