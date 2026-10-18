# 固定10张原easy地图重复训练：学习能力诊断（2026-10-18）

用户因100图一遍等待较久，要求换10张地图重复训练排查能否持续学会。100图轮训已在两组episode501/SAC19020完整保存并停止，STOP保留，原6008曲线保留；不恢复旧1000或旧10060训练。不发飞书。

新输出 `experiments/native_easy10_cycles_20261018_seed101`。根 `driver.py --finite-contact --continuous --train-maps 10` 冻结维护代码后启动。只取已审核100图清单的前10张原easy PNG，顺序原样，不按成功率选易图、不生成/缩放，不改任务门槛。

两组从100图小试的 **common_start.pth共同预训练起点** 开始各自适应；不是继承各自已训练501回合的不同网络。这个起点已使用100张原图做9000次共享移动更新和5000次消息头更新，所以这是“后续适应限定10张”的诊断，**不是从随机权重开始的纯10图实验**。模型/Q/目标/Adam/温度、冻结上下文与消息头、RNG和9000共享更新计数原样保留；回放、内容回放及成功轨迹缓存原本都为空，已核对且不导入100图的回放/适应日志。共享起点的历史imitation_updates=8190计数保留，它属于此前预训练，并不表示导入成功缓存。

每组新适应episode从1计，1→index0，10→9，11→0；每10回合是一遍。每完整训练图之后20 SAC，batch16，Actor1e-5/Q1e-4、SIL权重1、per_map成功缓存最多10张、回放压缩12000、微批16/节点平方预算2097152、Q先更新/clip10、Actor clip5、合法动作目标熵、alpha[-8,2]和tau.01保持。正常诊断196决策/3000物理秒上限、99%双机本地覆盖、付费有限通信等均不改。13个算法/环境/worker/图/通信文件与父快照逐字相同。两个独立方法CPU采样并行，GPU逐方法更新；同组下一图用最新完成更新的权重。

没有计划回合/时间上限，不自动测试，不因99%停训。只做训练日志统计；原100开发图保持隔离但不执行，原测试未用。用户决定何时停止或评估。不能保证训练达到99%，也不能用10张图拟合结果证明泛化或新算法优越性。

## 曲线

新TensorBoard **http://127.0.0.1:6009**，6008保留旧100图记录。

- `cycle/success_rate_percent`：每完整一遍10张的成功率，横轴遍数。每遍只有10次结果，取值以10个百分点跳变，单遍100%不是收敛。
- `train/success_rate_100_percent`：最近100回合，即最多最近10遍；未满100时分母是已有回合数，必须结合窗口样本量解释。建议Smoothing=0。
- `train/success_rate_10_percent`：最近10回合，适合快速观察但噪声大。
- `maps/map_00`至`maps/map_09`：每张图的`success`、最近10次访问成功率、完成/失败罚时、最小本地覆盖，横轴是该图访问次数。未达到10次访问时最近10次成功率分母为已有次数。
- `losses/`：SAC/Q/温度/模仿损失；原始updates.jsonl还保留梯度范数、有效batch、熵、成功示例等排查证据。

先比较相同10图跨多遍是否持续改善、哪些图反复失败，再判断是学习问题、随机探索、容量/目标还是任务限制。没有趋势时不能仅靠延长训练许诺99%。

## 启动和停止

后台已启动后不要重复执行。训练单元 `ir2-native-easy10-cycles-20261018`；守护 `ir2-native-easy10-cycles-review-20261018`；曲线 `ir2-native-easy10-tensorboard-20261018`。

```bash
/home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
  --output experiments/native_easy10_cycles_20261018_seed101 \
  --from-run experiments/native_easy_repair_100_20261017_seed101 \
  --dataset experiments/native_easy_repair_100_20261017_seed101/dataset/native_dataset.json \
  --continuous --train-maps 10 --num-meta-agent 24 --microbatch 16 --no-auto-evaluation
```

新实验逐图保存；RAM可用<3GiB训练器保护，守护<4GiB RESOURCE_PAUSE，恢复到6GiB续训。服务异常从完整检查点恢复；连续3次相同保存进度失败记录needs_review并停止，进度推进后重试窗口重置。NUM_META_AGENT上限24，当前独立方法训练只需2个CPU进程；不增加同方法滞后采样以假装提速。不依赖本机用户级cgroup v1硬限额。

手动暂停：`touch experiments/native_easy10_cycles_20261018_seed101/STOP`，等当前回合完整保存退出，守护尊重STOP；曲线继续可看。恢复仅在用户要求且确认没有现存训练后移除此STOP，再根driver `--finite-contact --output本轮 --resume`及守护。不要删除旧实验STOP/监督PAUSE。

工程检查 `preflight_checks.json` 使用原10图、短8决策、正常3000物理秒；核对跨10边界循环、两组同起点、实际梯度更新/仅10图回放、STOP与resume、逐图统计恢复、事件曲线和旧文件SHA。这些短工程回合不是学习效果证据，不混入正常诊断。新执行源码冻结后不再修改。

## 用户暂停（2026-10-18 13:22）

两组12回合/9240累计SAC完整保存退出，检查点读取核对通过，STOP保留、训练和守护均inactive。不自动恢复，TensorBoard6009保留。见user_pause_20261018.json。
