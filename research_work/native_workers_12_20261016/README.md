# NUM_META_AGENT 接口与12进程试用

用户2026-10-16明确要求补齐原项目式并行采样参数，并试用12进程，核对内存和速度。

正式入口仍为 `experiments/native_easy_20261015_seed101/source/driver.py --finite-contact`。该driver顶部现在有 `NUM_META_AGENT = 12`，传给有限通信入口作为新实验默认值。原仓库默认入口及parameter.py未改动；正式改进实验使用上述冻结副本。

新增 `--num-meta-agent N`，兼容旧名 `--workers N`；新建和续训均可显式传入。续训不传时保留已保存的进程数，防止监督重启把资源设置覆盖回默认值。实际值同时持久化在pipeline.json的 `workers` 和 `NUM_META_AGENT`，改变时记入parallelism_changes，并在启动日志输出。非正整数在启动前拒绝。

这个参数控制同时采样地图的进程数，不改变每张地图2个机器人。当前仍是单GPU、embedding128、有效批量128、微批量上限64、每完成回合8次SAC；自动评估关闭。资源参数显式覆盖不重置检查点中的模型、优化器、回放或随机状态。

切换进程数同时改变波次宽度。若旧保存点两组差8回合，先用不足12回合的小波次补齐，再两组共同进入12回合波次，保持每个对齐回合相同SAC预算。固定权重的采样等价不代表不同波次宽度的整个长期训练轨迹相同；这属于两组共同记录的资源/调度变化。

## 验证

- 12项参数/真实协调循环模拟检查通过：driver默认值、CLI别名、非法参数、续训保留/覆盖、12回合波次预算相等、旧8回合补齐及自动评估保持关闭。
- 监督兼容19项检查通过，取消“只能8进程”的硬编码，核对已登记的正整数进程数与NUM_META_AGENT一致。故障恢复不擅自改进程数；启动可用内存保护按进程数伸缩，至少8GiB、或2GiB+每进程700MiB取更大者。
- 真正的driver.py --finite-contact --help已验证新参数暴露与有限通信分流。
- 暂停正式训练后，用第5128回合AB冻结模型，在相同24张原训练图、同一逻辑随机种子、完整196决策/真实终止规则下比较8与12。共48个额外CPU工程回合，0次SAC，不进入正式回放；开发/测试策略执行0。采样经验哈希逐回合完全相同，几何/通信审计通过。

| 指标 | 8进程 | 12进程 |
|---|---:|---:|
| 24回合耗时，含worker启动 | 190.643秒 | 160.583秒 |
| 完整回合/分钟 | 7.553 | 8.967 |
| 最低可用内存 | 19.85GiB | 18.32GiB |
| worker聚合RSS峰值，含共享重复计数 | 4.20GiB | 6.25GiB |
| 新增换出页数 | 0 | 0 |

采样吞吐提高18.72%，这组任务耗时降低15.77%。该探针不含GPU学习，不能承诺整个训练也提高相同比例。实际CPU+GPU恢复后的资源记录在正式目录 `checks/parallelism_12_20261016/runtime_memory_samples.jsonl`。历史8进程完整训练参考为最近256成对回合6.91个总回合/分钟，地图不同，不作为严格因果比较。

## 安全切换和后续修改

正式训练在保存边界暂停且两把锁空闲后，才替换driver.py/native_driver.py两个文件，另58个冻结文件不变。旧60文件归档 `source_versions/pre_meta_agent_12_20261016/`，五个共享/初始/最新检查点原始SHA在切换前后不变；恢复边界完整副本保存在各组checkpoints/。监督暂时PAUSE，更新其兼容规则后重启并重新允许监督。

续训命令中的有效配置示例（先确认旧进程退出、清除训练STOP，不能与现有训练重复启动）：

```text
python <正式目录>/source/driver.py --finite-contact --output <正式目录> --resume --num-meta-agent 12 --no-auto-evaluation
```

将来改为其他值优先使用显式CLI参数；不要热改运行中的冻结源码或手工编辑正在写入的pipeline.json。先暂停监督恢复功能与训练，确认保存和退出，再修改启动参数，最后恢复监督。新实验才使用driver常量默认值；续训省略该参数将沿用已有值。

证据：capacity_review.json、probe_timings.json、parameter_tests.log、supervisor_tests.log、parameters.patch、cli_help.txt；正式切换和实际运行核验在 `experiments/native_easy_20261015_seed101/checks/parallelism_12_20261016/`。

### 12进程正式运行复核（2026-10-16 19:55）

两组均已实际保存完整12回合波次；快照FIFO5160回合/41432次SAC、AB5148回合/41336次SAC，更新数逐组符合216+8×(episode−8)。正式训练可用RAM最低17.628GiB，按MemTotal−MemAvailable计占用最高12.829GiB；GPU占用最高4513MiB，采样到GPU利用率98%。系统全局新增换出239页（约0.93MiB，包含无关进程，旧1.8GiB swap预先存在），不是“正式运行完全无换出”，未见OOM。60文件身份、数据清单及共享/初始/恢复点备份SHA核验通过。监督healthy，自动评估关闭；post_switch_review.json已保存实际证据。
