# 训练控制切换记录

## 最新用户指令：连续训练，评估仅手动触发（2026-10-15）

用户明确要求不再等待周期评估，先持续训练，之后由用户选择评估时间。本条优先于下文历史的“每512回合自动开发评估”安排。

- 正式目录仍为 `experiments/native_easy_20261015_seed101`。`pipeline.json` 已持久化 `automatic_development_evaluation=false` / `evaluation_policy=manual_only`；重启不会自行恢复周期评估。512仍是保留的旧间隔参数，关闭状态下不触发。
- 已开始的第512回合评估正常收尾后，旧协调器退出并释放锁，才替换 `source/native_driver.py` 的控制逻辑。其余59个冻结文件不变；旧60文件原样归档在正式目录 `source_versions/automatic_eval_20261015/`。新的60文件身份已写入 `source_identity.json` 和 `pipeline.json`，不得用旧身份校验新版本。
- 从两组第512回合、各4248次SAC更新的完整检查点继续，未重置模型、Adam、回放、随机状态或消息头。共享起点、两组初始/最新检查点的5个SHA256在切换前后全部一致。两组第512回合完整副本另存各组 `checkpoints/episode_00000512.pth`，可供以后评估。
- 仍为8个采样进程、有效批量128、每回合8次SAC，两组配对预算不变；每批完整保存、TensorBoard、逐回合轨迹/通信审计继续。48小时保存暂停保护仍保留，不代表训练成功；不自动进入difficult，不执行原测试集，不主动发飞书。
- 已恢复后台训练，切换启动时间 `2026-10-15T19:02:17.728256+08:00`、PID `72512` 仅作快照。当前进度用 `python3 experiments/native_easy_20261015_seed101/checks/progress.py` 核对；前台观察用 `tail -n 20 -f experiments/native_easy_20261015_seed101/training.log`，TensorBoard仍为 http://127.0.0.1:6007 。不要重复启动。
- 恢复命令沿用正式冻结 `source/driver.py --finite-contact --output <正式目录> --resume`，可显式加 `--no-auto-evaluation`。参数和源码身份已持久化；新开会话不得擅自重新开启自动评估。用户明确要求评估时，再选择已保存检查点、固定地图和评估预算执行；继续保留训练/开发/测试边界。
- 4项真实协调循环的模拟检查通过：关闭后跨512边界继续到520、两组更新预算相等、重启保持关闭、旧默认开启行为仍可用。本次检查不运行地图或真实梯度更新。证据：`checks/manual_evaluation_switch_20261015/`（补丁、测试日志、旧配置与SHA核验）。

正式证据目录：/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101/checks/manual_evaluation_switch_20261015
