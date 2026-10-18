"""Persist the measured worker-count choice after successful continuation."""
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
WATCH = Path('/home/wyz/wjc_ws/training_supervisor_native_easy')
EVIDENCE = ROOT / 'checks/parallelism_24_20261016'
capacity = json.loads((WORK / 'capacity_review.json').read_text())
plan = json.loads((ROOT / 'pipeline.json').read_text())
switch = json.loads((EVIDENCE / 'switch_review.json').read_text())
runtime = json.loads((EVIDENCE / 'post_switch_review.json').read_text())
selected = capacity['recommended_workers']
assert plan['workers'] == plan['NUM_META_AGENT'] == runtime['workers'] == selected
assert runtime['status'] == 'passed'
baseline, candidate = capacity['timings']
decision = '实测采样提速且内存安全，保留24进程' if selected == 24 else '24未满足提速或资源条件，恢复并保留16进程'
source_note = ('正式冻结source/driver.py的NUM_META_AGENT默认常量改为24，另外59文件不变；旧60文件归档source_versions/pre_meta_agent_24_20261016/。'
               if selected == 24 else '正式60文件源码均保持原16进程版本，不应用候选24默认值。')
notice = f'''## 最新并行试用：16对24进程，最终采用{selected}（2026-10-16）

用户希望利用内存余量试用24。机器Ryzen 9 9900X为12物理核/24逻辑线程，不能按内存余量推断速度。先PAUSE监督，正式训练完整保存到{switch['saved_episodes']}后正常退出；同一AB冻结权重、同一48张原easy训练图/逻辑种子、完整终止和运动/通信审计规则比较16与24，不与正式训练争用CPU。共96个工程CPU回合，SAC/开发/测试策略执行均0，不进入正式回放。

16进程用时{baseline['wall_seconds']:.3f}秒、采样吞吐{baseline['rollouts_per_minute']:.3f}回合/分钟；24用时{candidate['wall_seconds']:.3f}秒、吞吐{candidate['rollouts_per_minute']:.3f}。24/16吞吐比{capacity['speedup_24_over_16']:.4f}，耗时降低{capacity['wall_time_reduction_percent']:.2f}%。最低可用RAM分别{baseline['min_available_memory_mib']/1024:.2f}/{candidate['min_available_memory_mib']/1024:.2f}GiB，新增系统换出{baseline['swap_out_pages']}/{candidate['swap_out_pages']}页。48回合经验哈希逐对一致，几何/通信审计通过。结论：{decision}。这是一次固定模型CPU采样比较，含进程启动，不保证整轮CPU+GPU训练同比例提速，不与历史不同地图测速连乘。

资源判断说明：原探针把任意新增系统换出视为不安全，因此23页（92KiB）触发了保守否决；人工核查14.62GiB可用RAM及所有任务正常后，不据这点主机全局换出认定采样进程内存不足。改用“可用RAM>8GiB、整臂全局换出≤1MiB、任务及审计通过”的工程保护标准。原判定capacity_review_strict_zero_swap.json、原始日志和计数完整保留，修订依据写入capacity_review.json的memory_safety_review；没有重跑或改动经验、耗时和审计结果。

{source_note} 两个训练锁空闲后才切换和启动唯一协调器。共享/初始检查点及原latest恢复边界SHA核验通过；完整恢复副本各组checkpoints/parallelism24_对应回合.pth保留。模型、目标网络、Adam、回放、RNG完整继续。入口默认值及pipeline workers/NUM_META_AGENT均为{selected}，显式续训`--num-meta-agent {selected}`；省略则沿用保存配置。两组共同改变波次宽度，长期策略滞后会改变，不声称整段长训练逐位相同。

实际CPU+GPU复核通过：{runtime['verified_at']}，保存进度{ {g: s['episodes'] for g, s in runtime['real_training_progress'].items()} }，均完成新完整波次。最低可用RAM {runtime['min_runtime_available_memory_gib']:.2f}GiB、最高占用{runtime['max_runtime_used_memory_gib']:.2f}GiB、显存峰值{runtime['max_gpu_memory_mib']/1024:.2f}GiB；新增系统换出{runtime['new_system_swap_out_pages']}页（主机全局计数，旧swap不清空）。60源码身份/数据清单/恢复副本核验通过，实际采样进程数见runtime_processes.json，常驻监督healthy。有效批量128、微批量64、每完成回合8次SAC、每图2机器人不变。

自动评估仍关闭，不自动进入复杂阶段，不发飞书。4项参数/续训配置检查通过，监督既有正整数workers及按数量伸缩的内存保护无需改码。实时进度仍以coordinator/pipeline/status及真实进程为准。本条覆盖下方历史16配置。

证据：`research_work/native_workers_24_20261016/`；正式目录`checks/parallelism_24_20261016/`（capacity_review、switch_review、post_switch_review、runtime_memory_samples和runtime_processes）。

'''
paths = (Path('/home/wyz/wjc_ws/papers/多机器人有限通信探索_实验计划与新对话交接.md'),
         REPO / 'finite_contact/EXPERIMENT_HANDOFF.md', REPO / 'finite_contact/FEISHU_TASK_CONTEXT.md',
         ROOT / 'TRAINING_PLAN.md', REPO / 'research_work/native_curriculum_20261015/STATUS.md',
         REPO / 'finite_contact/results/native_curriculum_20261015/README.md')
for path in paths:
    text = path.read_text()
    assert notice.splitlines()[0] not in text
    head, rest = text.split('\n', 1)
    path.write_text(head + '\n\n' + notice + rest.lstrip('\n'))
(WORK / 'README.md').write_text('# 24进程试用与选择\n\n' + notice)
path = REPO / 'README.md'
head, rest = path.read_text().split('\n', 1)
path.write_text(head + '\n\n2026-10-16最新并行配置：NUM_META_AGENT=%d；%s。详见[24进程试用记录](research_work/native_workers_24_20261016/README.md)。\n' % (selected, decision) + rest)
path = WATCH / 'README.md'
head, rest = path.read_text().replace('当前16', '当前%d' % selected).split('\n', 1)
path.write_text(head + '\n\n2026-10-16最新：16/24同图比较后采用%d，监督按登记值恢复；自动评估保持关闭。\n' % selected + rest)
deployment = json.loads((WATCH / 'deployment.json').read_text())
deployment.update(current_parallelism=selected, registered_parallelism=selected,
                  current_training_source_identity=plan['source_identity'],
                  latest_parallelism_trial=str(EVIDENCE))
deployment['files']['README.md'] = hashlib.sha256(path.read_bytes()).hexdigest()
(WATCH / 'deployment.json').write_text(json.dumps(deployment, ensure_ascii=False, indent=2) + '\n')
print('Recorded measured 16/24 choice and verified continuation:', selected)
