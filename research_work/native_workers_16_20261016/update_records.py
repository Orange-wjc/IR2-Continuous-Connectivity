"""Record the measured decision without changing running training code."""
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
WATCH = Path('/home/wyz/wjc_ws/training_supervisor_native_easy')
capacity = json.loads((WORK / 'capacity_review.json').read_text())
plan = json.loads((ROOT / 'pipeline.json').read_text())
selected = capacity['recommended_workers']
assert plan['workers'] == plan['NUM_META_AGENT'] == selected
baseline, candidate = capacity['timings']
decision = '采样提速且内存安全，已正式保留16进程' if selected == 16 else '16未达到稳定提速/资源要求，已恢复12进程'
notice = f'''## 最新并行试用：12对16进程，最终采用{selected}（2026-10-16）

用户要求从12改为16试用。监督先PAUSE，12进程训练正常保存退出到两组5172回合，各41528次SAC。用同一AB冻结模型、同一48张原训练图、相同逻辑随机种子、完整终止/审计规则比较12与16，无正式训练竞争；共96个额外工程CPU回合，梯度更新0，不进入正式回放，开发/测试策略执行0。

12进程用时{baseline['wall_seconds']:.3f}秒，16用时{candidate['wall_seconds']:.3f}秒；吞吐比{capacity['speedup_16_over_12']:.4f}，耗时降低{capacity['wall_time_reduction_percent']:.2f}%。最低可用内存分别{baseline['min_available_memory_mib']/1024:.2f}/{candidate['min_available_memory_mib']/1024:.2f}GiB，新增换出分别{baseline['swap_out_pages']}/{candidate['swap_out_pages']}页。48回合经验哈希完全一致，运动/字节容量审计通过。结论：{decision}。这只是固定模型CPU采样探针，不能承诺GPU整轮训练同比例提速，不能与前一批不同地图直接连乘报总提速。

正式入口source/driver.py的NUM_META_AGENT与pipeline的workers/NUM_META_AGENT均为{selected}；续训显式 `--num-meta-agent {selected}`，省略时沿用保存值。若选16，只改driver.py默认常量，另外59个冻结文件不变，旧60文件归档source_versions/pre_meta_agent_16_20261016/；若保留12，则正式源码不变。五个共享/初始/最新检查点SHA在切换前后不变，恢复点完整副本保存各组checkpoints/parallelism16_00005172.pth。

原模型/Adam/回放/RNG完整继续，每图2机器人、单GPU、有效批量128、微批量上限64、每回合8次SAC不变。两组共同采用相同新波次宽度，长期轨迹不声称逐位一致。自动评估保持关闭，不自动进入复杂阶段，不发送飞书。监督已支持登记的正整数进程数及伸缩内存保护，无需再改其程序；只在实际模型加载成功后重新允许监督。4项16配置检查通过。

证据：`research_work/native_workers_16_20261016/` 和正式目录 `checks/parallelism_16_20261016/`。实际CPU+GPU运行复核以post_switch_review.json为准，实时进度仍读取coordinator/pipeline/status，不把本条历史回合当实时进度。

'''
for path in (Path('/home/wyz/wjc_ws/papers/多机器人有限通信探索_实验计划与新对话交接.md'), REPO / 'finite_contact/EXPERIMENT_HANDOFF.md',
             REPO / 'finite_contact/FEISHU_TASK_CONTEXT.md', ROOT / 'TRAINING_PLAN.md',
             REPO / 'research_work/native_curriculum_20261015/STATUS.md', REPO / 'finite_contact/results/native_curriculum_20261015/README.md'):
    text = path.read_text()
    assert notice.splitlines()[0] not in text
    head, rest = text.split('\n', 1)
    path.write_text(head + '\n\n' + notice + rest.lstrip('\n'))
(WORK / 'README.md').write_text('# 16进程试用与选择\n\n' + notice)
path = REPO / 'README.md'
head, rest = path.read_text().split('\n', 1)
path.write_text(head + '\n\n2026-10-16最新并行配置：NUM_META_AGENT=%d；12/16对比已完成，%s。详见[16进程试用记录](research_work/native_workers_16_20261016/README.md)。\n' % (selected, decision) + rest)
path = WATCH / 'README.md'
text = path.read_text().replace('当前12', '当前%d' % selected)
head, rest = text.split('\n', 1)
path.write_text(head + '\n\n2026-10-16最新：12/16实测后正式采用%d，监督按登记的workers/NUM_META_AGENT恢复，保持用户主动暂停和评估关闭规则。\n' % selected + rest)
deployment = json.loads((WATCH / 'deployment.json').read_text())
deployment.update(current_parallelism=selected, registered_parallelism=selected,
                  current_training_source_identity=plan['source_identity'],
                  latest_parallelism_trial=str(ROOT / 'checks/parallelism_16_20261016'))
deployment['files']['README.md'] = hashlib.sha256(path.read_bytes()).hexdigest()
(WATCH / 'deployment.json').write_text(json.dumps(deployment, ensure_ascii=False, indent=2) + '\n')
print('Recorded measured 12/16 decision and persisted parallelism:', selected)
