"""Read saved training records only; no model execution or training mutation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/ir2-review-mpl'
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
OUT = ROOT / 'checks/training_review_20261017_0830'
OUT.mkdir(exist_ok=True)
CUTOFF = 9988
UPDATES = 216 + 8 * (CUTOFF - 8)
GROUPS = ('IR2_FIFO', 'AB_learned')
COLORS = ('#3767A6', '#D35E35')
LABELS = ('基线 IR2_FIFO', '改进 AB_learned')
plan = json.loads((ROOT / 'pipeline.json').read_text())
assert min(plan['completed_episodes'].values()) >= CUTOFF
manifest_path = ROOT / 'dataset/native_dataset.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == plan['dataset_sha256']
dataset = json.loads(manifest_path.read_text())
train = dataset['families']['easy']['splits']['train']


def records(path, key, boundary):
    selected = {}
    ignored = 0
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            ignored += 1
            continue
        if row[key] > boundary:
            ignored += 1
            continue
        assert row[key] not in selected, (path, row[key])
        selected[row[key]] = row
    return selected, ignored


rollouts = {}
updates = {}
ignored = {}
for group in GROUPS:
    rollouts[group], it = records(ROOT / group / 'training.jsonl', 'episode', CUTOFF)
    updates[group], iu = records(ROOT / group / 'updates.jsonl', 'updates', UPDATES)
    assert set(rollouts[group]) == set(range(1, CUTOFF + 1))
    assert set(updates[group]) == set(range(217, UPDATES + 1))
    for episode, row in rollouts[group].items():
        entry = train[(episode - 1) % len(train)]
        assert row['map']['file_sha256'] == entry['file_sha256']
        assert row['map']['index'] == (episode - 1) % len(train)
        assert row['learner_updates'] <= UPDATES
    ignored[group] = dict(training_tail_or_above_boundary=it, updates_tail_or_above_boundary=iu)
for episode in range(1, CUTOFF + 1):
    assert rollouts[GROUPS[0]][episode]['map']['sha256'] == rollouts[GROUPS[1]][episode]['map']['sha256']


def window(first, last):
    indices = list(range(first, last + 1))
    both = [i for i in indices if all(rollouts[g][i]['success'] for g in GROUPS)]
    result = dict(first_episode=first, last_episode=last, maps=len(indices), both_success_maps=len(both), groups={})
    for group in GROUPS:
        rows = [rollouts[group][i] for i in indices]
        failed = [x for x in rows if not x['success']]
        result['groups'][group] = dict(
            successes=sum(x['success'] for x in rows), success_rate=float(np.mean([x['success'] for x in rows])),
            mean_capped_time=float(np.mean([x['capped_time'] for x in rows])),
            mean_union_coverage=float(np.mean([x['union_coverage'] for x in rows])),
            mean_min_local_coverage=float(np.mean([min(x['coverage']) for x in rows])),
            both_success_mean_time=float(np.mean([rollouts[group][i]['time'] for i in both])) if both else None,
            mean_robot_return=float(np.mean([np.mean(x['return_values']) for x in rows])),
            failure_stop_reasons=dict(Counter(x['stop_reason'] for x in failed)),
            mean_decisions=float(np.mean([x['decisions'] for x in rows])),
            mean_sent_bytes=float(np.mean([x['sent_bytes'] for x in rows])),
            mean_wait_seconds_per_robot=float(np.mean([np.mean(x['wait_seconds']) for x in rows])),
            mean_wait_without_new_map_seconds_per_robot=float(np.mean([np.mean(x['wait_without_new_map_seconds']) for x in rows])),
            failed_union_at_least_99=sum(x['union_coverage'] >= .99 for x in failed),
            failed_min_local_at_least_95=sum(min(x['coverage']) >= .95 for x in failed))
    a, b = [result['groups'][g] for g in GROUPS]
    result['comparison'] = dict(
        success_difference_percentage_points=100 * (b['success_rate'] - a['success_rate']),
        capped_time_reduction_percent=100 * (1 - b['mean_capped_time'] / a['mean_capped_time']),
        both_success_time_increase_percent=100 * (b['both_success_mean_time'] / a['both_success_mean_time'] - 1) if both else None,
        AB_only_success=sum(rollouts['AB_learned'][i]['success'] and not rollouts['IR2_FIFO'][i]['success'] for i in indices),
        FIFO_only_success=sum(rollouts['IR2_FIFO'][i]['success'] and not rollouts['AB_learned'][i]['success'] for i in indices))
    return result


def numeric_summary(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    return dict(mean=float(np.mean(finite)), median=float(np.median(finite)),
                p95=float(np.percentile(finite, 95)), p99=float(np.percentile(finite, 99)),
                maximum=float(np.max(finite)), minimum=float(np.min(finite)),
                nonfinite=int(len(a) - len(finite)))


def behavior(first, last, both_only=False):
    indices = [i for i in range(first, last + 1)
               if not both_only or all(rollouts[g][i]['success'] for g in GROUPS)]
    result = dict(maps=len(indices), groups={})
    for group in GROUPS:
        rows = [rollouts[group][i] for i in indices]
        values = {k: float(np.mean([x[k] for x in rows])) for k in
                  ('time', 'total_distance', 'sent_bytes', 'novel_received_cells',
                   'duplicate_received_cells', 'connected_seconds', 'decisions')}
        values.update({k: float(np.mean([np.mean(x[k]) for x in rows])) for k in
                       ('wait_seconds', 'wait_without_new_map_seconds', 'synchronization_idle_seconds')})
        values['reward_components'] = {k: float(np.mean([np.mean(x['reward_components'][k]) for x in rows]))
                                       for k in rows[0]['reward_components']}
        result['groups'][group] = values
    return result


keys = ('q1_loss', 'q2_loss', 'policy_loss', 'alpha', 'entropy', 'wait_probability',
        'policy_grad_norm', 'q_grad_norm')
optimization = {}
for group in GROUPS:
    rows = list(updates[group].values())
    optimization[group] = dict(
        all_updates_nonfinite=sum(not np.isfinite(x[k]) for x in rows for k in keys),
        first_4096={k: numeric_summary([x[k] for x in rows[:4096]]) for k in keys},
        last_4096={k: numeric_summary([x[k] for x in rows[-4096:]]) for k in keys},
        last_update=rows[-1],
        last_4096_q_grad_above_clip=sum(x['q_grad_norm'] > 2000 for x in rows[-4096:]))
blocks = [window(i, min(i + 1023, CUTOFF)) for i in range(1, CUTOFF + 1, 1024)]
recent_blocks = [window(CUTOFF - 4095 + i * 1024, CUTOFF - 4095 + (i + 1) * 1024 - 1) for i in range(4)]
review = dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
              scope='Saved matching training records only; excludes partially committed active wave',
              cutoff_episode=CUTOFF, sac_updates_per_group=UPDATES,
              original_easy_files=9999, training_files=len(train), development_files=100,
              complete_training_sweeps=CUTOFF // len(train),
              second_sweep_episodes=CUTOFF % len(train), workers=plan['workers'],
              source_snapshot_plan_progress=plan['completed_episodes'],
              ignored_active_tail=ignored,
              paired_map_hashes_verified=CUTOFF,
              windows=[window(CUTOFF - n + 1, CUTOFF) for n in (512, 1024, 2048, 4096)],
              sequential_blocks=blocks, recent_four_blocks=recent_blocks,
              optimization=optimization,
              behavior_last_1024=behavior(CUTOFF - 1023, CUTOFF),
              behavior_last_1024_both_success=behavior(CUTOFF - 1023, CUTOFF, True),
              latest_development=json.loads((ROOT / 'pipeline.json').read_text()).get('last_review_episode'),
              new_policy_executions=0, new_sac_updates=0, test_policy_executions=0,
              caveats=['Exploratory, changing training weights; not a frozen model development score',
                       'Chronological blocks use different maps; the second sweep has only 89 maps',
                       'One seed; worker counts and rollout wave width changed during training',
                       'Low losses or lower entropy alone do not establish convergence'])
(OUT / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'font.family': font_name, 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none'})


def smooth(y, width):
    return np.convolve(np.asarray(y, dtype=float), np.ones(width) / width, mode='valid')


fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
task_metrics = [('成功率（%）', lambda r: 100 * r['success']),
                ('失败按3000秒计的平均任务耗时（秒）', lambda r: r['capped_time']),
                ('联合覆盖率（%）', lambda r: 100 * r['union_coverage']),
                ('较弱机器人局部覆盖率（%）', lambda r: 100 * min(r['coverage'])),
                ('双方均成功地图上的任务耗时（秒）', None),
                ('平均每机器人训练回报', lambda r: np.mean(r['return_values']))]
both_mask = np.array([all(rollouts[g][i]['success'] for g in GROUPS) for i in range(1, CUTOFF + 1)])
count = np.convolve(both_mask.astype(float), np.ones(512), mode='valid')
for ax, (title, metric) in zip(axes.flat, task_metrics):
    for group, color, label in zip(GROUPS, COLORS, LABELS):
        rows = [rollouts[group][i] for i in range(1, CUTOFF + 1)]
        if metric:
            y = smooth([metric(r) for r in rows], 512)
        else:
            total = np.convolve(np.array([r['time'] for r in rows]) * both_mask, np.ones(512), mode='valid')
            y = total / np.maximum(count, 1)
            y[count < 20] = np.nan
        ax.plot(np.arange(512, CUTOFF + 1), y, color=color, label=label, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel('已保存训练回合')
    ax.axvline(5380, color='#999999', linestyle=':', linewidth=1)
    ax.axvline(9899, color='#777777', linestyle='--', linewidth=1)
    ax.grid(alpha=.18)
axes[0, 0].legend(loc='lower right')
fig.suptitle('原easy训练曲线：两组配对至9988回合｜512回合滑动平均\n点线：启用24进程；虚线：9899张训练图第一遍结束。训练记录不等同于开发集成绩。', fontsize=14)
for suffix in ('png', 'svg'):
    fig.savefig(OUT / ('task_curves.' + suffix), dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
opt_metrics = [('Q损失（Q1/Q2平均，对数轴）', lambda r: (r['q1_loss'] + r['q2_loss']) / 2, True),
               ('策略损失', lambda r: r['policy_loss'], False),
               ('熵温度 alpha（对数轴）', lambda r: r['alpha'], True),
               ('策略熵', lambda r: r['entropy'], False),
               ('WAIT动作概率（%）', lambda r: 100 * r['wait_probability'], False),
               ('Q梯度范数（裁剪前，对数轴）', lambda r: r['q_grad_norm'], True)]
for ax, (title, metric, log) in zip(axes.flat, opt_metrics):
    for group, color, label in zip(GROUPS, COLORS, LABELS):
        rows = [updates[group][i] for i in range(217, UPDATES + 1)]
        y = smooth([metric(r) for r in rows], 2048)
        xs = np.arange(216 + 2048, UPDATES + 1)
        ax.plot(xs[::32], y[::32], color=color, label=label, linewidth=1.6)
    if log:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('SAC更新次数')
    ax.grid(alpha=.18)
axes[0, 0].legend()
fig.suptitle('原easy优化过程｜2048次更新滑动平均\n损失与熵用于诊断数值状态，不能单独判定任务收敛。', fontsize=14)
for suffix in ('png', 'svg'):
    fig.savefig(OUT / ('optimization_curves.' + suffix), dpi=160)
plt.close(fig)
print(json.dumps(dict(windows=review['windows'], recent_four_blocks=recent_blocks,
                     sequential_success=[dict(first=b['first_episode'], last=b['last_episode'],
                         rates={g: round(100 * b['groups'][g]['success_rate'], 2) for g in GROUPS}) for b in blocks],
                     optimization={g: dict(nonfinite=optimization[g]['all_updates_nonfinite'],
                         last_4096={k: optimization[g]['last_4096'][k] for k in keys},
                         q_grad_above_clip=optimization[g]['last_4096_q_grad_above_clip']) for g in GROUPS}), ensure_ascii=False))
