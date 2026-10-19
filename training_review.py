"""Cheap periodic training-curve review; no evaluation, inference or control.

Checks run synchronously on completed episodes, so terminal closure/resume
does not depend on an assistant staying online. Flags are descriptive
heuristics, not statistical significance or proof of convergence.
"""
from pathlib import Path
import argparse
import json
import math

from baseline_support import write_json


def aggregate(rows):
    if not rows:
        return None
    metric = lambda key: sum(r['metrics'][key] for r in rows)/len(rows)
    result = dict(episodes=len(rows), success_percent=100*sum(bool(r['task_success']) for r in rows)/len(rows),
                mean_local_coverage_percent=100*metric('explored_rate'),
                mean_wait_actions=metric('contact_wait_actions'),
                mean_travel_distance=metric('travel_dist'))
    for key in ('contact_min_local_coverage','contact_union_coverage','contact_positive_empty_waits',
                'contact_reward_rendezvous','contact_reward_delivery'):
        values=[r['metrics'][key] for r in rows if key in r['metrics']]
        if values:
            result[key]=sum(values)/len(values)
    return result


def summarize(rows, maps, through, interval=300):
    valid = {}
    duplicate = 0
    invalid = 0
    for r in rows:
        if not r['valid_rollout']:
            invalid += 1
            continue
        step = r['completed_episodes']
        if step <= through:
            duplicate += int(step in valid)
            valid[step] = r
    ordered = [valid[k] for k in sorted(valid)]
    current = [r for r in ordered if through-interval < r['completed_episodes'] <= through]
    previous = [r for r in ordered if through-2*interval < r['completed_episodes'] <= through-interval]
    a, b = aggregate(current), aggregate(previous)
    if a is None:
        raise ValueError('No valid training rows for the requested review')
    full_comparison = len(current) == len(previous) == interval
    delta = a['success_percent']-b['success_percent'] if full_comparison else None
    early, late = aggregate(current[:100]), aggregate(current[-100:])
    within_delta = late['success_percent']-early['success_percent'] if len(current)>=200 else None
    trend = ('insufficient_history' if delta is None else
             'rising' if delta >= 3. else 'falling' if delta <= -3. else 'no_clear_rise')
    per_map = {}
    flags = []
    if len(current) != interval or len(ordered) != through or duplicate:
        flags.append('progress_history_requires_audit')
    for name in maps:
        visits = [r for r in ordered if r['map'] == name]
        per_map[name] = dict(visits=len(visits),total_successes=sum(bool(r['task_success']) for r in visits),
                             first20=aggregate(visits[:20]),last20=aggregate(visits[-20:]),
                             current_block=aggregate([r for r in current if r['map'] == name]),
                             previous_block=aggregate([r for r in previous if r['map'] == name]))
        info = per_map[name]
        if len(visits) >= 20 and info['total_successes'] == 0:
            flags.append('never_successful:'+name)
        if len(visits) >= 40:
            f, l = info['first20'], info['last20']
            if l['mean_local_coverage_percent'] < f['mean_local_coverage_percent']-2.:
                flags.append('coverage_decline:'+name)
            if l['mean_wait_actions'] > f['mean_wait_actions']+5.:
                flags.append('more_waiting:'+name)
    if not all(math.isfinite(v) for r in current for v in r['metrics'].values()):
        flags.append('non_finite_metrics')
    if any(r['metrics'].get('contact_positive_empty_waits',0)>0 for r in current):
        flags.append('positive_reward_idle_wait')
    invalid_current = sum(not r['valid_rollout'] and through-interval < r['completed_episodes'] <= through
                          for r in rows)
    if invalid_current:
        flags.append('invalid_rollouts')
    if a['success_percent'] >= 95. and all(p['last20'] and p['last20']['success_percent'] >= 90.
                                           for p in per_map.values()):
        trend = 'high_success_window'
    if b and a['mean_local_coverage_percent'] < b['mean_local_coverage_percent']-1.:
        flags.append('overall_coverage_decline')
    attention = (trend in ('falling', 'no_clear_rise') or bool(flags))
    return dict(completed_episodes=through, interval=interval,
                current_block=a,previous_block=b,last100=aggregate(ordered[-100:]),
                full_comparison=full_comparison,success_change_percentage_points=delta,
                within_block_first100=early,within_block_last100=late,
                within_block_change_percentage_points=within_delta,
                trend=trend,needs_diagnosis=attention,flags=flags,per_map=per_map,
                invalid_rows_in_history=invalid,duplicate_completed_steps=duplicate,
                invalid_rows_current_block=invalid_current,
                last_sac_updates=ordered[-1]['training_updates'],
                policy='Numeric review only. With IR2_AUTO_REVIEW=1 the driver may save/pause; this module never changes weights or hyperparameters.',
                interpretation='Training samples, not held-out performance. +/-3 percentage points is a descriptive threshold, not significance. Short fluctuations are allowed.')


def plot_review(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    valid = {}
    for r in rows:
        if r['valid_rollout']:
            valid[r['completed_episodes']] = r
    data = [valid[k] for k in sorted(valid)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, key, title, scale in [(axes[0,0], None, 'Rolling 100 task success (%)',100),
            (axes[0,1], 'explored_rate','Rolling 100 mean individual coverage (%)',100),
            (axes[1,0], 'contact_wait_actions','Rolling 100 WAIT actions per episode',1)]:
        y = np.array([r['metrics'][key] if key else r['task_success'] for r in data],float)
        window = min(100,len(y))
        ax.plot([r['completed_episodes'] for r in data][window-1:],
                np.convolve(y,np.ones(window)/window,'valid')*scale)
        ax.set_title(title);ax.set_xlabel('Completed episodes');ax.grid(alpha=.25)
    names = sorted({r['map'] for r in data}, reverse=True)
    values = [100*np.mean([r['task_success'] for r in data if r['map']==m][-20:]) for m in names]
    axes[1,1].bar([m[4:8] for m in names], values)
    axes[1,1].set_title('Per-map last 20 visits success (%)');axes[1,1].set_ylim(0,105)
    axes[1,1].tick_params(axis='x',labelrotation=45)
    fig.suptitle('Periodic training review (not test performance)')
    fig.savefig(directory/'curves.png',dpi=140);plt.close(fig)


class TrainingReview:
    def __init__(self, directory, maps, writer=None, interval=300):
        self.directory = Path(directory)
        self.maps, self.writer, self.interval = maps, writer, interval

    def maybe_record(self, completed):
        if completed <= 0 or completed % self.interval:
            return None
        return self.record(completed)

    def record(self, through):
        rows = [json.loads(x) for x in (self.directory/'progress.jsonl').read_text().splitlines() if x.strip()]
        rows = [r for r in rows if r['completed_episodes'] <= through]
        report = summarize(rows, self.maps, through, self.interval)
        target = self.directory/'reviews'/('episode_{:06d}'.format(through))
        target.mkdir(parents=True, exist_ok=True)
        try:
            plot_review(rows,target)
        except Exception as exc:
            report['plot_error'] = repr(exc)
        write_json(target/'review.json',report)
        write_json(self.directory/'reviews/latest.json',report)
        labels=dict(insufficient_history='首个300回合窗口，尚无完整前窗可比较',
                    rising='成功率有上升',falling='成功率下降',no_clear_rise='尚无明确上升',
                    high_success_window='高成功率窗口；仍需后续验证稳定性')
        lines=['# 每300有效回合训练复查', '',
               '完成回合：{}；判断：{}。'.format(through,labels[report['trend']]),
               '本窗口成功率：{:.2f}%；最近100：{:.2f}%。'.format(
                   report['current_block']['success_percent'],report['last100']['success_percent']),
               '前后窗口差值：{}个百分点；这是描述统计，不代表显著性或收敛。'.format(
                   report['success_change_percentage_points']), '',
               '本窗口最初100回合→最后100回合成功率：{:.2f}%→{:.2f}%，变化{}个百分点。'.format(
                   report['within_block_first100']['success_percent'],
                   report['within_block_last100']['success_percent'],
                   report['within_block_change_percentage_points']), '',
               '| 地图 | 累计访问 | 最近20次成功率 | 最近20次平均本地覆盖 | 最近20次WAIT |',
               '|---|---:|---:|---:|---:|']
        for name,info in report['per_map'].items():
            m=info['last20']
            if m:
                lines.append('| {} | {} | {:.1f}% | {:.2f}% | {:.2f} |'.format(name,info['visits'],
                    m['success_percent'],m['mean_local_coverage_percent'],m['mean_wait_actions']))
        lines+=['','待核查标记：'+(', '.join(report['flags']) or '无'),
                '覆盖率为机器人本地覆盖均值，非并集。检查不运行测试回合、不改参；IR2_AUTO_REVIEW=1时driver可据门槛保存停训。']
        (target/'README.md').write_text('\n'.join(lines)+'\n')
        if self.writer:
            self.writer.add_scalar('Review/Success Rate 300 Percent',report['current_block']['success_percent'],through)
            self.writer.add_scalar('Review/Needs Diagnosis',int(report['needs_diagnosis']),through)
            if report['success_change_percentage_points'] is not None:
                self.writer.add_scalar('Review/Success Change Percentage Points',report['success_change_percentage_points'],through)
            self.writer.flush()
        print(json.dumps(dict(stage='training_review',completed=through,trend=report['trend'],
            success_percent=report['current_block']['success_percent'],
            success_change_percentage_points=report['success_change_percentage_points'],
            flags=report['flags'],report=str(target/'README.md')),ensure_ascii=False),flush=True)
        return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--through',type=int,required=True)
    args=parser.parse_args()
    manifest=json.loads((Path(args.experiment)/'map_manifest.json').read_text())
    TrainingReview(args.experiment,[m['name'] for m in manifest['maps']]).record(args.through)
