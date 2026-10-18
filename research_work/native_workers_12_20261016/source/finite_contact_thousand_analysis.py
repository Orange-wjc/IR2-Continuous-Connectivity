#!/usr/bin/env python3
"""Read existing experiment evidence; never start training or execute a policy.

--live audits a committed log prefix without reading the large active replay.
--watch waits for the original coordinator's successful terminal audit, then
writes supplemental results and compact evidence. The original foreground
script alone owns training, evaluation, status.json and report/summary.json.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-finite-contact-matplotlib')
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.maps import training_map_index
from finite_contact_stability import audit_record
from finite_contact_multimap_joint import identical, weight_change
from finite_contact_identity import active_identity

ROOT = Path(__file__).resolve().parent
GROUPS = ('AB_learned', 'A_rule', 'A_fifo')
LABELS = {'AB_learned': '学习选包', 'A_rule': '规则选包', 'A_fifo': 'FIFO'}


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.analysis.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def iter_rows(path):
    if not Path(path).exists():
        return
    with Path(path).open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'):
                break  # A concurrent append is not a complete record yet.
            if line.strip():
                yield json.loads(line)


def rows(path):
    return list(iter_rows(path))


def identity(root):
    current=active_identity(root)
    before = read(root/'source_identity_before.json')
    for group in GROUPS:
        directory = root/group
        snapshots = [directory/'source'] + [p/'source' for p in directory.glob('resume_*')]
        for snapshot in snapshots:
            for path in snapshot.rglob('*.py'):
                original = ROOT/path.relative_to(snapshot)
                name=str(path.relative_to(snapshot))
                allowed={current.get(name),before.get(name),digest(original) if name not in current else None}
                assert original.is_file() and digest(path) in allowed, 'Runtime dependency lineage changed: '+str(original)
    return current


def failure_detail(record, truth, beliefs):
    free = truth == 255
    union = ((beliefs == 255).any(axis=0)) & free
    worst = int(np.argmin(record['coverage']))
    return {'map_index': record['map']['index'], 'coverage': record['coverage'],
            'union_coverage': record['union_coverage'], 'time': record['time'],
            'direct_task_deficit': ('team_exploration_below_threshold'
                if record['union_coverage'] < .99 else 'local_map_synchronization_below_threshold'),
            'unexplored_free_cells': int(np.count_nonzero(free & ~union)),
            'explored_cells_missing_at_worst_robot': int(np.count_nonzero(union & (beliefs[worst] != 255))),
            'queue_bytes': record['queue_bytes'], 'connected_seconds': record['connected_seconds'],
            'wait_seconds': record['wait_seconds']}


def audit_rollouts(directory, cfg, count, evaluation=False):
    log = 'evaluation' if evaluation else 'training'
    records = rows(directory/(log+'.jsonl'))[:count]
    traces = rows(directory/('evaluation_trajectories.jsonl' if evaluation else 'trajectories.jsonl'))[:count]
    assert len(records) == len(traces) == count
    segments, failures, boundary_contacts = 0, [], 0
    for index, (record, trace) in enumerate(zip(records, traces)):
        expected_split = 'validation' if evaluation else 'train'
        assert record['map'] == trace['map'] and record['map']['split'] == expected_split
        if evaluation:
            assert record['map']['index'] == index
            assert record['evaluation_policy'] == 'greedy' and record['repeat'] == 0
            path = directory/('evaluation_map_%d_repeat_0.npz' % index)
        else:
            assert record['episode'] == trace['episode'] == index+1
            assert record['map']['index'] == training_map_index(cfg, index+1)
            path = directory/('training_map_%03d.npz' % (index+1))
        with np.load(path) as data:
            geometry = audit_record(record, trace, data['truth'], data['beliefs'], cfg, geometry_details=True)
            segments += geometry['movement_segments']
            boundary_contacts += geometry['boundary_contacts']
            if not record['success']:
                failures.append(failure_detail(record, data['truth'], data['beliefs']))
            np.testing.assert_array_equal(data['positions'], trace['trajectory'][-1]['positions'])
        assert abs(trace['trajectory'][-1]['time'] - record['time']) < 1e-6
        if not record['success']:
            assert abs(record['time'] - cfg.horizon_seconds) < 1e-6
    return {'rollouts': count, 'movement_segments': segments, 'wall_crossings': 0,
            'boundary_contacts': boundary_contacts,
            'geometry_model': 'continuous point / pixel interiors',
            'coverage_union_and_per_segment_speed_recomputed': True,
            'radio_and_time_caps_passed': True, 'failures': failures}


def transfer_audit(root):
    transfer = read(root/'checks/foreground_transfer.json')
    archive, retained = Path(transfer['archive']), root/'AB_learned'
    episodes = set(transfer['completed_maps_beyond_checkpoint'])
    result = {'status': 'passed', 'time': now(), 'episodes': sorted(episodes),
              'extra_complete_training_rollouts': len(episodes),
              'extra_sac_updates': transfer['actual_updates_beyond_checkpoint'],
              'excluded_comparison_fields': ['training.jsonl:wall_seconds'], 'comparisons': {}}
    for name, key, select in [
        ('training.jsonl', 'episode', lambda r: r['episode'] in episodes),
        ('trajectories.jsonl', 'episode', lambda r: r['episode'] in episodes),
        ('updates.jsonl', 'updates', lambda r: r['updates'] > transfer['checkpoint_sac_updates'])]:
        old = [r for r in iter_rows(archive/name) if select(r)]
        wanted = {r[key] for r in old}
        new = [r for r in iter_rows(retained/name) if r[key] in wanted]
        assert len(old) == len(new) and old
        for a, b in zip(old, new):
            if name == 'training.jsonl':
                a = {k: v for k, v in a.items() if k != 'wall_seconds'}
                b = {k: v for k, v in b.items() if k != 'wall_seconds'}
            assert a == b, 'Resume evidence mismatch: '+name
        result['comparisons'][name] = {'records': len(old), 'mismatches': 0}
    assert result['comparisons']['updates.jsonl']['records'] == result['extra_sac_updates']
    old_events = (r for r in iter_rows(archive/'communication.jsonl') if r['episode'] in episodes)
    new_events = (r for r in iter_rows(retained/'communication.jsonl') if r['episode'] in episodes)
    sentinel, event_count = object(), 0
    import itertools
    for a, b in itertools.zip_longest(old_events, new_events, fillvalue=sentinel):
        assert a == b, 'Resume communication event mismatch'
        event_count += 1
    result['comparisons']['communication.jsonl'] = {'records': event_count, 'mismatches': 0}
    cfg = Config.from_checkpoint(read(retained/'config.json'))
    traces = {r['episode']: r for r in iter_rows(archive/'trajectories.jsonl') if r['episode'] in episodes}
    segments = 0
    for record in iter_rows(archive/'training.jsonl'):
        ep = record['episode']
        if ep not in episodes:
            continue
        filename = 'training_map_%03d.npz' % ep
        assert digest(archive/filename) == digest(retained/filename)
        with np.load(archive/filename) as data:
            segments += audit_record(record, traces[ep], data['truth'], data['beliefs'], cfg)
    result['extra_complete_rollouts_physical_audit'] = {'rollouts': len(episodes),
        'movement_segments': segments, 'wall_crossings': 0}
    result['partial_interrupted_rollouts'] = rows(archive/'interruptions.jsonl')
    result['partial_rollouts_note'] = 'Incomplete executions retained separately; no complete trajectory/array was saved for them.'
    result['array_files_identical'] = len(episodes)
    write(root/'checks/foreground_transfer_equivalence.json', result)
    return result


def optimizer_audit(initial, final, expected):
    result = {}
    pairs = [('policy_optimizer', initial['policy_optimizer'], final['policy_optimizer']),
             ('alpha_optimizer', initial['alpha_optimizer'], final['alpha_optimizer'])]
    pairs += [('q%d_optimizer' % i, a, b) for i, (a, b) in
              enumerate(zip(initial['q_optimizers'], final['q_optimizers']))]
    for name, a, b in pairs:
        assert a['state'].keys() == b['state'].keys() and a['state']
        before = [int(a['state'][k]['step']) for k in a['state']]
        after = [int(b['state'][k]['step']) for k in a['state']]
        assert all(y-x == expected for x, y in zip(before, after)), name
        result[name] = {'parameters_with_state': len(before),
            'initial_step_range': [min(before), max(before)],
            'final_step_range': [min(after), max(after)], 'increment_per_parameter': expected}
    assert identical(initial['content_optimizer'], final['content_optimizer'])
    result['content_optimizer_unchanged'] = True
    return result


def live(root):
    source = identity(root)
    plan = read(root/'pipeline.json')
    result = {'time': now(), 'scope': 'Committed online training prefix; changing policies, not fixed model validation.',
              'source_identity_matches': len(source), 'groups': {}}
    for group in GROUPS:
        d = root/group
        if not (d/'status.json').exists():
            result['groups'][group] = {'status': 'queued'}
            continue
        status = read(d/'status.json')
        n = status['completed_episodes']
        cfg = Config.from_checkpoint(read(d/'config.json'))
        physical = audit_rollouts(d, cfg, n)
        records = rows(d/'training.jsonl')[:n]
        updates = [r for r in iter_rows(d/'updates.jsonl')
                   if r['updates'] <= plan['source_sac_updates']+n*20]
        assert len(updates) == n*20
        assert [r['updates'] for r in updates] == list(range(8896, 8896+n*20))
        assert all(r['post_episode_sac_updates'] == 20 and r['online_sac_updates'] == 0 for r in records)
        assert all(r['learner_updates'] == 8895+20*r['episode'] and r['content_updates'] == 5000 for r in records)
        for i, update in enumerate(updates):
            assert update['episode'] == i//20+1
            assert update['map_index'] == records[i//20]['map']['index']
            assert all(math.isfinite(v) for v in update.values() if isinstance(v, (int, float)))
        tail = records[-50:]
        result['groups'][group] = {'status': status['status'], 'committed_complete_maps': n,
            'new_sac_updates': len(updates), 'cumulative_sac_updates': 8895+len(updates),
            'online_successes': sum(r['success'] for r in records),
            'last_50_successes': sum(r['success'] for r in tail), 'last_50_count': len(tail),
            'last_50_mean_time': statistics.mean(r['time'] for r in tail) if tail else None,
            'physical_audit': physical}
    result['foreground_transfer'] = transfer_audit(root)
    write(root/'checks/continuation_live_audit.json', result)
    return result


def communication_audit(directory, records, evaluation=False):
    key = 'map_index' if evaluation else 'episode'
    sums = defaultdict(Counter)
    events = 0
    for r in iter_rows(directory/('evaluation_communication.jsonl' if evaluation else 'communication.jsonl')):
        c = sums[r[key]]
        events += 1
        if r['event'] == 'transmit':
            assert r['sender'] in (0, 1) and r['receiver'] == 1-r['sender']
            assert 0 < r['bytes'] <= 256
            c['sent'] += r['bytes']
            c['sender_%d' % r['sender']] += r['bytes']
            if r['kind'] != 0:
                c['control'] += r['bytes']
        elif r['event'] == 'map_delivery':
            c['novel'] += r['new_cells']
            c['duplicate'] += r['duplicate_cells']
        else:
            raise AssertionError('Unknown radio event')
    wanted = set()
    for r in records:
        index = r['map']['index'] if evaluation else r['episode']
        wanted.add(index)
        c = sums[index]
        assert c['sent'] == r['sent_bytes'] and c['control'] == r['control_bytes']
        assert c['novel'] == r['novel_received_cells'] and c['duplicate'] == r['duplicate_received_cells']
        assert all(c['sender_%d' % i] <= 2048*r['connected_seconds']+1e-6 for i in (0, 1))
    assert set(sums) == wanted
    return {'events': events, 'rollouts': len(records), 'transmission_and_delivery_totals_match': True,
            'per_sender_capacity_passed': True}


def paired(a, b):
    assert [r['map'] for r in a] == [r['map'] for r in b]
    # Positive savings mean b is faster than a; include every failed task at 256s.
    differences = [x['time']-y['time'] for x, y in zip(a, b)]
    both = [x['time']-y['time'] for x, y in zip(a, b) if x['success'] and y['success']]
    return {'success_gain_for_second': sum(r['success'] for r in b)-sum(r['success'] for r in a),
            'mean_capped_time_saved_seconds': statistics.mean(differences),
            'median_capped_time_saved_seconds': statistics.median(differences),
            'second_faster_maps': sum(x > 1e-6 for x in differences),
            'second_slower_maps': sum(x < -1e-6 for x in differences),
            'both_successful_maps': len(both),
            'both_successful_mean_time_saved_seconds': statistics.mean(both) if both else None,
            'gained_success_maps': [x['map']['index'] for x,y in zip(a,b) if not x['success'] and y['success']],
            'lost_success_maps': [x['map']['index'] for x,y in zip(a,b) if x['success'] and not y['success']],
            'per_map_time_saved_seconds': differences}


def finalize(root):
    plan = read(root/'pipeline.json')
    assert plan['status'] == 'complete', 'Original training/evaluation/audit pipeline is not complete'
    source = identity(root)
    assert source == read(root/'source_identity_after.json')
    summary = read(root/'report/summary.json')
    assert summary['combined_physical_audit']['rollouts'] == 3060
    common = torch.load(root/'common_start.pth', map_location='cpu')
    result = {'status': 'complete', 'time': now(),
        'scope': 'Post hoc fixed-endpoint evidence; no additional policy execution or model selection.',
        'test_policy_evaluations': 0, 'groups': {}, 'pairs': {},
        'limitations': ['Single adaptation seed sharing prior 100-map movement/head training.',
            'The same 20 development maps were used repeatedly; these are not independent test results.',
            'Communication head and context remain frozen at the 100-map version.',
            '100-map and 1000-map results have different update budgets; map count is not isolated.',
            'Failure classes describe terminal deficits, not causal attribution to individual modules.']}
    development = {}
    for group in GROUPS:
        d = root/group
        assert read(d/'status.json')['status'] == 'complete'
        initial = torch.load(d/'initial.pth', map_location='cpu')
        final = torch.load(d/'final.pth', map_location='cpu')
        assert identical(initial['learner'], common['learner'])
        assert final['episode'] == 1000 and final['learner']['updates'] == 28895
        assert final['learner']['content_updates'] == 5000
        assert all(weight_change(initial['learner'], final['learner'], k) == 0 for k in ('content', 'context_encoder'))
        adam = optimizer_audit(initial['learner'], final['learner'], 20000)
        cfg = Config.from_checkpoint(final['config'])
        train = audit_rollouts(d, cfg, 1000)
        evaluation = audit_rollouts(d/'validation_final', cfg, 20, evaluation=True)
        records = rows(d/'training.jsonl')
        dev = rows(d/'validation_final/evaluation.jsonl')
        assert len(records) == 1000 and len(dev) == 20
        assert all(r['learner_updates'] == 28895 for r in dev)
        development[group] = dev
        radio = {'training': communication_audit(d, records),
                 'development': communication_audit(d/'validation_final', dev, evaluation=True)}
        result['groups'][group] = {'validation_final': summary['groups'][group]['validation_final'],
            'optimizer_audit': adam, 'training_physical_audit': train,
            'development_physical_audit': evaluation, 'communication_event_audit': radio}
        del initial, final
        write(root/'checks/supplemental_analysis_in_progress.json', result)
    for first, second in [('A_fifo', 'AB_learned'), ('A_rule', 'AB_learned'), ('A_fifo', 'A_rule')]:
        result['pairs'][first+'_vs_'+second] = paired(development[first], development[second])
    transfer = transfer_audit(root)
    extra_maps,extra_updates=6,120
    recovery_audits=[]
    expected_archive = 'superseded_1791787994470042341'
    for group in GROUPS:
        for archive in (root/group).glob('superseded_*'):
            if group=='AB_learned' and archive.name==expected_archive:continue
            account=read(archive/'recovery.json')
            ep_ids=set(account['extra_complete_episode_ids'])
            old=[r for r in iter_rows(archive/'training.jsonl') if r['episode'] in ep_ids]
            new={r['episode']:r for r in iter_rows(root/group/'training.jsonl') if r['episode'] in ep_ids}
            for r in old:
                assert {k:v for k,v in r.items() if k!='wall_seconds'}=={k:v for k,v in new[r['episode']].items() if k!='wall_seconds'}
            old_updates=[r for r in iter_rows(archive/'updates.jsonl') if r['updates']>account['restored_sac_updates']]
            new_updates={r['updates']:r for r in iter_rows(root/group/'updates.jsonl')}
            assert all(r==new_updates[r['updates']] for r in old_updates)
            old_traces={r['episode']:r for r in iter_rows(archive/'trajectories.jsonl') if r['episode'] in ep_ids}
            new_traces={r['episode']:r for r in iter_rows(root/group/'trajectories.jsonl') if r['episode'] in ep_ids}
            assert old_traces==new_traces
            cfg=Config.from_checkpoint(read(root/group/'config.json'))
            segments=0
            for r in old:
                name='training_map_%03d.npz'%r['episode']
                with np.load(archive/name) as a,np.load(root/group/name) as b:
                    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                    segments+=audit_record(r,old_traces[r['episode']],a['truth'],a['beliefs'],cfg)
            assert len(old_updates)==account['extra_sac_updates']
            extra_maps+=len(old);extra_updates+=len(old_updates)
            recovery_audits.append({'group':group,'archive':str(archive),'complete_rollouts':len(old),
                                   'extra_updates':len(old_updates),'movement_segments':segments,
                                   'wall_crossings':0,'trajectory_array_and_update_equivalence':True})
    result['compute_accounting'] = {'retained_training_rollouts': 3000, 'development_rollouts': 60,
        'retained_new_sac_updates': 60000, 'known_extra_complete_rollouts': extra_maps,
        'known_extra_sac_updates': extra_updates, 'known_total_actual_new_sac_updates': 60000+extra_updates,
        'partial_interrupted_rollouts': len(transfer['partial_interrupted_rollouts']),
        'extra_complete_rollouts_physical_audit': transfer['extra_complete_rollouts_physical_audit'],
        'additional_recovery_audits':recovery_audits}
    write(root/'report/supplemental_analysis.json', result)
    write_report(root, result, development)
    export(root)
    write(root/'checks/analysis_watcher_status.json', {'status': 'complete', 'time': now(),
          'report': str(root/'report/RESULTS_ANALYSIS.md'), 'test_policy_evaluations': 0})
    return result


def write_report(root, result, development):
    lines = ['# 1000 张简单地图：固定终点结果与验收', '', '生成时间：'+now(), '',
        '三组均完成 1000 回合、新增 20000 次 SAC，累计终点 28895；通信头 5000 次且冻结。每组仅评估同一 20 张开发图，失败按 256 秒计；未运行测试集策略。', '',
        '| 方法 | 成功／20 | 含失败平均秒 | 平均覆盖缺口 |', '| --- | --- | --- | --- |']
    for group in GROUPS:
        v = result['groups'][group]['validation_final']
        lines.append('| %s | %d | %.3f | %.6f |' % (LABELS[group], v['successes'], v['mean_capped_time'], v['mean_coverage_deficit']))
    lines += ['', '## 逐图配对比较', '']
    for key, v in result['pairs'].items():
        first, second = key.split('_vs_')
        lines.append('%s相对%s：成功数变化 %+d；全部 20 图平均节省 %.3f 秒；双方均成功 %d 图，在这些图上平均节省 %s 秒。新增成功图 %s，丢失成功图 %s。' % (
            LABELS[second], LABELS[first], v['success_gain_for_second'], v['mean_capped_time_saved_seconds'],
            v['both_successful_maps'], '%.3f' % v['both_successful_mean_time_saved_seconds'] if v['both_successful_maps'] else '无',
            v['gained_success_maps'], v['lost_success_maps']))
        lines.append('')
    comparison = result['pairs']['A_rule_vs_AB_learned']
    gain, saving = comparison['success_gain_for_second'], comparison['mean_capped_time_saved_seconds']
    if gain >= 0 and saving > 0:
        judgment = '学习选包在本轮单种子开发评估中完成数不低于规则、平均时间更短，提供有限的开发支持。'
    elif gain < 0:
        judgment = '学习选包相对规则损失了完成数，不能只凭时间指标宣称整体更好。'
    else:
        judgment = '学习选包未同时获得完成数不低于规则、平均时间更短的结果；本轮不足以支持稳定优势。'
    lines += ['## 结果边界与下一步', '', judgment, '',
        '本轮没有独立测试、多条完整独立训练链或模块 A 的独立消融；不能据此认定论文贡献成立。失败类型来自终点覆盖状态，不能单独作为算法因果解释。', '',
        '先依据全部开发失败和训练记录判断探索不足或本地同步不足，固定下一轮协议后，再做预算匹配的模块消融和独立训练种子。保留 20 张测试图，不针对开发失败地图编写专用路线，也不因负结果自动追加训练。', '',
        '## 验收与额外计算', '',
        '主执行 3000 条训练＋60 条开发轨迹已重算覆盖、联合覆盖、逐段速度、穿墙、时限与发送方容量，并核对通信事件总量及全部 Adam 参数的真实更新增量。原汇总审计和训练源码身份检查同时通过。', '',
        '前台恢复与内存故障恢复另有 %d 条完整轨迹、%d 次 SAC 重做；轨迹、更新与地图数组等价，保留模型仍只计 60000 次新增更新，已知实际新增执行为 %d 次。未完成的中断回合单列于 foreground_transfer_equivalence.json。' % (result['compute_accounting']['known_extra_complete_rollouts'],result['compute_accounting']['known_extra_sac_updates'],result['compute_accounting']['known_total_actual_new_sac_updates']), '',
        '所有开发失败详情与三组训练失败均保存在 supplemental_analysis.json。原始模型、回放、轨迹继续保存在 experiments 目录；小型证据副本位于 finite_contact/results/thousand_maps_20261012。', '']
    (root/'report/RESULTS_ANALYSIS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for group in GROUPS:
        dev = development[group]
        axes[0].plot(range(20), [r['time'] for r in dev], 'o-', label=group, lw=.8)
        axes[1].plot(range(20), [min(r['coverage']) for r in dev], 'o-', label=group, lw=.8)
    axes[0].set(xlabel='Development map', ylabel='Task time (s); failures = 256', ylim=(0, 265))
    axes[1].set(xlabel='Development map', ylabel='Minimum robot coverage', ylim=(0, 1.02))
    axes[1].axhline(.99, color='black', linestyle='--', lw=.6)
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(root/('report/paired_development.'+suffix), dpi=150)
    plt.close(fig)
    for group in GROUPS:
        failures = [r for r in development[group] if not r['success']]
        if not failures:
            continue
        columns = min(4, len(failures))
        height = math.ceil(len(failures)/columns)
        fig, axes = plt.subplots(height, columns, figsize=(4*columns, 4*height), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, record in zip(axes.flat, failures):
            index = record['map']['index']
            with np.load(root/group/('validation_final/evaluation_map_%d_repeat_0.npz' % index)) as data:
                truth, beliefs = data['truth'], data['beliefs']
                union = (beliefs == 255).any(axis=0)
                worst = int(np.argmin(record['coverage']))
                rgb = np.zeros((*truth.shape, 3))
                rgb[truth == 255] = (.85, .85, .85)
                rgb[(truth == 255) & ~union] = (1, .25, .2)
                rgb[(truth == 255) & union & (beliefs[worst] != 255)] = (1, .7, .1)
                ax.imshow(rgb, origin='upper')
                ax.scatter(data['positions'][:, 0], data['positions'][:, 1], c=['blue', 'purple'], s=20)
            ax.set_title('Map %d; min %.3f, union %.3f' % (index, min(record['coverage']), record['union_coverage']))
        fig.suptitle(group+': red = unexplored; orange = known by peer, missing locally')
        fig.tight_layout()
        fig.savefig(root/('report/%s_development_failures.png' % group), dpi=150)
        plt.close(fig)


def export(root):
    dest = ROOT/'finite_contact/results/thousand_maps_20261012'
    files = {'summary.json': root/'report/summary.json',
        'supplemental_analysis.json': root/'report/supplemental_analysis.json',
        'RESULTS_ANALYSIS.md': root/'report/RESULTS_ANALYSIS.md',
        'pipeline_final.json': root/'pipeline.json',
        'source_identity_after.json': root/'source_identity_after.json',
        'foreground_transfer_equivalence.json': root/'checks/foreground_transfer_equivalence.json',
        'finalize_analysis_source.py': Path(__file__).resolve()}
    for name in ('source_identity_memory_resume.json',):
        if (root/name).exists():files[name]=root/name
    for name in ('memory_recovery_review.json','resource_interruption_20261012.json','memory_recovery_live_equivalence.json'):
        if (root/'checks'/name).exists():files[name]=root/'checks'/name
    for name in ('matched_adaptation.png', 'matched_adaptation.pdf', 'paired_development.png', 'paired_development.pdf'):
        files[name] = root/'report'/name
    for pic in (root/'report').glob('*_development_failures.png'):
        files[pic.name] = pic
    for name, source in files.items():
        shutil.copy2(source, dest/name)
        assert digest(source) == digest(dest/name)
    (dest/'README.md').write_text('# 1000 张简单地图的固定终点证据\n\n'
        '三组固定预算与原汇总审计均完成，补充逐图分析见 [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)。'
        '仅单适应种子、共享预训练链、重复使用的 20 张开发图；通信头沿用 100 图版本且冻结，测试策略执行 0。'
        '全部模型与原始数据保留本机 experiments/thousand_maps_20261012_seed1，本目录只保存小型证据。\n')
    write(dest/'manifest.json', {'scope': 'Compact evidence only; initial and terminal records kept separately.',
        'files': {str(p.relative_to(dest)): digest(p) for p in sorted(dest.rglob('*'))
                  if p.is_file() and p.name != 'manifest.json'}})
    marker = '\n## 本轮固定终点完成后的补充分析\n'
    section = marker + '\n'+now()+'：三组训练、固定终点评估及原汇总审计完成，补充物理／优化器／通信审计通过。结果与局限见 '
    section += '[固定终点结果分析](/home/wyz/wjc_ws/IR2-Continuous-Connectivity/finite_contact/results/thousand_maps_20261012/RESULTS_ANALYSIS.md)。\n'
    for p in (ROOT/'finite_contact/THOUSAND_MAP_EXPERIMENT.md', ROOT/'finite_contact/EXPERIMENT_HANDOFF.md',
              ROOT.parent/'papers/多机器人有限通信探索_实验计划与新对话交接.md'):
        content = p.read_text().split(marker)[0]
        p.write_text(content.rstrip()+'\n'+section)


def watch(root, interval):
    identity(root)
    output = root/'checks/analysis_watcher_status.json'
    last_signature, last_change = None, time.monotonic()
    while True:
        try:
            plan = read(root/'pipeline.json')
            groups = {g: read(root/g/'status.json') if (root/g/'status.json').exists() else {'status': 'queued'} for g in GROUPS}
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(interval)
            continue
        state = {'status': 'waiting_for_original_pipeline', 'time': now(), 'pid': os.getpid(),
                 'pipeline_status': plan['status'], 'groups': groups,
                 'scope': 'Read-only observer; does not launch/stop training or execute policies.'}
        signature = json.dumps(groups, sort_keys=True)
        if signature != last_signature:
            last_signature, last_change = signature, time.monotonic()
        state['seconds_since_last_group_status_change'] = round(time.monotonic()-last_change)
        with (root/'.foreground.lock').open('r') as foreground:
            try:
                fcntl.flock(foreground, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state['foreground_session_lock_held'] = False
                fcntl.flock(foreground, fcntl.LOCK_UN)
            except BlockingIOError:
                state['foreground_session_lock_held'] = True
        if plan['status'] == 'interrupted_or_failed' or any(v['status'] == 'interrupted_or_failed' for v in groups.values()):
            state['status'] = 'training_needs_attention'
        if plan['status'] != 'complete' and (not state['foreground_session_lock_held'] or time.monotonic()-last_change > 1800):
            state['status'] = 'training_needs_attention'
        write(output, state)
        if plan['status'] == 'complete':
            finalize(root)
            return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--finalize', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    assert args.interval > 0
    torch.set_num_threads(1)
    root = args.experiment.resolve()
    with (root/'checks/.analysis.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.live:
                value = live(root)
                print(json.dumps({g: {k:v for k,v in r.items() if k != 'physical_audit'}
                                  for g,r in value['groups'].items()}, ensure_ascii=False))
            elif args.finalize:
                finalize(root)
            else:
                watch(root, args.interval)
        except Exception as exc:
            write(root/'checks/analysis_watcher_status.json', {'status': 'analysis_failed',
                  'time': now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
