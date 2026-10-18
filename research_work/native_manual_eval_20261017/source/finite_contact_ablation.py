#!/usr/bin/env python3
"""Complete the missing shared-start ablations without mutating prior run sources."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-finite-contact-matplotlib')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.memory import load_checkpoint
import finite_contact_multimap_joint as joint
from finite_contact_identity import active_identity
from finite_contact_stability import recover_initialization
from finite_contact_thousand_analysis import (audit_rollouts, communication_audit,
    optimizer_audit, paired, read, rows, write, now, digest)

ROOT = Path(__file__).resolve().parent
NEW = {'B0_fifo': ('B0', 'fifo', True), 'B_learned': ('B', 'learned', True),
       'AB_uncoupled': ('AB', 'learned', False)}
OLD = {'A_fifo': ('A', 'fifo', True), 'A_rule': ('A', 'rule', True),
       'AB_learned': ('AB', 'learned', True)}
ORDER = ['B0_fifo', 'A_fifo', 'B_learned', 'AB_learned', 'A_rule', 'AB_uncoupled']
CONTRASTS = [('B0_fifo', 'A_fifo'), ('B0_fifo', 'B_learned'),
             ('B_learned', 'AB_learned'), ('A_fifo', 'AB_learned'),
             ('AB_uncoupled', 'AB_learned'), ('A_rule', 'AB_learned')]


def group_config(common, group):
    cfg = Config.from_checkpoint(common['config'])
    cfg.variant, cfg.content, cfg.coupled = {**OLD, **NEW}[group]
    cfg.validate()
    return cfg


def install_groups():
    # The imported trainer/report resolves these module globals at call time.
    # No old source file or prior experiment is edited.
    joint.GROUPS = {g: value[:2] for g, value in NEW.items()}
    joint.group_config = group_config
    original = joint.provenance
    def provenance(directory, cfg):
        original(directory, cfg)
        shutil.copy2(__file__, directory/'source'/Path(__file__).name)
    joint.provenance = provenance


def source_identity():
    names = [p.relative_to(ROOT) for p in (ROOT/'finite_contact').rglob('*.py')]
    names += [Path(n) for n in ('finite_contact_ablation.py', 'finite_contact_multimap_joint.py',
        'finite_contact_train.py', 'finite_contact_multimap_content.py', 'finite_contact_multimap_report.py',
        'finite_contact_stability.py', 'finite_contact_identity.py', 'finite_contact_thousand_analysis.py',
        'finite_contact_dataset.py', 'model.py', 'env.py', 'sensor.py', 'parameter.py',
        'graph_generator.py', 'graph.py', 'node.py', 'ss_realistic_model.py',
        'finite_contact/profiles/rtx3090_32gb.json', 'finite_contact/ABLATION_20261013.md')]
    return {str(p): digest(ROOT/p) for p in sorted(set(names))}


def reference_artifacts(reference):
    names = ['pipeline.json', 'common_start.pth', 'source_identity_before.json',
             'source_identity_memory_resume.json', 'source_identity_after.json',
             'checks/memory_recovery_review.json', 'report/summary.json',
             'report/supplemental_analysis.json', 'dataset/manifest.json']
    for group in OLD:
        names += [str(Path(group)/n) for n in ('initial.pth', 'final.pth', 'status.json', 'config.json',
            'validation_final/config.json', 'validation_final/evaluation.jsonl',
            'validation_final/evaluation_trajectories.jsonl', 'validation_final/evaluation_communication.jsonl')]
        names += [str(p.relative_to(reference)) for p in (reference/group/'validation_final').glob('*.npz')]
    return {name: digest(reference/name) for name in names}


def verify(root):
    plan = read(root/'pipeline.json')
    assert source_identity() == read(root/'source_identity_before.json'), 'Frozen ablation source changed'
    assert digest(root/'common_start.pth') == plan['common_sha256']
    reference = Path(plan['reference_experiment'])
    active_identity(reference)
    assert reference_artifacts(reference) == read(root/'reference_artifacts.json'), 'Reference evidence changed'
    return plan


def prepare(root, reference):
    if (root/'pipeline.json').exists() or any(p.name != '.ablation.lock' for p in root.iterdir()):
        raise ValueError('Use a new output directory or --resume')
    active_identity(reference)
    prior = read(reference/'pipeline.json')
    assert prior['status'] == 'complete'
    assert read(reference/'checks/analysis_watcher_status.json')['status'] == 'complete'
    assert read(reference/'report/supplemental_analysis.json')['status'] == 'complete'
    common = torch.load(reference/'common_start.pth', map_location='cpu')
    assert digest(reference/'common_start.pth') == prior['common_sha256']
    cfg = Config.from_checkpoint(common['config'])
    assert cfg.train_maps == 1000 and cfg.validation_maps == cfg.test_maps == 20
    assert cfg.success_sampling == 'pooled' and cfg.compact_replay
    assert prior['new_sac_updates_per_group'] == 20000 and prior['workers'] == 1
    assert common['learner']['updates'] == 8895 and common['learner']['content_updates'] == 5000
    for group in OLD:
        initial = torch.load(reference/group/'initial.pth', map_location='cpu')
        assert joint.identical(initial['learner'], common['learner'])
        assert initial['config'] == group_config(common, group).to_dict()
        assert read(reference/group/'status.json')['status'] == 'complete'
    shutil.copy2(reference/'common_start.pth', root/'common_start.pth')
    shutil.copy2(ROOT/'finite_contact/ABLATION_20261013.md', root/'pre_run_design.md')
    (root/'source').mkdir()
    identity = source_identity()
    for name in identity:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    write(root/'source_identity_before.json', identity)
    write(root/'reference_artifacts.json', reference_artifacts(reference))
    plan = copy.deepcopy(prior)
    for key in ('source_identity_file', 'source_history', 'worker_order'):
        plan.pop(key, None)
    plan.update(scope='Single-seed shared-start 1000-map 2x2 ablation and uncoupled control; three prior endpoints reused.',
        groups=list(NEW), all_comparison_groups=ORDER, worker_order=list(NEW), reference_root=None,
        reference_experiment=str(reference), status='prepared', created=now(),
        reference_policy='Frozen endpoints; no retraining or re-selection of prior three groups.',
        contrasts=[list(p) for p in CONTRASTS], group_definitions={g:list(v) for g,v in {**OLD,**NEW}.items()},
        prior_initialization_bias='Shared movement source was pretrained using A/FIFO, not variant-independent pretraining.')
    write(root/'pipeline.json', plan)
    return verify(root)


def mechanism(records):
    result = {}
    for key in ('total_distance', 'connected_seconds', 'sent_bytes', 'delivered_bytes',
                'control_bytes', 'novel_received_cells', 'duplicate_received_cells', 'coverage_deficit'):
        result[key] = float(np.mean([r[key] for r in records]))
    result['team_wait_seconds'] = float(np.mean([sum(r['wait_seconds']) for r in records]))
    sent = sum(r['sent_bytes'] for r in records)
    result['novel_cells_per_sent_byte'] = sum(r['novel_received_cells'] for r in records)/sent if sent else 0.
    return result


def analyze(root):
    plan = verify(root)
    reference = Path(plan['reference_experiment'])
    common = torch.load(root/'common_start.pth', map_location='cpu')
    # Full cache, map schedule, initial Adam, physical and update checks for new groups.
    joint.report(root, plan)
    new_summary = read(root/'report/summary.json')
    prior_summary = read(reference/'report/summary.json')
    prior_audit = read(reference/'report/supplemental_analysis.json')
    result = {'status':'complete', 'time':now(), 'scope':plan['scope'], 'groups':{}, 'pairs':{},
        'test_policy_evaluations':0, 'source_identity_checked':True,
        'limitations':['Single adaptation seed; shared A/FIFO-pretrained movement and shared communication head.',
            'Repeatedly used 20 development maps; no independent test policy evaluation.',
            'Head and context remain frozen at the 100-map version.',
            'A source-prior bias remains; this is adaptation ablation, not independent from-scratch comparison.',
            'Map-level effects do not provide independent training-seed replication or causal proof.']}
    development = {}
    for group in ORDER:
        d = (root if group in NEW else reference)/group
        final = torch.load(d/'final.pth', map_location='cpu')
        initial = torch.load(d/'initial.pth', map_location='cpu')
        cfg = group_config(common, group)
        assert initial['config'] == final['config'] == cfg.to_dict()
        assert joint.identical(initial['learner'], common['learner'])
        assert final['episode'] == 1000 and final['learner']['updates'] == 28895
        assert final['learner']['content_updates'] == 5000
        assert all(joint.weight_change(initial['learner'],final['learner'],k)==0 for k in ('content','context_encoder'))
        dev = rows(d/'validation_final/evaluation.jsonl')
        assert len(dev) == 20 and all(r['learner_updates']==28895 and not r['interrupted'] for r in dev)
        assert [r['map']['index'] for r in dev] == list(range(20))
        development[group] = dev
        if group in NEW:
            train = rows(d/'training.jsonl')
            audit = {'optimizer_audit':optimizer_audit(initial['learner'],final['learner'],20000),
                'training_physical_audit':audit_rollouts(d,cfg,1000),
                'development_physical_audit':audit_rollouts(d/'validation_final',cfg,20,True),
                'communication_event_audit':{'training':communication_audit(d,train),
                    'development':communication_audit(d/'validation_final',dev,True)}}
            metric = new_summary['groups'][group]
        else:
            audit = prior_audit['groups'][group]
            metric = prior_summary['groups'][group]
        result['groups'][group] = {'directory':str(d), 'reused_endpoint':group in OLD,
            'config_switches':list({**OLD,**NEW}[group]), 'summary':metric,
            'audit':audit, 'mechanism':mechanism(dev), 'status':read(d/'status.json')}
        write(root/'checks/analysis_in_progress.json', result)
    for a,b in CONTRASTS:
        result['pairs'][a+'_vs_'+b] = paired(development[a], development[b])
    effects = [a['time']+b['time']-base['time']-ab['time'] for base,a,b,ab in zip(
        development['B0_fifo'],development['A_fifo'],development['B_learned'],development['AB_learned'])]
    result['factorial_interaction'] = {'extra_time_saved_seconds':float(np.mean(effects)),
        'per_map_extra_time_saved_seconds':effects,
        'definition':'T_A + T_B - T_B0 - T_AB; positive is complementary time savings; descriptive only.'}
    result['compute_accounting'] = {'new_retained_training_rollouts':3000,'new_retained_sac_updates':60000,
        'new_development_rollouts':60, 'reused_prior_training_rollouts':3000, 'reused_prior_sac_updates':60000,
        'recovery_archives':[]}
    for group in NEW:
        for archive in sorted((root/group).glob('superseded_*')):
            if (archive/'recovery.json').exists():
                result['compute_accounting']['recovery_archives'].append({'group':group,**read(archive/'recovery.json')})
    result['combined_physical_audit'] = {'rollouts':6120, 'wall_crossings':0,
        'movement_segments':sum(v['audit'][part]['movement_segments'] for v in result['groups'].values()
                               for part in ('training_physical_audit','development_physical_audit'))}
    verify(root)
    write(root/'source_identity_after.json', source_identity())
    write(root/'report/ablation_summary.json', result)
    render(root,result,development)
    return result


def render(root, result, development):
    output=root/'report'
    lines=['# 1000 图共享起点模块消融结果','', '完成时间：'+result['time'],'',
        '三个新增组均完整训练 1000 回合／新增 20000 次 SAC；另外三组使用已冻结的同预算终点。测试策略执行 0。','',
        '| 组 | 来源 | 成功／20 | 含失败平均秒 | 平均覆盖缺口 |','| --- | --- | --- | --- | --- |']
    for g in ORDER:
        v=result['groups'][g]; m=v['summary']['validation_final']
        lines.append('| %s | %s | %d | %.3f | %.6f |'%(g,'上轮冻结' if v['reused_endpoint'] else '本轮新增',m['successes'],m['mean_capped_time'],m['mean_coverage_deficit']))
    lines+=['','失败按 256 秒计，不能仅比较成功任务的时间。','',
        '## 事前固定的配对比较','',
        '| 比较（后者相对前者） | 成功数变化 | 平均节省秒 | 更快／更慢图 | 双方成功图数 | 双方成功平均节省秒 |',
        '| --- | --- | --- | --- | --- | --- |']
    for name,v in result['pairs'].items():
        both=v['both_successful_mean_time_saved_seconds']
        lines.append('| %s | %+d | %.3f | %d／%d | %d | %s |'%(name,v['success_gain_for_second'],
            v['mean_capped_time_saved_seconds'],v['second_faster_maps'],v['second_slower_maps'],
            v['both_successful_maps'],'%.3f'%both if both is not None else '无'))
    lines+=['','2×2 额外时间收益（T_A + T_B − T_B0 − T_AB）：%.3f 秒。正数表示互补时间收益，负数表示未获得超过单项收益之和的效果；单种子开发值不代表统计显著性。'%result['factorial_interaction']['extra_time_saved_seconds'],
        '', '## 机制与失败','', '| 组 | 全队等待秒 | 接触秒 | 总路径 | 每发送字节新增接收格 | 失败地图及终点未达标项 |',
        '| --- | --- | --- | --- | --- | --- |']
    for g in ORDER:
        v=result['groups'][g]; m=v['mechanism']; failures=v['audit']['development_physical_audit']['failures']
        lines.append('| %s | %.3f | %.3f | %.3f | %.6f | %s |'%(g,m['team_wait_seconds'],m['connected_seconds'],
            m['total_distance'],m['novel_cells_per_sent_byte'],', '.join('%d:%s'%(f['map_index'],f['direct_task_deficit']) for f in failures) or '无'))
    lines+=['','全部逐图差值、训练失败与开发失败状态、优化器／物理／通信审计见 ablation_summary.json。失败类型只描述终点，不能作为单个模块的因果证明。','',
        '## 结论边界','',
        '共享 A/FIFO 预训练起点和同一冻结通信头，仅一个适应种子；开发图已重复使用。此结果用于判断下一轮值得验证哪些模块，不替代完整独立多种子链或留出测试。',
        '', '本轮到此结束，不自动启动下一轮训练。是否进入多种子与更广通信条件，结合完成率、效率和消融一致性向用户汇报后决定。','']
    (output/'RESULTS_ANALYSIS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    colors=['gray','tab:orange','tab:green','tab:blue','tab:brown','tab:purple']
    for ax,key,title in [(axes[0],'successes','Successes on 20 development maps'),
                         (axes[1],'mean_capped_time','Mean task time; failures = 256 s')]:
        values=[result['groups'][g]['summary']['validation_final'][key] for g in ORDER]
        bars=ax.bar(ORDER,values,color=colors)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value,'%.2f'%value,ha='center',va='bottom',fontsize=8)
        ax.set_title(title);ax.tick_params(axis='x',rotation=40);ax.set_ylim(0,22 if key=='successes' else 280)
    for g,color in zip(ORDER,colors):axes[2].plot(range(20),[r['time'] for r in development[g]],'o-',lw=.7,ms=3,label=g,color=color)
    axes[2].set(xlabel='Development map',ylabel='Task time (s)',ylim=(0,265));axes[2].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/('ablation_results.'+ext),dpi=160)
    plt.close(fig)
    dest=ROOT/'finite_contact/results/ablation_20261013'
    dest.mkdir(parents=True,exist_ok=True)
    for p in [output/'ablation_summary.json',output/'RESULTS_ANALYSIS.md',output/'ablation_results.png',
              output/'ablation_results.pdf',root/'pre_run_design.md',root/'source_identity_before.json',
              root/'source_identity_after.json',root/'reference_artifacts.json']:
        shutil.copy2(p,dest/p.name)
    write(dest/'manifest.json',{'files':{p.name:digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='manifest.json'}})


def execute(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lockname='.worker.lock' if args.stage=='worker' else '.ablation.lock'
    with (root/lockname).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='worker':
            assert read(root/'coordinator.json')['pid']==os.getppid(), 'Launch workers through the coordinator'
            plan=verify(root)
            install_groups()
            # Workers are launched under the coordinator lock; do not relock it.
            joint.train_group(root,plan,args.group,args.device,args.checkpoint)
            verify(root)
            return
        plan=verify(root) if args.resume or args.stage=='report' else prepare(root,args.reference.resolve())
        install_groups()
        if args.stage=='report':
            analyze(root);return
        plan['status']='running';plan['started']=now();write(root/'pipeline.json',plan)
        write(root/'coordinator.json',{'pid':os.getpid(),'time':now(),'device':args.device})
        try:
            for group in NEW:
                d=root/group
                if (d/'status.json').exists() and read(d/'status.json')['status']=='complete':continue
                if args.resume:recover_initialization(d)
                command=[sys.executable,str(Path(__file__).resolve()),'--stage','worker','--output',str(root),
                         '--group',group,'--device',args.device]
                if (d/'latest.pth').exists():command+=['--checkpoint',str(d/'latest.pth')]
                with (root/(group+'.log')).open('a') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        if child.wait()!=0:raise RuntimeError('Ablation worker failed: '+group)
                    finally:
                        if child.poll() is None:
                            child.send_signal(signal.SIGTERM)
                            child.wait()
            analyze(root)
            plan.update(status='complete',completed=now())
        except BaseException:
            plan.update(status='interrupted_or_failed',stopped=now());raise
        finally:write(root/'pipeline.json',plan)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'experiments/ablation_20261013_seed1')
    p.add_argument('--reference',type=Path,default=ROOT/'experiments/thousand_maps_20261012_seed1')
    p.add_argument('--stage',choices=('all','worker','report'),default='all')
    p.add_argument('--group',choices=list(NEW))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args()
    if Path(sys.prefix)!=Path('/home/wyz/anaconda3/envs/ir2_train'):p.error('Use ir2_train')
    if args.stage=='worker' and not args.group:p.error('Worker requires group')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    def stop(signum,frame):raise KeyboardInterrupt('Ablation interrupted')
    if args.stage!='worker':
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
    execute(args)


if __name__=='__main__':main()
