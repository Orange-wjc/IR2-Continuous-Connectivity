"""Offline endpoint review for the bounded repair pilot; no environment or learning."""
import gzip
import json
from pathlib import Path
import numpy as np


def analyze(root, point, engineering_only=False):
    from native_driver import read, write, now, identity
    from finite_contact.memory import load_checkpoint
    root=Path(root);groups=('IR2_FIFO','AB_learned');rows={};audit_rows=[]
    for group in groups:
        directory=root/group/('validation_%08d'%point)
        rows[group]=read(directory/'metrics.json')
        for p in sorted((directory/'episodes').glob('*.json.gz')):
            with gzip.open(p,'rt') as handle:data=json.load(handle)
            audit_rows.append(data['audit'])
    a,b=(rows[g] for g in groups)
    assert len(a)==len(b)==(2 if engineering_only else 100)
    assert [r['map']['file_sha256'] for r in a]==[r['map']['file_sha256'] for r in b]
    assert all(r['map']['split']==('train' if engineering_only else 'validation') for r in a+b)
    common=[(x,y) for x,y in zip(a,b) if x['success'] and y['success']]
    deltas=np.array([y['capped_time']-x['capped_time'] for x,y in zip(a,b)])
    common_delta=np.array([y['time']-x['time'] for x,y in common])
    rng=np.random.RandomState(20261017)
    bootstrap=deltas[rng.randint(len(a),size=(10000,len(a)))].mean(axis=1)
    metrics={}
    for g in groups:
        rr=rows[g]
        metrics[g]=dict(maps=len(rr),successes=sum(x['success'] for x in rr),
            mean_capped_time=float(np.mean([x['capped_time'] for x in rr])),
            mean_union_coverage=float(np.mean([x['union_coverage'] for x in rr])),
            mean_min_local_coverage=float(np.mean([min(x['coverage']) for x in rr])),
            common_success_mean_time=float(np.mean([x['time'] for x,y in common])) if common and g==groups[0]
                else (float(np.mean([y['time'] for x,y in common])) if common else None))
    checkpoints={};engineering_checks={}
    common_state=load_checkpoint(root/'common_start.pth');source_updates=common_state['learner']['updates']
    for group in groups:
        state=load_checkpoint(root/group/'latest.pth')
        start=load_checkpoint(root/group/'initial.pth')
        checkpoints[group]=dict(episode=state['episode'],sac_updates=state['learner']['updates'],
            adaptation_sac_updates=state['learner']['updates']-source_updates,
            head_updates=state['learner']['content_updates'],
            adaptation_imitation_updates=state['learner']['imitation_updates']-start['learner']['imitation_updates'],
            success_episodes=len(state['success_episodes']),replay_size=len(state['replay']),
            log_alpha=float(state['learner']['log_alpha'].item()))
        engineering_checks[group+'_training_budget']=(state['episode']==point and
            state['learner']['updates']-source_updates==point*(2 if engineering_only else 20))
        engineering_checks[group+'_successful_imitation_active']=(checkpoints[group]['adaptation_imitation_updates']>0)
        engineering_checks[group+'_bounded_alpha']=(-8<=checkpoints[group]['log_alpha']<=2)
        engineering_checks[group+'_head_and_context_frozen']=all(
            all(np.array_equal(v.cpu().numpy(),common_state['learner'][key][k].cpu().numpy())
                for k,v in state['learner'][key].items()) for key in ('context_encoder','content'))
        del state,start
    engineering_checks.update(source_identity_unchanged=identity()==read(root/'source_identity.json'),
        all_geometry_radio_audits_passed=all(x['wall_crossings']==0 and x['radio_totals_and_capacity_passed'] for x in audit_rows),
        full_evaluation=len(audit_rows)==len(a)*2,
        shared_sac_budget=source_updates==(4 if engineering_only else 9000),
        head_budget=common_state['learner']['content_updates']==(4 if engineering_only else 5000))
    # Engineering preflight may have no completed exploration episode; its SIL
    # branch is tested separately on a completed self-generated fixture.
    gate=(not engineering_only and all(engineering_checks.values()) and
          metrics[groups[1]]['successes']>=metrics[groups[0]]['successes'] and
          float(deltas.mean())<0 and len(common)>0 and
          float(common_delta.mean())<0 and float(np.median(common_delta))<0)
    report=dict(time=now(),scope='engineering_preflight' if engineering_only else '100-map native easy repair pilot; fixed 100-map development endpoint',
        easy_episodes_per_method=point,groups=metrics,common_success_maps=len(common),
        ab_only_success=sum(y['success'] and not x['success'] for x,y in zip(a,b)),
        fifo_only_success=sum(x['success'] and not y['success'] for x,y in zip(a,b)),
        common_ab_faster=int((common_delta<0).sum()),common_ab_slower=int((common_delta>0).sum()),
        mean_capped_time_delta_ab_minus_fifo=float(deltas.mean()),
        paired_bootstrap_95ci_capped_delta=np.percentile(bootstrap,[2.5,97.5]).tolist(),
        common_mean_time_delta=float(common_delta.mean()) if common else None,
        common_median_time_delta=float(np.median(common_delta)) if common else None,
        engineering_checks=engineering_checks,checkpoints=checkpoints,gate_passed=bool(gate),
        interpretation='One-seed exploratory pilot, not a confirmed generalization or causal attribution; held-out development has been inspected before. Original test remains unused.',
        next_action='Review promising pilot before any new authorized large-scale run' if gate else 'Keep large-scale/difficult training stopped; analyze failure and learning budget',
        test_policy_evaluations=0,large_scale_started=False)
    out=root/'analysis';out.mkdir(exist_ok=True)
    write(out/'comparison.json',report)
    write(out/'paired_maps.json',[dict(map=x['map']['source_file'],fifo=x,ab=y,
        capped_delta=y['capped_time']-x['capped_time']) for x,y in zip(a,b)])
    title='工程预检（不能用于效果结论）' if engineering_only else '100张原easy训练图修复试验'
    text=f'''# {title}

结束时间：{report['time']}。单种子101，固定原图，端点比较；原正式10060回合结果保留。

|开发评估|IR2_FIFO|AB_learned|
|---|---:|---:|
|成功图数|{metrics[groups[0]]['successes']}/{len(a)}|{metrics[groups[1]]['successes']}/{len(a)}|
|失败罚时均值/秒|{metrics[groups[0]]['mean_capped_time']:.3f}|{metrics[groups[1]]['mean_capped_time']:.3f}|
|共同成功实际耗时/秒|{metrics[groups[0]]['common_success_mean_time']}|{metrics[groups[1]]['common_success_mean_time']}|

共同成功{len(common)}图，AB快{report['common_ab_faster']}图、慢{report['common_ab_slower']}图。
AB减基线的失败罚时均值差{deltas.mean():.3f}秒，配对bootstrap 95%区间{report['paired_bootstrap_95ci_capped_delta']}。
共同成功耗时差均值{report['common_mean_time_delta']}秒，中位数{report['common_median_time_delta']}秒。

预先约定的小试筛选条件通过：**{bool(gate)}**。条件为完整审计、相同训练预算、模仿确实更新、成功图数不下降、失败罚时均值下降、共同成功的平均及中位耗时均下降。单种子小试不能直接证明普遍有效，区间用于显示不确定性；不根据结果挑检查点。未开启大规模或复杂地图训练，未使用原测试集。

检查点与逐项工程检查见[comparison.json](analysis/comparison.json)。完整逐图差值见[paired_maps.json](analysis/paired_maps.json)。本轮将训练器配方、前缀和头采样一起恢复，不足以定位单个机制的因果作用。100张指训练图的不同文件数：共享移动100条、头采样100条、两组适应各100条，另有开发各100条；不是总共只执行100条轨迹。
'''
    (root/'RESULTS.md').write_text(text)
    if not engineering_only:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(12,3.7))
        colors=['#566b89','#da7e30'];names=['FIFO','AB']
        ax[0].bar(names,[metrics[g]['successes'] for g in groups],color=colors);ax[0].set_ylim(0,100);ax[0].set_title('Development successes / 100')
        ax[1].bar(names,[metrics[g]['mean_capped_time'] for g in groups],color=colors);ax[1].set_title('Failure-capped mean time (s)')
        if common:
            ax[2].scatter([x['time'] for x,y in common],[y['time'] for x,y in common],s=16,alpha=.65)
            limit=max(max(x['time'],y['time']) for x,y in common)*1.05
            ax[2].plot([0,limit],[0,limit],'--',color='gray');ax[2].set_xlabel('FIFO time (s)');ax[2].set_ylabel('AB time (s)')
        ax[2].set_title('Common successes; below line = AB faster')
        fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    return report
