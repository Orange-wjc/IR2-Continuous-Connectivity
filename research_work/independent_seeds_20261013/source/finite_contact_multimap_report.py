#!/usr/bin/env python3
"""Audit one completed movement pilot and compare the same 20 development maps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-finite-contact-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.draw import line
from finite_contact.config import Config
from finite_contact.maps import configured_map


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit_segments(truth, trajectory, return_details=False):
    """Audit actual continuous point trajectories against occupied cell interiors.

    Grid coordinates denote cell centers. Rounding a truncated final edge first
    changes its geometry and can rasterize an adjacent occupied cell. Zero-length
    boundary/corner contacts are recorded separately, not interior penetration.
    This point model does not certify clearance for finite-radius robots.
    """
    positions = np.asarray([s['positions'] for s in trajectory], dtype=np.float64)
    assert np.isfinite(positions).all(), 'Non-finite trajectory'
    limit = np.array([truth.shape[1] - 1, truth.shape[0] - 1])
    assert np.all(positions >= 0) and np.all(positions <= limit), 'Trajectory outside map'
    segments, boundary_contacts = 0, 0
    for a, b in zip(positions[:-1], positions[1:]):
        for start, end in zip(a, b):
            delta = end - start
            low = np.maximum(0, np.ceil(np.minimum(start, end) - .5).astype(int))
            high = np.minimum(limit, np.floor(np.maximum(start, end) + .5).astype(int))
            yy, xx = np.nonzero(truth[low[1]:high[1]+1, low[0]:high[0]+1] != 255)
            if len(xx):
                centers = np.column_stack((xx + low[0], yy + low[1])).astype(float)
                lower, upper = centers - .5, centers + .5
                enter, leave = np.zeros(len(xx)), np.ones(len(xx))
                interior = np.ones(len(xx), dtype=bool)
                for axis in (0, 1):
                    if delta[axis] == 0:
                        interior &= ((start[axis] > lower[:, axis] + 1e-10)
                                     & (start[axis] < upper[:, axis] - 1e-10))
                    else:
                        x = (lower[:, axis] - start[axis]) / delta[axis]
                        y = (upper[:, axis] - start[axis]) / delta[axis]
                        enter = np.maximum(enter, np.minimum(x, y))
                        leave = np.minimum(leave, np.maximum(x, y))
                penetration = interior & (leave - enter > 1e-10)
                if np.all(delta == 0):
                    penetration = interior
                assert not penetration.any(), 'Wall crossing: occupied cell interior'
                boundary_contacts += int(np.count_nonzero((leave >= enter - 1e-10) & ~penetration))
            segments += 1
    details = {'movement_segments': segments, 'wall_crossings': 0,
               'boundary_contacts': boundary_contacts, 'geometry_model': 'continuous point / pixel interiors',
               'geometry_tolerance': 1e-10}
    return details if return_details else segments


def physical_audit(directory,config):
    records=rows(directory/'evaluation.jsonl')
    trajectories=rows(directory/'evaluation_trajectories.jsonl')
    assert len(records)==len(trajectories)==20
    segments=0
    for r,t in zip(records,trajectories):
        assert r['map']==t['map'] and r['map']['split']=='validation'
        data=np.load(directory/f"evaluation_map_{r['map']['index']}_repeat_0.npz")
        truth,beliefs=data['truth'],data['beliefs']
        assert hashlib.sha256(truth.tobytes()).hexdigest()==r['map']['sha256']
        free=truth==255
        coverage=[float(np.count_nonzero((b==255)&free)/free.sum()) for b in beliefs]
        np.testing.assert_array_equal(coverage,r['coverage'])
        assert r['success']==all(c>=config['completion'] for c in coverage)
        assert r['sent_bytes']<=2*config['bytes_per_second']*r['connected_seconds']+1e-6
        assert r['delivered_bytes']<=r['sent_bytes'] and r['control_bytes']<=r['sent_bytes']
        assert r['total_distance']<=2*config['speed']*r['time']+1e-6
        assert r['time']<=config['horizon_seconds']+1e-6
        segments+=audit_segments(truth,t['trajectory'])
    return {'rollouts':20,'movement_segments':segments,'wall_crossings':0,
            'coverage_recomputed':True,'radio_and_speed_caps_passed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();root=args.run
    plan=json.loads((root/'pipeline.json').read_text());assert plan['status']=='pilot_complete'
    movement=root/'movement_A'
    first=torch.load(movement/'initial.pth',map_location='cpu')
    final=torch.load(movement/'latest.pth',map_location='cpu')
    training=rows(movement/'training.jsonl');updates=rows(movement/'updates.jsonl')
    assert len(training)==plan['training_sweeps']*100 and all(not r['interrupted'] for r in training)
    assert {r['map']['index'] for r in training}==set(range(100))
    manifest=json.loads((root/'dataset/manifest.json').read_text())
    train_ids={e['layout_id']:e['sha256'] for e in manifest if e['split']=='train'}
    assert all(r['map']['split']=='train' and train_ids[r['map']['layout_id']]==r['map']['sha256'] for r in training)
    old,new=first['learner']['updates'],final['learner']['updates']
    assert [r['updates'] for r in updates]==list(range(old+1,new+1))
    assert all(np.isfinite(v) for r in updates for v in r.values() if isinstance(v,(int,float)))
    assert all(e['map_key'] in train_ids for e in final['success_episodes'])
    assert len({e['map_key'] for e in final['success_episodes']})==len(final['success_episodes'])
    successful={}
    for r in training:
        if r['success']:
            key=r['map']['layout_id']
            successful[key]=min(successful.get(key,float('inf')),r['time'])
    assert {e['map_key'] for e in final['success_episodes']}==set(successful)
    assert all(e['duration']==successful[e['map_key']] for e in final['success_episodes'])
    changes={}
    for name in ['policy','critics','targets','content','context_encoder']:
        a,b=first['learner'][name],final['learner'][name]
        pairs=zip(a,b) if isinstance(a,list) else [(a,b)]
        changes[name]=float(sum((x[k]-y[k]).double().square().sum().item() for x,y in pairs for k in x)**.5)
    assert changes['policy']>0 and changes['critics']>0 and changes['targets']>0
    assert changes['content']==0 and changes['context_encoder']==0
    cfg=Config.from_checkpoint(final['config'])
    traces=rows(movement/'trajectories.jsonl')
    assert len(traces)==len(training)
    train_segments=0
    for r,t in zip(training,traces):
        assert r['episode']==t['episode']
        truth,_,_=configured_map(cfg,'train',r['map']['index'])
        train_segments+=audit_segments(truth,t['trajectory'])
        assert r['sent_bytes']<=2*cfg.bytes_per_second*r['connected_seconds']+1e-6
        assert r['delivered_bytes']<=r['sent_bytes'] and r['control_bytes']<=r['sent_bytes']
        assert r['total_distance']<=2*cfg.speed*r['time']+1e-6
        assert r['time']<=cfg.horizon_seconds+1e-6
        assert r['success']==all(c>=cfg.completion for c in r['coverage'])
    before,after=[rows(root/name/'evaluation.jsonl') for name in ['validation_initial','validation_final']]
    assert [r['map'] for r in before]==[r['map'] for r in after]
    summary={'status':'pilot_complete','scope':plan['scope'],'source':plan['source'],
        'training_seed':plan['seed'],'dataset_seed':plan['dataset_seed'],
        'complete_training_episodes':len(training),'distinct_training_maps':100,
        'successful_online_training_episodes':sum(r['success'] for r in training),
        'new_sac_updates':new-old,'cumulative_sac_updates':new,'success_memory_maps':len(final['success_episodes']),
        'content_updates':final['learner']['content_updates'],
        'weight_change_l2':changes,'validation':{},'test_policy_evaluations':0,
        'training_physical_audit':{'rollouts':len(training),'movement_segments':train_segments,
                                  'wall_crossings':0,'radio_and_speed_caps_passed':True},
        'limitations':['One training seed and one sweep; not statistical convergence.',
                       'Warm-started A/FIFO movement only; no multi-map head/AB comparison.',
                       'New dataset differs from the old two-map validation; compare paired new maps only.']}
    for name,records in [('initial',before),('final',after)]:
        summary['validation'][name]={'maps':20,'successes':sum(r['success'] for r in records),
            'mean_minimum_coverage':float(np.mean([min(r['coverage']) for r in records])),
            'mean_capped_time':float(np.mean([r['time'] for r in records])),
            'physical_audit':physical_audit(root/('validation_'+name),final['config'])}
    summary['paired_outcomes']={
        'both_successful':sum(a['success'] and b['success'] for a,b in zip(before,after)),
        'gained_success':sum(not a['success'] and b['success'] for a,b in zip(before,after)),
        'lost_success':sum(a['success'] and not b['success'] for a,b in zip(before,after)),
        'both_failed':sum(not a['success'] and not b['success'] for a,b in zip(before,after))}
    output=root/'report';output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot([r['episode'] for r in training],[min(r['coverage']) for r in training],'.-',lw=.7)
    axes[0,0].axhline(.99,color='gray',ls='--');axes[0,0].set(title='Online training: different map each episode',xlabel='Episode',ylabel='Minimum robot coverage',ylim=(0,1.02))
    for name,records in [('Initial',before),('Final',after)]:
        axes[0,1].plot(range(20),[min(r['coverage']) for r in records],'o-',label=name)
    axes[0,1].axhline(.99,color='gray',ls='--');axes[0,1].set(title='Same 20 development maps, greedy policy',xlabel='Validation map',ylabel='Minimum robot coverage',ylim=(0,1.02));axes[0,1].legend()
    successes=[sum(r['success'] for r in records) for records in (before,after)]
    bars=axes[1,0].bar(['Initial','Final'],successes,color=['tab:gray','tab:blue'])
    for bar,n in zip(bars,successes): axes[1,0].text(bar.get_x()+bar.get_width()/2,n+.3,f'{n}/20',ha='center')
    axes[1,0].set(title='Development completion, each robot >=99%',ylabel='Successful maps',ylim=(0,22))
    axes[1,1].plot([r['updates'] for r in updates],[r['policy_loss'] for r in updates],lw=.5)
    axes[1,1].set(title='Actual Actor learning',xlabel='Cumulative SAC updates',ylabel='Actor loss')
    fig.tight_layout();fig.savefig(output/'multimap_learning.png',dpi=160);fig.savefig(output/'multimap_learning.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
