"""Numerical/restore contracts; generated observations are fixtures, not training maps."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,copy,json,random,tempfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'native_training'))
from finite_contact.config import Config
from finite_contact.learning import Learner,compact_observation
from finite_contact.memory import pack_observation,load_checkpoint
from native_learning import NativeLearner
from native_driver import save_state,restore_state,commit_episode,configuration

def seed():random.seed(20261017);np.random.seed(20261017);torch.manual_seed(20261017)
def max_error(a,b):
    return max((a[k]-b[k]).abs().max().item() for k in a)

def main():
    torch.set_num_threads(2);seed()
    cfg=Config(nodes=8,grid_points=2,neighbors=4,embedding_dim=32,batch_size=16,warmup=16,
               success_imitation_weight=1.,success_memory_mode='per_map',success_episode_limit=1,compact_replay=True)
    cfg.validate();base=Learner(cfg)
    def obs(n):
        return (np.random.randn(n,12).astype(np.float32),np.array([[0,1,2,3]],np.int64),
                np.array([[0]],np.int64),np.zeros((1,n),np.uint8),np.zeros((1,4),np.uint8),np.zeros((n,n),np.uint8))
    transitions=[]
    for i in range(24):
        a=pack_observation(compact_observation(obs(5+i%2)));b=pack_observation(compact_observation(obs(5+(i+1)%2)))
        transitions.append((a,i%4,float(i%3),1.,b,float(i>=22),False))
    base.replay.extend(transitions);base.remember_success(transitions,12.,'training_fixture')
    results=[]
    for micro in (16,2):
        x=NativeLearner(cfg,'cpu',micro);x.load(copy.deepcopy(base.state()),optimizers=True)
        x.replay.extend(transitions);x.success_episodes=copy.deepcopy(base.success_episodes)
        reference=copy.deepcopy(base)
        seed();a=reference.update();seed();b=x.update()
        error=max(max_error(reference.policy.state_dict(),x.policy.state_dict()),
                  *(max_error(c.state_dict(),d.state_dict()) for c,d in zip(reference.critics,x.critics)))
        assert error<2e-5,(micro,error)
        assert x.imitation_updates==1 and b['imitation_loss']>0 and len(x.success_episodes)==1
        assert abs(a['policy_loss']-b['policy_loss'])<2e-4
        assert all(torch.allclose(c,d,atol=2e-5,rtol=1e-5) for t,q in zip(reference.targets,x.targets) for c,d in zip(t.parameters(),q.parameters()))
        with tempfile.TemporaryDirectory(prefix='ir2-repair-') as tmp:
            path=Path(tmp)/'full.pth';save_state(path,x,cfg,3,'numeric_fixture')
            y,loaded,episode,_=restore_state(path,'cpu',micro)
            assert episode==3 and len(y.success_episodes)==1 and len(y.replay)==24
            seed();r1=x.update();seed();r2=y.update()
            assert max_error(x.policy.state_dict(),y.policy.state_dict())==0
            assert r1['imitation_updates']==r2['imitation_updates']==2
            # Exercise the real worker-result -> remember_success -> update path.
            directory=Path(tmp)/'commit';directory.mkdir()
            metric=dict(success=True,time=12.,map=dict(layout_id='training_fixture'))
            commit_episode(y,cfg,directory,dict(metric=metric,transitions=transitions,worker_rss_mib=1),4,updates=1)
            assert y.imitation_updates==3 and len(y.success_episodes)==1
        results.append(dict(microbatch=micro,max_parameter_error_vs_validated_recipe=error,
            imitation_loss=b['imitation_loss'],checkpoint_restore_exact=True,commit_success_chain=True))
    # Alpha guard is active even when restoring a historical out-of-range value.
    with torch.no_grad():base.log_alpha.fill_(-10)
    x=NativeLearner(cfg,'cpu',16);x.load(copy.deepcopy(base.state()),optimizers=True)
    x.replay.extend(transitions);x.success_episodes=base.success_episodes;x.update()
    assert x.log_alpha.item()>=-8
    original=configuration(REPO/'experiments/native_easy_20261015_seed101/dataset/native_dataset.json')
    assert original.batch_size==16 and original.learning_rate==1e-4
    assert original.success_imitation_weight==1 and original.success_episode_limit==original.train_maps
    report=dict(numerical_checks=results,alpha_guard=True,native_graphs_retained=original.native_graph,
                actual_training_map_rollouts=0,actual_development_rollouts=0,fixture_updates_only=True)
    (Path(__file__).parent/'unit_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
