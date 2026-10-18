#!/usr/bin/env python3
"""Verify an actual checkpoint's four RNG states, without reseeding the continuation."""
import argparse,json,os,random,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-performance-matplotlib');sys.path.insert(0,str(a.source.resolve()))
import numpy as np
import torch
from finite_contact.learning import Learner
from finite_contact.memory import semantic_digest,load_checkpoint
from finite_contact_train import rollout,checkpoint
from finite_contact_ablation import group_config
from finite_contact_multimap_joint import seed_episode
root=Path('/home/wyz/wjc_ws/IR2-Continuous-Connectivity')
common=torch.load(root/'experiments/ablation_20261013_seed1/common_start.pth',map_location='cpu');cfg=group_config(common,'AB_learned')
assert torch.cuda.is_available();torch.set_num_threads(2)
def make(state):
 x=Learner(cfg,'cuda');x.lossless_observations=True;x.load(state,optimizers=True)
 x.content.requires_grad_(False).eval();x.context_encoder.requires_grad_(False).eval()
 return x
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def run(x,index):
 metric,env=rollout(x,cfg,'train',index,'finetune',episode_update_limit=0)
 updates=[]
 for _ in range(20):updates.append(x.update())
 digest=semantic_digest(dict(metrics=metric,trajectory=env.trajectory,events=env.radio.events,beliefs=env.beliefs,positions=env.positions,learner=x.state(),replay=list(x.replay),success=x.success_episodes,updates=updates,rng=rng()))
 return digest
x=make(common['learner']);seed_episode(1,1);prefix=run(x,0)
path=a.output/'checkpoint.pth';checkpoint(path,x,cfg,'finetune',1,None,0.)
saved=load_checkpoint(path)
checkpoint_rng=dict(python=saved['python_rng'],numpy=saved['numpy_rng'],torch=saved['torch_rng'],cuda=saved['cuda_rng'])
assert semantic_digest(rng())==semantic_digest(checkpoint_rng)
expected=run(x,37)  # Deliberately no seed_episode here.
y=make(saved['learner']);y.replay.extend(saved['replay']);y.success_episodes=saved['success_episodes']
random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
assert semantic_digest(rng())==semantic_digest(checkpoint_rng)
actual=run(y,37)
assert expected==actual,'Resumed continuation diverged'
path.unlink()
report=dict(passed=True,source=str(a.source.resolve()),cuda=True,prefix_sha256=prefix,continuation_sha256=expected,restored_sha256=actual,rng_restored_from_checkpoint=['python','numpy','torch_cpu','torch_cuda'],continuation_reseeded=False,training_rollouts=3,sac_updates=60,test_policy_evaluations=0)
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
