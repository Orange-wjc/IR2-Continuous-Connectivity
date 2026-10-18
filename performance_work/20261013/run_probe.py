#!/usr/bin/env python3
"""Isolated real-CUDA engineering probe; never changes research experiment state."""
import argparse, cProfile, hashlib, json, os, pstats, random, resource, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--groups',default='AB_learned')
p.add_argument('--maps',default='0,37,470')
p.add_argument('--profile',action='store_true')
p.add_argument('--trace',action='store_true')
p.add_argument('--resume-check',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-performance-matplotlib')
sys.path.insert(0,str(a.source.resolve()))
import numpy as np
import torch
from finite_contact.learning import Learner
from finite_contact.memory import semantic_digest,load_checkpoint
from finite_contact_train import rollout,checkpoint
from finite_contact_ablation import group_config
from finite_contact_multimap_joint import seed_episode
ROOT=Path('/home/wyz/wjc_ws/IR2-Continuous-Connectivity')
common=torch.load(ROOT/'experiments/ablation_20261013_seed1/common_start.pth',map_location='cpu')
assert torch.cuda.is_available();torch.set_num_threads(2)
all_results=[]
def rng():
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def learner_for(cfg):
 x=Learner(cfg,'cuda');x.lossless_observations=True;x.load(common['learner'],optimizers=True)
 x.content.requires_grad_(False).eval();x.context_encoder.requires_grad_(False).eval()
 return x
def state_hash(x):
 return semantic_digest(dict(learner=x.state(),replay=list(x.replay),success_episodes=x.success_episodes,rng=rng()))
def episode(x,cfg,index,number,trace):
 seed_episode(1,number)
 decisions=[]
 original=x.act
 if trace:
  def act(obs,*args,**kw):
   actions=original(obs,*args,**kw)
   decisions.append(dict(observations=semantic_digest(dict(obs=obs)),actions=actions))
   return actions
  x.act=act
 torch.cuda.synchronize();t=time.perf_counter()
 metrics,env=rollout(x,cfg,'train',index,'finetune',episode_update_limit=0)
 torch.cuda.synchronize();rollout_time=time.perf_counter()-t
 u=time.perf_counter()
 for _ in range(20):assert x.update()
 torch.cuda.synchronize();update_time=time.perf_counter()-u
 if trace:x.act=original
 result=dict(map_index=index,episode=number,metrics=metrics,rollout_seconds=rollout_time,update_seconds=update_time,total_seconds=rollout_time+update_time,decisions=metrics['decisions'])
 if trace:
  result.update(policy_inputs_actions_sha256=semantic_digest(dict(decisions=decisions)),trajectory_sha256=semantic_digest(dict(trajectory=env.trajectory)),communication_sha256=semantic_digest(dict(events=env.radio.events)),final_environment_sha256=semantic_digest(dict(truth=env.truth,beliefs=env.beliefs,positions=env.positions)),learner_replay_rng_sha256=state_hash(x))
 return result
# Warm the actual CUDA model and kernels, then reset weights, buffers and RNG for measurements.
warm_cfg=group_config(common,'AB_learned');warm=learner_for(warm_cfg)
episode(warm,warm_cfg,0,99,False);del warm;torch.cuda.empty_cache()
prof=cProfile.Profile()
if a.profile:prof.enable()
for group in a.groups.split(','):
 cfg=group_config(common,group);x=learner_for(cfg)
 for number,index in enumerate(map(int,a.maps.split(',')),1):
  result=episode(x,cfg,index,number,a.trace);result['group']=group
  all_results.append(result)
  print(json.dumps({k:result[k] for k in ['group','map_index','episode','rollout_seconds','update_seconds','decisions']},ensure_ascii=False),flush=True)
 if a.resume_check and group == 'AB_learned':
  saved=a.output/(group+'_resume.pth');checkpoint(saved,x,cfg,'finetune',number,None,0.)
  expected_rng=rng();r=episode(x,cfg,37,number+1,True)
  state=load_checkpoint(saved);restored=learner_for(cfg);restored.load(state['learner'],optimizers=True)
  restored.replay.extend(state['replay']);restored.success_episodes=state['success_episodes']
  restore_rng(expected_rng);rr=episode(restored,cfg,37,number+1,True)
  for k in ['metrics','policy_inputs_actions_sha256','trajectory_sha256','communication_sha256','final_environment_sha256','learner_replay_rng_sha256']:assert r[k]==rr[k],(group,k)
  all_results.append(dict(group=group,resume_equivalence=True,continuation=r))
  saved.unlink();del restored
 del x;torch.cuda.empty_cache()
if a.profile:
 prof.disable();prof.dump_stats(str(a.output/'profile.pstats'))
 with (a.output/'profile.txt').open('w') as f:pstats.Stats(prof,stream=f).strip_dirs().sort_stats('cumulative').print_stats(55)
 s=pstats.Stats(prof)
 stats=[dict(file=k[0],line=k[1],name=k[2],primitive_calls=v[0],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in s.stats.items()]
 (a.output/'profile.json').write_text(json.dumps(sorted(stats,key=lambda x:x['cumulative_seconds'],reverse=True),indent=2))
report=dict(source=str(a.source.resolve()),cuda=True,device=torch.cuda.get_device_name(0),torch_version=torch.__version__,torch_threads=torch.get_num_threads(),profiled=a.profile,traced=a.trace,warmup_rollouts=1,warmup_updates=20,rows=all_results,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
(a.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PROBE_COMPLETE',str(a.output),flush=True)
