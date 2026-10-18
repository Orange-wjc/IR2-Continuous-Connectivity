#!/usr/bin/env python3
"""Three independent neutral-start training chains, serial, frozen and auditable."""
import argparse,copy,fcntl,gc,hashlib,json,os,random,resource,shutil,signal,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-independent-matplotlib')
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner
from finite_contact.memory import load_checkpoint,semantic_digest
from finite_contact.maps import manifest,training_map_index
from finite_contact_train import rollout,checkpoint,evaluate,append_json
from finite_contact_multimap_joint import common_state,seed_episode,identical,archive_after_checkpoint
import finite_contact_multimap_joint as joint
from finite_contact_multimap_content import Reservoir,balanced_batch
from finite_contact_thousand_analysis import audit_rollouts,communication_audit,optimizer_audit,paired,rows,write,read,now,digest
ROOT=Path(__file__).resolve().parent
GROUPS={'B0_fifo':('B0','fifo',True),'A_fifo':('A','fifo',True),'B_learned':('B','learned',True),'AB_learned':('AB','learned',True),'AB_uncoupled':('AB','learned',False)}
CONTRASTS=[('B0_fifo','A_fifo'),('B0_fifo','B_learned'),('B_learned','AB_learned'),('A_fifo','AB_learned'),('AB_uncoupled','AB_learned')]

def identity():
 return {str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.md')}
def verify(root):
 plan=read(root/'pipeline.json');name=plan.get('source_identity_file','source_identity.json')
 if name!='source_identity.json':
  review=read(root/'checks/physical_audit_recovery_review.json')
  assert review['status']=='passed' and review['source_identity_file']==name
  assert review['source_identity_sha256']==digest(root/name)
  assert review['original_source_identity_sha256']==digest(root/'source_identity.json')
 assert identity()==read(root/name),'Frozen independent source changed'
 return plan
def status(path,**value):
 write(path/'status.json',dict(time=now(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**value))
def provenance(path,cfg):
 path.mkdir(parents=True,exist_ok=True)
 write(path/'config.json',cfg.to_dict());write(path/'source_identity.json',identity())
 write(path/'runtime.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device='cuda',gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads()))
def configuration(seed,plan):
 cfg=Config(seed=seed,map_family='simple_rooms_v1',map_seed=20261011,map_order='shuffled',sensor_implementation='vectorized',train_maps=plan['pretrain_maps'],validation_maps=20,test_maps=20,variant='B0',content='fifo',batch_size=16,warmup=16,replay_size=12000,compact_replay=True,success_imitation_weight=1.,success_memory_mode='per_map',success_episode_limit=1000,torch_threads=2)
 if plan['preflight']:cfg.horizon_seconds=4.;cfg.completion=.2;cfg.batch_size=2;cfg.warmup=2
 cfg.validate();return cfg
def fresh(cfg,device='cuda',state=None):
 learner=Learner(cfg,device);learner.lossless_observations=True
 if state is not None:learner.load(state,optimizers=True)
 return learner
def save_episode(path,episode,metric,env):
 append_json(path/'training.jsonl',{**metric,'episode':episode})
 append_json(path/'trajectories.jsonl',dict(episode=episode,map=env.map_info,trajectory=env.trajectory))
 for event in env.radio.events:append_json(path/'communication.jsonl',dict(episode=episode,**event))
 np.savez_compressed(path/('training_map_%03d.npz'%episode),truth=env.truth,beliefs=np.stack(env.beliefs),positions=env.positions)
def check_optimizer(initial,final,steps):
 pairs=[('policy',initial['policy_optimizer'],final['policy_optimizer']),('alpha',initial['alpha_optimizer'],final['alpha_optimizer'])]+[('q%d'%i,a,b) for i,(a,b) in enumerate(zip(initial['q_optimizers'],final['q_optimizers']))]
 out={}
 for name,a,b in pairs:
  assert b['state']
  if a['state']:assert a['state'].keys()==b['state'].keys()
  increments=[int(v['step'])-int(a['state'].get(k,{}).get('step',0)) for k,v in b['state'].items()]
  assert set(increments)=={steps},(name,set(increments),steps)
  out[name]=dict(parameters=len(increments),increment=steps)
 assert identical(initial['content_optimizer'],final['content_optimizer'])
 return out

def movement(root,seed,plan):
 path=root/('seed_%d'%seed)/'movement';cfg=configuration(seed,plan);provenance(path,cfg)
 source=torch.load(path/'start.pth',map_location='cpu');x=fresh(cfg,state=source['learner']);completed=0;previous=0.
 if (path/'latest.pth').exists():
  saved=load_checkpoint(path/'latest.pth');x.load(saved['learner'],optimizers=True);x.replay.extend(saved['replay']);x.success_episodes=saved['success_episodes'];completed=saved['episode'];previous=saved['training_wall_seconds'];archive_after_checkpoint(path,completed,x.updates);del saved
 else:checkpoint(path/'initial.pth',x,cfg,'movement',0,None,0.,full=False)
 context={};x.on_update=lambda row:append_json(path/'updates.jsonl',{**context,**row})
 stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
 started=time.monotonic();deadline=started+4*3600
 for ep in range(completed+1,cfg.train_maps+1):
  seed_episode(seed,ep);index=training_map_index(cfg,ep);context.update(episode=ep,map_index=index)
  metric,env=rollout(x,cfg,'train',index,'movement',deadline=deadline,stop_requested=lambda:stop[0],episode_update_limit=0)
  assert not metric['interrupted'],'Movement interrupted; resume the last boundary'
  before=x.updates
  for _ in range(plan['pretrain_updates_per_map']):
   assert not stop[0] and time.monotonic()<deadline
   assert x.update()
  assert x.updates-before==plan['pretrain_updates_per_map']
  metric.update(online_sac_updates=0,post_episode_sac_updates=x.updates-before,learner_updates=x.updates,content_updates=x.content_updates,wall_seconds=previous+time.monotonic()-started)
  save_episode(path,ep,metric,env)
  if ep%10==0 or ep==cfg.train_maps:checkpoint(path/'latest.pth',x,cfg,'movement',ep,None,previous+time.monotonic()-started)
  status(path,status='training',completed_episodes=ep,total_episodes=cfg.train_maps,sac_updates=x.updates)
  print(json.dumps(dict(stage='movement',seed=seed,episode=ep,updates=x.updates)),flush=True)
 assert x.updates==cfg.train_maps*plan['pretrain_updates_per_map']
 checkpoint(path/'final.pth',x,cfg,'movement',cfg.train_maps,None,previous+time.monotonic()-started,full=False)
 final=torch.load(path/'final.pth',map_location='cpu')['learner'];audit=dict(optimizer=check_optimizer(source['learner'],final,x.updates),physical=audit_rollouts(path,cfg,cfg.train_maps),communication=communication_audit(path,rows(path/'training.jsonl')))
 assert all(identical(source['learner'][k],final[k]) for k in ('content','context_encoder'))
 write(path/'audit.json',audit);status(path,status='complete',completed_episodes=cfg.train_maps,sac_updates=x.updates)

def head(root,seed,plan):
 seedroot=root/('seed_%d'%seed);path=seedroot/'content';source=load_checkpoint(seedroot/'movement/latest.pth');cfg=Config.from_checkpoint(source['config']);seed_episode(seed,10001)
 path.mkdir(exist_ok=True);collection=path/'collection';collection.mkdir(exist_ok=True)
 if any(collection.iterdir()):raise RuntimeError('Partial head collection requires manual archive/accounting before resume')
 x=fresh(cfg,state=source['learner']);x.context_encoder=copy.deepcopy(x.policy).requires_grad_(False).eval();x.policy.requires_grad_(False).eval()
 for q in x.critics+x.targets:q.requires_grad_(False).eval()
 x.replay.clear();x.success_episodes=[]
 provenance(collection,cfg);datasets=[];started=time.monotonic()
 for ep in range(1,cfg.train_maps+1):
  assert time.monotonic()-started<6*3600,'Head collection wall budget exceeded'
  index=training_map_index(cfg,ep);env=Environment(cfg,'train',index,scorer=x.score);reservoir=Reservoir(plan['samples_per_map'],seed+index+1000);done=False;decisions=0
  while not done:
   if decisions%4==0:
    contexts=x.contexts(env);env.feature_provider=lambda sender,ids:contexts[sender][ids]
    for sender in range(2):reservoir.add(env.collect_labels(sender))
   obs=[env.observation(i) for i in range(2)];actions=x.act(obs,greedy=True);_,_,terminated,truncated=env.step(actions);done=terminated or truncated;decisions+=1
  assert reservoir.data is not None and len(reservoir.data[2])>=cfg.batch_size
  datasets.append((index,reservoir.data));f,c,y=reservoir.data
  np.savez_compressed(collection/('labels_%03d.npz'%index),features=f,context=c,labels=y)
  metric=env.metrics(terminated);metric.update(map=env.map_info,decisions=decisions,interrupted=False,learner_updates=x.updates,content_updates=0)
  save_episode(collection,ep,metric,env)
  status(path,status='collecting_labels',completed_episodes=ep,total_episodes=cfg.train_maps,sac_updates=x.updates,head_updates=0)
  print(json.dumps(dict(stage='head_collect',seed=seed,episode=ep)),flush=True)
 datasets=[d for _,d in sorted(datasets)];directory=path/'head';provenance(directory,cfg)
 checkpoint(directory/'initial.pth',x,cfg,'content',0,None,0.,full=False);initial=copy.deepcopy(x.state());x.on_content_update=lambda r:append_json(directory/'content_updates.jsonl',r)
 rng=np.random.RandomState(seed);counts=np.zeros(cfg.train_maps,dtype=int)
 for step in range(plan['head_updates']):
  batch,indices=balanced_batch(datasets,rng,cfg.batch_size);counts+=np.bincount(indices,minlength=cfg.train_maps);x.update_content(batch)
  if (step+1)%1000==0:status(path,status='training_head',completed_episodes=cfg.train_maps,head_updates=step+1)
 assert x.content_updates==plan['head_updates'] and np.all(counts>0)
 final=x.state();assert all(identical(initial[k],final[k]) for k in ('policy','critics','targets','log_alpha','context_encoder'))
 assert identical(initial['policy_optimizer'],final['policy_optimizer'])
 assert all(int(v['step'])==plan['head_updates'] for v in final['content_optimizer']['state'].values())
 checkpoint(directory/'final.pth',x,cfg,'content',cfg.train_maps,None,time.monotonic()-started,full=False)
 write(path/'audit.json',dict(physical=audit_rollouts(collection,cfg,cfg.train_maps),communication=communication_audit(collection,rows(collection/'training.jsonl')),movement_frozen=True,head_updates=plan['head_updates'],samples_per_map=[len(d[2]) for d in datasets],map_draw_counts=counts.tolist()))
 # Start all variants from exactly this seed's common network/Adam/head, no old replay.
 common=common_state(source,torch.load(directory/'final.pth',map_location='cpu'),1e-5,seed);acfg=Config.from_checkpoint(common['config']);acfg.train_maps=plan['adaptation_maps'];acfg.validate();common['config']=acfg.to_dict()
 torch.save(common,seedroot/'common_start.pth',pickle_protocol=4);write(seedroot/'common_identity.json',dict(sha256=digest(seedroot/'common_start.pth'),movement_sha256=digest(seedroot/'movement/latest.pth'),head_sha256=digest(directory/'final.pth'),source_sac_updates=common['learner']['updates'],head_updates=common['learner']['content_updates']))
 status(path,status='complete',completed_episodes=cfg.train_maps,head_updates=x.content_updates)

def group_config(common,name):
 cfg=Config.from_checkpoint(common['config']);cfg.variant,cfg.content,cfg.coupled=GROUPS[name];cfg.validate();return cfg

def adapt(root,seed,name,plan):
 path=root/('seed_%d'%seed);ci=read(path/'common_identity.json');assert digest(path/'common_start.pth')==ci['sha256']
 common=torch.load(path/'common_start.pth',map_location='cpu');assert common['learner']['updates']==plan['pretrain_maps']*plan['pretrain_updates_per_map'];assert common['learner']['content_updates']==plan['head_updates']
 joint.group_config=group_config;joint.provenance=provenance
 if plan['preflight']:
  old=joint.evaluate
  joint.evaluate=lambda learner,cfg,count,directory,stop_requested:old(learner,cfg,2,split='train',repeats=10,directory=directory,stop_requested=stop_requested)
 small=dict(max_hours_per_group=12,workers=1,memory_profile=None if plan['preflight'] else 'rtx3090_32gb_lossless_v1',sweeps=1,updates_per_map=plan['adaptation_updates_per_map'],source_sac_updates=common['learner']['updates'],new_sac_updates_per_group=plan['adaptation_maps']*plan['adaptation_updates_per_map'])
 d=path/name;resume=d/'latest.pth' if (d/'latest.pth').exists() else None
 committed=(d/'status.json').exists() and read(d/'status.json')['status']=='complete'
 if not committed:joint.train_group(path,small,name,'cuda',resume)
 else:
  assert (d/'final.pth').exists() and resume is not None
  records=rows(d/'training.jsonl');assert len(records)==plan['adaptation_maps']
  assert [v['episode'] for v in records]==list(range(1,plan['adaptation_maps']+1))
  print(json.dumps(dict(stage='audit_only',seed=seed,group=name,new_training_updates=0,new_policy_evaluations=0)),flush=True)
 cfg=group_config(common,name);initial=torch.load(d/'initial.pth',map_location='cpu');final=load_checkpoint(d/'latest.pth')
 assert identical(initial['learner'],common['learner']);assert final['learner']['updates']==small['source_sac_updates']+small['new_sac_updates_per_group'];assert final['learner']['content_updates']==plan['head_updates']
 assert all(identical(initial['learner'][k],final['learner'][k]) for k in ('content','context_encoder'))
 audit=dict(optimizer=optimizer_audit(initial['learner'],final['learner'],small['new_sac_updates_per_group']),training_physical=audit_rollouts(d,cfg,plan['adaptation_maps']),training_communication=communication_audit(d,rows(d/'training.jsonl')))
 if not plan['preflight']:
  audit['development_physical']=audit_rollouts(d/'validation_final',cfg,20,True);audit['development_communication']=communication_audit(d/'validation_final',rows(d/'validation_final/evaluation.jsonl'),True)
 else:assert all(r['map']['split']=='train' for r in rows(d/'validation_final/evaluation.jsonl'))
 write(d/'audit.json',audit)

def finish(root,plan):
 result=dict(status='complete',time=now(),seeds={},test_policy_evaluations=0,preflight=plan['preflight'],scope=plan['scope'],limitations=['Three complete independent seeds; still a small replication count.','Same repeatedly used development maps; no held-out test execution.','Neutral B0/FIFO pretraining is a new protocol, not a matched ranking against the previous A/FIFO-start round.'])
 for seed in plan['seeds']:
  path=root/('seed_%d'%seed);groups={};data={}
  for name in GROUPS:
   assert read(path/name/'status.json')['status']=='complete' and (path/name/'audit.json').exists()
   records=rows(path/name/'validation_final/evaluation.jsonl');assert len(records)==20
   data[name]=records;groups[name]=dict(successes=sum(r['success'] for r in records),mean_capped_time=float(np.mean([r['time'] for r in records])),audit=str(path/name/'audit.json'))
  pairs={a+'_vs_'+b:paired(data[a],data[b]) for a,b in CONTRASTS}
  interaction=float(np.mean([a['time']+b['time']-base['time']-ab['time'] for base,a,b,ab in zip(data['B0_fifo'],data['A_fifo'],data['B_learned'],data['AB_learned'])]))
  result['seeds'][str(seed)]=dict(groups=groups,pairs=pairs,interaction_seconds=interaction)
 if not plan['preflight']:
  result['aggregate']={name:dict(seed_successes=[v['groups'][name]['successes'] for v in result['seeds'].values()],seed_mean_times=[v['groups'][name]['mean_capped_time'] for v in result['seeds'].values()],mean_of_seed_means=float(np.mean([v['groups'][name]['mean_capped_time'] for v in result['seeds'].values()])),sample_std_seed_means=float(np.std([v['groups'][name]['mean_capped_time'] for v in result['seeds'].values()],ddof=1))) for name in GROUPS}
 write(root/'report/summary.json',result)
 lines=['# 独立链预检（无开发/测试策略执行）' if plan['preflight'] else '# 完整独立三种子模块验证','', '本轮训练、固定终点评估和物理/通信/优化器审计完成。测试策略执行0。','', '| seed | 组 | 成功/20 | 平均任务秒 |','| --- | --- | ---: | ---: |']
 for seed,v in result['seeds'].items():
  for name,g in v['groups'].items():lines.append('| %s | %s | %d | %.3f |'%(seed,name,g['successes'],g['mean_capped_time']))
 lines+=['','完整逐种子配对和交互效应见 summary.json。三种子不能保证普遍泛化；旧轮起点协议不同，不混合排名。','本轮到此结束，不自动启动下一轮；飞书主动通知关闭。']
 (root/'report/RESULTS_ANALYSIS.md').write_text('\n'.join(lines)+'\n');write(root/'source_identity_after.json',identity())

def prepare(root,preflight):
 root.mkdir(parents=True,exist_ok=True)
 assert not any(p.name not in {'.coordinator.lock','.worker.lock'} for p in root.iterdir()),'Use a new directory or --resume'
 plan=dict(status='prepared',created=now(),scope='Independent neutral B0/FIFO pretraining + independently trained communication head + paired five-method adaptation',preflight=preflight,seeds=[101,202] if preflight else [101,202,303],groups=list(GROUPS),pretrain_maps=2 if preflight else 100,pretrain_updates_per_map=2 if preflight else 90,head_updates=4 if preflight else 5000,samples_per_map=32 if preflight else 2048,adaptation_maps=2 if preflight else 1000,adaptation_updates_per_map=2 if preflight else 20,test_policy_evaluations=0,workers=1,notifications=False,stages_complete=[])
 write(root/'source_identity.json',identity());shutil.copy2(ROOT/'INDEPENDENT_PLAN.md',root/'pre_run_design.md')
 actors=[]
 for seed in plan['seeds']:
  cfg=configuration(seed,plan);seed_episode(seed,0);x=fresh(cfg,'cpu');d=root/('seed_%d'%seed)/'movement';d.mkdir(parents=True)
  checkpoint(d/'start.pth',x,cfg,'movement',0,None,0.,full=False);h=semantic_digest(dict(policy=x.state()['policy']));actors.append(h);write(d/'initialization.json',dict(seed=seed,actor_sha256=h,independent=True));del x;gc.collect()
 assert len(set(actors))==len(actors),'Seeds did not initialize independently';plan['initial_actor_hashes']=actors
 write(root/'pipeline.json',plan);return plan

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--worker-stage',choices=['movement','head','adapt']);p.add_argument('--seed',type=int);p.add_argument('--group',choices=list(GROUPS));a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 assert Path(sys.prefix)==Path('/home/wyz/anaconda3/envs/ir2_train')
 with (root/('.worker.lock' if a.worker_stage else '.coordinator.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.worker_stage:
   assert read(root/'coordinator.json')['pid']==os.getppid(),'Workers must be started by coordinator';plan=verify(root);assert torch.cuda.is_available();torch.set_num_threads(2)
   if a.worker_stage=='movement':movement(root,a.seed,plan)
   elif a.worker_stage=='head':head(root,a.seed,plan)
   else:adapt(root,a.seed,a.group,plan)
   verify(root);return
  with (root/'.worker.lock').open('a') as probe:
   fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=verify(root) if a.resume else prepare(root,a.preflight)
  write(root/'coordinator.json',dict(pid=os.getpid(),time=now()));plan['status']='running';write(root/'pipeline.json',plan)
  child=None
  def stop_coordinator(*_):raise KeyboardInterrupt('Coordinator stop requested')
  signal.signal(signal.SIGTERM,stop_coordinator)
  try:
   for seed in plan['seeds']:
    for stage,name in [('movement',None),('head',None)]+[('adapt',g) for g in GROUPS]:
     key='%d/%s/%s'%(seed,stage,name or '')
     if key in plan['stages_complete']:continue
     verify(root);plan.update(active_seed=seed,active_stage=stage,active_group=name,updated=now());write(root/'pipeline.json',plan)
     command=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--worker-stage',stage,'--seed',str(seed)]
     if name:command+=['--group',name]
     with (root/('%d_%s_%s.log'%(seed,stage,name or ''))).open('a') as log:
      child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
      while True:
       try:code=child.wait(timeout=30);break
       except subprocess.TimeoutExpired:plan['updated']=now();write(root/'pipeline.json',plan)
      assert code==0,'Worker failed: '+key
     plan['stages_complete'].append(key);write(root/'pipeline.json',plan)
   verify(root);finish(root,plan);plan.update(status='complete',completed=now(),active_stage=None,active_group=None);write(root/'pipeline.json',plan)
  except BaseException:
   if child is not None and child.poll() is None:child.terminate();child.wait()
   plan.update(status='interrupted_or_failed',stopped=now());write(root/'pipeline.json',plan);raise
if __name__=='__main__':main()
