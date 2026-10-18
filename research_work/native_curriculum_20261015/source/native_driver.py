#!/usr/bin/env python3
"""Native-map, paired IR2 SAC training; easy first, difficult gated by review.

Original maps/ordering, original hierarchical graph, replay warmup 2000,
effective batch 128, 8 updates/finished episode, hard targets every 64 updates.
Finite-contact environment and the paired control share byte/time constraints.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-native-mpl')
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone,timedelta
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from finite_contact.config import Config
from finite_contact.memory import load_checkpoint,semantic_digest
from native_data import dataset,digest
from native_learning import NativeLearner
from native_worker import initialize_worker,run_episode,cpu_weights,seed_episode

SOURCE=Path(__file__).resolve().parent
GROUPS={'IR2_FIFO':('B0','fifo'), 'AB_learned':('AB','learned')}

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

def identity():
    return {str(p.relative_to(SOURCE)):digest(p) for p in sorted(SOURCE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json')}

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None:torch.cuda.set_rng_state_all(state['cuda'])

def save_state(path,learner,cfg,episode,role):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    value=dict(format='ir2-native-finite-v1',config=cfg.to_dict(),learner=learner.state(),
               replay=list(learner.replay),content_replay=list(learner.content_replay),
               success_episodes=learner.success_episodes,episode=episode,role=role,rng=rng_state(),time=now())
    tmp=p.with_suffix('.tmp');torch.save(value,tmp,pickle_protocol=4);tmp.replace(p)
    return value['rng']

def restore_state(path,device='cuda',microbatch=32):
    saved=load_checkpoint(path);cfg=Config.from_checkpoint(saved['config'])
    x=NativeLearner(cfg,device,microbatch);x.load(saved['learner'],optimizers=True)
    x.replay.extend(saved['replay']);x.content_replay.extend(saved['content_replay'])
    x.success_episodes=saved['success_episodes'];restore_rng(saved['rng'])
    return x,cfg,saved['episode'],saved['rng']

def configuration(manifest,family='easy',seed=101,preflight=False):
    d=dataset(str(manifest))['families'][family]['splits']
    cfg=Config(dataset_manifest=str(Path(manifest).resolve()),native_graph=True,dynamic_padding=True,
        map_family='dungeon_'+family,map_size=640 if family=='easy' else 1000,
        train_maps=len(d['train']),validation_maps=len(d['validation']),test_maps=500,
        map_order='cyclic',seed=seed,sensor_implementation='vectorized',robots=2,
        sensor_range=80,grid_points=50,nodes=2048,neighbors=30,embedding_dim=128,
        max_decisions=(196 if family=='easy' else 384),horizon_seconds=(3000. if family=='easy' else 6000.),
        batch_size=128,warmup=2000,replay_size=10000,compact_replay=True,
        learning_rate=1e-5,policy_learning_rate=1e-5,alpha_learning_rate=1e-4,
        success_imitation_weight=0.,success_memory_mode='fastest',success_episode_limit=1000,
        torch_threads=2,pointer_mode='centered')
    if preflight:cfg.max_decisions=8;cfg.horizon_seconds=60.;cfg.batch_size=16;cfg.warmup=16
    cfg.validate();return cfg

def atomic_weights(path,learner):
    tmp=Path(path).with_suffix('.tmp');torch.save(cpu_weights(learner),tmp);tmp.replace(path)

def jobs_for(cfg,learner,directory,first,count,collect_labels=False,evaluation=False,indices=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    weights=directory/'worker_weights.pth';atomic_weights(weights,learner)
    return [dict(config=cfg.to_dict(),weights=str(weights),episode=ep,
                 map_index=((ep-1)%cfg.train_maps if indices is None else indices[i]),
                 split='validation' if evaluation else 'train',
                 greedy=bool(evaluation or collect_labels),evaluation=bool(evaluation or collect_labels),
                 collect_labels=collect_labels,output=str(directory/'episodes'))
            for i,ep in enumerate(range(first,first+count))]

def wave(pool,jobs):
    # Consume in submission order: worker completion order never changes replay order.
    futures=[pool.submit(run_episode,job) for job in jobs]
    return [f.result() for f in futures]

def commit_episode(x,cfg,directory,item,episode,writer=None,enable_updates=True):
    before=x.updates;x.replay.extend(item['transitions'])
    for _ in range(8):
        if enable_updates and len(x.replay)>=cfg.warmup:
            assert x.update()
    row={**item['metric'],'learner_updates':x.updates,'new_sac_updates':x.updates-before,
         'head_updates':x.content_updates,'worker_rss_mib':item['worker_rss_mib'],
         'committed':now()}
    append(directory/'training.jsonl',row)
    if writer:
        for name in ('success','time','capped_time','union_coverage','peak_graph_nodes'):
            writer.add_scalar(name,float(row[name]),episode)
        writer.add_scalar('sac_updates',x.updates,episode)
        writer.add_scalar('replay_size',len(x.replay),episode)
    return row

def recover_logs(directory,episode,updates):
    """Archive any uncommitted completed wave; never silently discard extra compute."""
    paths=[directory/'training.jsonl',directory/'updates.jsonl']
    extra={};archive=directory/('superseded_'+str(time.time_ns()))
    for p in paths:
        if not p.exists():continue
        retained=[];removed=[]
        with p.open() as handle:
            for line in handle:
                try:r=json.loads(line)
                except json.JSONDecodeError:removed.append(line);continue
                valid=(r['episode']<=episode if p.name=='training.jsonl' else r['updates']<=updates)
                (retained if valid else removed).append(line)
        if removed:
            archive.mkdir(exist_ok=True);shutil.copy2(p,archive/p.name)
            p.write_text(''.join(retained));extra[p.name]=len(removed)
    if extra:write(archive/'accounting.json',dict(restored_episode=episode,restored_updates=updates,extra_records=extra))
    artifacts=directory/'episodes'
    uncommitted=set()
    if artifacts.exists():
        for p in artifacts.iterdir():
            try:n=int(p.name.split('.')[0])
            except ValueError:continue
            if n>episode:
                uncommitted.add(n)
                archive.mkdir(exist_ok=True);(archive/'episodes').mkdir(exist_ok=True)
                shutil.move(str(p),str(archive/'episodes'/p.name))
    if uncommitted:
        write(archive/'uncommitted_rollouts.json',dict(episodes=sorted(uncommitted),
            note='Speculative CPU rollouts not committed to retained replay; zero corresponding retained SAC updates'))

def shared_initialization(root,plan,pool,stop):
    cfg=configuration(root/'dataset/native_dataset.json',seed=plan['seed'],preflight=plan['preflight'])
    directory=root/'shared_warmup';directory.mkdir(exist_ok=True)
    latest=directory/'latest.pth'
    if latest.exists():x,cfg,completed,_=restore_state(latest,microbatch=plan['microbatch']);recover_logs(directory,completed,x.updates)
    else:
        seed_episode(cfg.seed,0);x=NativeLearner(cfg,'cuda',plan['microbatch']);completed=0
        save_state(directory/'initial.pth',x,cfg,0,'shared_warmup')
        save_state(latest,x,cfg,0,'shared_warmup')
    warmup_writer=SummaryWriter(str(root/'tensorboard/shared_warmup'),flush_secs=30)
    def log_warmup(r):
        append(directory/'updates.jsonl',r)
        for key in ('policy_loss','q1_loss','q2_loss','alpha','entropy'):
            warmup_writer.add_scalar('losses/'+key,r[key],r['updates'])
    x.on_update=log_warmup
    target=2 if plan['preflight'] else 32
    while completed<target:
        if stop[0]:raise InterruptedError('Stop requested at warmup boundary')
        jobs=jobs_for(cfg,x,directory,completed+1,min(plan['workers'],target-completed))
        results=wave(pool,jobs)
        for item in results:
            completed+=1;commit_episode(x,cfg,directory,item,completed,warmup_writer)
        save_state(latest,x,cfg,completed,'shared_warmup')
        plan.update(stage='easy_shared_warmup',warmup_episodes=completed,warmup_sac_updates=x.updates,updated=now())
        write(root/'pipeline.json',plan)
        print(json.dumps(dict(stage=plan['stage'],episodes=completed,updates=x.updates)),flush=True)
    # Reuse trained native policy as a frozen graph context, then train the head.
    marker=root/'head_complete.json'
    warmup_writer.close()
    if not marker.exists():
        collection=root/'head_collection';collection.mkdir(exist_ok=True)
        x.context_encoder.load_state_dict(x.policy.state_dict())
        label_count=2 if plan['preflight'] else 16
        datasets=[]
        for first in range(1,label_count+1,plan['workers']):
            if stop[0]:raise InterruptedError('Stop requested during label collection')
            episodes=list(range(first,min(label_count+1,first+plan['workers'])))
            missing=[ep for ep in episodes if not (collection/('labels_%03d.pth'%ep)).exists()]
            if missing:
                all_jobs=jobs_for(cfg,x,collection,first,len(episodes),collect_labels=True)
                jobs=[job for job in all_jobs if job['episode'] in missing]
                for job,item in zip(jobs,wave(pool,jobs)):
                    assert item['labels'],'No supervised map-packet samples'
                    label_path=collection/('labels_%03d.pth'%job['episode'])
                    torch.save(item['labels'],label_path.with_suffix('.tmp'));label_path.with_suffix('.tmp').replace(label_path)
            for ep in episodes:datasets.append(torch.load(collection/('labels_%03d.pth'%ep),map_location='cpu'))
            plan.update(stage='easy_head_collection',head_collection_maps=len(datasets),updated=now());write(root/'pipeline.json',plan)
        plan.update(stage='easy_head_fit',updated=now());write(root/'pipeline.json',plan)
        steps=4 if plan['preflight'] else 5000
        head_latest=collection/'latest.pth'
        completed_head=0
        if head_latest.exists():
            x,_,completed_head,_=restore_state(head_latest,microbatch=plan['microbatch'])
            log=collection/'head_updates.jsonl'
            if log.exists():
                lines=log.read_text().splitlines(True)
                keep=[line for line in lines if json.loads(line)['content_updates']<=completed_head]
                if len(keep)!=len(lines):
                    archive=collection/('superseded_head_'+str(time.time_ns())+'.jsonl')
                    shutil.copy2(log,archive);log.write_text(''.join(keep))
        else:
            save_state(head_latest,x,cfg,0,'head_fit')
        # Sampling is a pure function of update index, so checkpoint recovery has
        # exactly the same map-balanced supervised batches without replaying draws.
        for step in range(completed_head,steps):
            if stop[0]:
                save_state(head_latest,x,cfg,step,'head_fit')
                raise InterruptedError('Head fitting checkpointed at update boundary')
            head_rng=np.random.RandomState((cfg.seed+313+step*100003)%2**32)
            batch=[]
            for _ in range(cfg.batch_size):
                data=datasets[head_rng.randint(len(datasets))];batch.append(data[head_rng.randint(len(data))])
            r=x.update_content(batch)
            append(collection/'head_updates.jsonl',r)
            if (step+1)%1000==0:
                save_state(head_latest,x,cfg,step+1,'head_fit')
                plan.update(head_updates=step+1,updated=now());write(root/'pipeline.json',plan)
        x.replay.clear()
        save_state(root/'common_start.pth',x,cfg,0,'common_native_easy')
        write(marker,dict(head_updates=x.content_updates,sha256=digest(root/'common_start.pth'),
                          native_label_maps=label_count,source_sac_updates=x.updates))
    del x

def group_config(cfg,name):
    cfg=Config.from_checkpoint(cfg.to_dict());cfg.variant,cfg.content=GROUPS[name];return cfg

def prepare(root,args):
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh native output required')
    root.mkdir(parents=True)
    (root/'dataset').mkdir();shutil.copy2(args.dataset,root/'dataset/native_dataset.json')
    plan=dict(status='prepared',created=now(),stage='easy_shared_warmup',seed=args.seed,
        groups=list(GROUPS),workers=args.workers,microbatch=args.microbatch,
        effective_batch=16 if args.preflight else 128,replay_warmup=16 if args.preflight else 2000,
        updates_per_completed_episode=8,source_identity=identity(),dataset_sha256=digest(root/'dataset/native_dataset.json'),
        stage2_requires_easy_review=True,preflight=args.preflight,notifications=False,
        source_origin='Reviewed finite-contact implementation + original driver update/collection schedule + native hierarchical graph',
        original_training_file_counts={k:v['original_training_files'] for k,v in read(args.dataset)['families'].items()},
        development_maps={k:len(v['splits']['validation']) for k,v in read(args.dataset)['families'].items()},
        test_policy_evaluations=0,training_status='Easy stage continuous; no 1000-map or one-sweep completion rule',
        review_interval=args.review_interval,wall_guard_hours=args.max_hours,
        cross_method_prefetch=not args.no_prefetch,
        task='two robots, equal paid finite-contact protocol in both methods',
        methods_note='IR2_FIFO is a capacity-matched IR2-style map-difference/FIFO control, not unchanged unlimited-merger upstream IR2',
        preserved='original images, graph classes, cyclic file ordering, SAC/replay schedule, easy-to-difficult curriculum',
        necessary_task_differences='finite-byte radio, explicit time/WAIT, 12 local features, two-robot paired task, duration-aware Bellman target')
    write(root/'pipeline.json',plan)
    write(root/'source_identity.json',plan['source_identity'])
    snapshot=root/'source'
    for name in plan['source_identity']:
        p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,p)
    return plan

def evaluate_pair(root,plan,pool,learners,configs,completed):
    point=min(completed.values());results={}
    for name,x in learners.items():
        cfg=configs[name];directory=root/name/('validation_%08d'%point)
        count=min(2,cfg.validation_maps) if plan['preflight'] else cfg.validation_maps
        records=[]
        for start in range(0,count,plan['workers']):
            indices=list(range(start,min(start+plan['workers'],count)))
            jobs=jobs_for(cfg,x,directory,start+1,len(indices),evaluation=True,indices=indices)
            if plan['preflight']:
                for job in jobs:job['split']='train'
            records.extend(r['metric'] for r in wave(pool,jobs))
        results[name]=dict(successes=sum(r['success'] for r in records),maps=count,
            mean_capped_time=float(np.mean([r['capped_time'] for r in records])),
            mean_actual_time=float(np.mean([r['time'] for r in records])),sac_updates=x.updates)
        write(directory/'summary.json',results[name])
    report=dict(time=now(),easy_episodes_per_method=point,groups=results,test_policy_evaluations=0,
                scope='Training-map engineering preflight' if plan['preflight'] else 'Fixed geometry-group-held-out native easy development maps')
    write(root/'latest_comparison.json',report)
    plan['last_review_episode']=point
    return report

def train(root,args):
    plan=read(root/'pipeline.json')
    assert plan['source_identity']==identity(),'Frozen training source changed'
    assert plan['dataset_sha256']==digest(root/'dataset/native_dataset.json')
    lock=(root/'.coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    global_lock=(root.parent/'.native_training.lock').open('a');fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/'coordinator.json',dict(pid=os.getpid(),time=now()))
    torch.set_num_threads(2)
    stop=[False]
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:stop.__setitem__(0,True))
    started=time.monotonic()
    pool=ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker)
    writers={};learners={};configs={};completed={};rngs={}
    try:
        plan.update(status='running',updated=now());write(root/'pipeline.json',plan)
        if not (root/'head_complete.json').exists():
            # Partial label fitting is not silently retried as new training.
            p=root/'head_collection/head_updates.jsonl'
            if p.exists() and p.stat().st_size and not (root/'head_collection/latest.pth').exists():
                raise RuntimeError('Partial head fit found; explicit accounting required before retry')
            shared_initialization(root,plan,pool,stop)
        assert digest(root/'common_start.pth')==read(root/'head_complete.json')['sha256']
        for name in GROUPS:
            directory=root/name;directory.mkdir(exist_ok=True)
            latest=directory/'latest.pth'
            if latest.exists():
                x,cfg,ep,rng=restore_state(latest,microbatch=plan['microbatch'])
                recover_logs(directory,ep,x.updates)
            else:
                x,cfg,_,rng=restore_state(root/'common_start.pth',microbatch=plan['microbatch'])
                cfg=group_config(cfg,name);x.config=cfg;ep=0
                x.replay.clear();x.success_episodes=[]
                rng=save_state(directory/'initial.pth',x,cfg,0,name)
                save_state(latest,x,cfg,0,name)
            x.content.requires_grad_(False).eval();x.context_encoder.requires_grad_(False).eval()
            assert (cfg.variant,cfg.content)==GROUPS[name]
            assert cfg.dataset_manifest==str((root/'dataset/native_dataset.json').resolve())
            learners[name]=x;configs[name]=cfg;completed[name]=ep;rngs[name]=rng
            writers[name]=SummaryWriter(str(root/'tensorboard'/name),flush_secs=30)
            def log_update(r,d=directory,w=writers[name]):
                append(d/'updates.jsonl',r)
                for key in ('policy_loss','q1_loss','q2_loss','alpha','entropy','microbatch','max_nodes'):
                    w.add_scalar('losses/'+key,r[key],r['updates'])
            x.on_update=log_update
        source_updates=read(root/'head_complete.json')['source_sac_updates']
        if any(x.updates>source_updates for x in learners.values()):
            plan['paired_learning_enabled']=True
        elif len(set(completed.values()))==1 and min(len(x.replay) for x in learners.values())>=next(iter(configs.values())).warmup:
            plan['paired_learning_enabled']=True
            plan.setdefault('paired_warmup_episodes',min(completed.values()))
        plan.update(stage='easy_paired_training',updated=now());write(root/'pipeline.json',plan)
        pending=None
        while not stop[0]:
            if (root/'STOP').exists():break
            if time.monotonic()-started>plan['wall_guard_hours']*3600:break
            if pending is None:
                name=min(GROUPS,key=lambda n:completed[n]);x=learners[name];cfg=configs[name]
                count=plan['workers']
                if plan['preflight']:count=min(count,4-completed[name])
                if count<=0:break
                jobs=jobs_for(cfg,x,root/name,completed[name]+1,count)
                pending=(name,[pool.submit(run_episode,job) for job in jobs])
            name,futures=pending;pending=None
            results=[future.result() for future in futures]
            x=learners[name];cfg=configs[name];directory=root/name
            hypothetical=completed.copy();hypothetical[name]+=len(results)
            review_due=(len(set(hypothetical.values()))==1 and min(hypothetical.values())>=plan.get('last_review_episode',0)+plan['review_interval'])
            finished_probe=plan['preflight'] and min(hypothetical.values())>=4
            # The other model is unchanged by the current model's optimizer.
            # Snapshot it now, overlap its CPU sampling with current GPU updates,
            # and preserve exactly the weights an unprefetched alternating loop uses.
            next_name=min(GROUPS,key=lambda n:hypothetical[n])
            if plan.get('cross_method_prefetch',False) and next_name!=name and not review_due and not finished_probe and not stop[0]:
                count=plan['workers']
                if plan['preflight']:count=min(count,4-hypothetical[next_name])
                jobs=jobs_for(configs[next_name],learners[next_name],root/next_name,hypothetical[next_name]+1,count)
                pending=(next_name,[pool.submit(run_episode,job) for job in jobs])
            restore_rng(rngs[name])
            for item in results:
                completed[name]+=1
                commit_episode(x,cfg,directory,item,completed[name],writers[name],
                               enable_updates=plan.get('paired_learning_enabled',False))
            rngs[name]=save_state(directory/'latest.pth',x,cfg,completed[name],name)
            state=dict(status='training',episodes=completed[name],sac_updates=x.updates,
                       head_updates=x.content_updates,replay_size=len(x.replay),updated=now(),
                       last_map=results[-1]['metric']['map']['source_file'],
                       last_worker_peak_rss_mib=max(r['worker_rss_mib'] for r in results))
            write(directory/'status.json',state)
            if not plan.get('paired_learning_enabled') and len(set(completed.values()))==1 and min(len(v.replay) for v in learners.values())>=cfg.warmup:
                plan.update(paired_learning_enabled=True,paired_warmup_episodes=min(completed.values()))
            if len(set(completed.values()))==1:
                assert len({v.updates for v in learners.values()})==1,'Paired SAC budgets diverged'
            plan.update(active_group=name,completed_episodes=completed.copy(),sac_updates={g:v.updates for g,v in learners.items()},updated=now())
            write(root/'pipeline.json',plan)
            print(json.dumps(dict(stage='easy',group=name,episode=completed[name],updates=x.updates)),flush=True)
            point=min(completed.values())
            if plan['preflight'] and point>=4:
                evaluate_pair(root,plan,pool,learners,configs,completed);break
            if point>=plan.get('last_review_episode',0)+plan['review_interval'] and len(set(completed.values()))==1:
                plan.update(stage='easy_development_review',updated=now());write(root/'pipeline.json',plan)
                evaluate_pair(root,plan,pool,learners,configs,completed)
                plan['stage']='easy_paired_training';write(root/'pipeline.json',plan)
        plan.update(status='preflight_complete' if plan['preflight'] else 'paused_for_easy_review',
                    stage='easy',updated=now(),stage2_started=False)
        write(root/'pipeline.json',plan)
    except BaseException as exc:
        plan.update(status='interrupted_or_failed',error=repr(exc),updated=now())
        write(root/'pipeline.json',plan)
        raise
    finally:
        for writer in writers.values():writer.close()
        pool.shutdown(wait=True)
        if 'pending' in locals() and pending is not None:
            role,futures=pending
            episodes=[]
            for future in futures:
                if not future.cancelled() and future.exception() is None:
                    episodes.append(future.result()['metric']['episode'])
            write(root/'uncommitted_cpu_work.json',dict(group=role,episodes=episodes,retained_sac_updates=0,
                note='CPU prefetch completed after the last committed training boundary; archive on resume'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--microbatch',type=int,default=32)
    p.add_argument('--seed',type=int,default=101);p.add_argument('--preflight',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-hours',type=float,default=48)
    p.add_argument('--review-interval',type=int,default=512)
    p.add_argument('--no-prefetch',action='store_true',help='Disable exact cross-method CPU/GPU overlap for engineering comparison')
    args=p.parse_args();root=args.output.resolve()
    if not args.resume:
        if args.dataset is None:p.error('--dataset is required for a new run')
        prepare(root,args)
    train(root,args)

if __name__=='__main__':main()
