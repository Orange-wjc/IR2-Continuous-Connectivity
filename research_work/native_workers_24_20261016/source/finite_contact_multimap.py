#!/usr/bin/env python3
"""Bounded first 100-map movement pilot and held-out development comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parent


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initialize-from',type=Path,help='Optional movement warm start; replay and optimizer reset')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--map-seed',type=int,default=20261011)
    p.add_argument('--sweeps',type=int,default=1)
    p.add_argument('--max-additional-updates',type=int,default=20000,help='Safety cap, not a convergence promise')
    p.add_argument('--max-hours',type=float,default=.5,help='Movement-stage wall budget')
    p.add_argument('--sensor-implementation',choices=['legacy','vectorized'],default='vectorized')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if Path(sys.prefix).resolve()!=Path('/home/wyz/anaconda3/envs/ir2_train').resolve(): p.error('Use ir2_train Python')
    if args.sweeps<1 or args.max_additional_updates<1 or args.max_hours<=0: p.error('Positive budgets required')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()): p.error('Use a new output directory')
    source=args.initialize_from.resolve() if args.initialize_from else None
    source_updates=0
    if source:
        saved=torch.load(source,map_location='cpu')
        if saved.get('format')!='ir2-finite-contact-v1' or saved['stage']!='movement':
            p.error('Warm start must be a finite-contact movement checkpoint')
        source_updates=saved['learner']['updates']
    movement=output/'movement_A'
    python=[sys.executable]
    train=python+[str(ROOT/'finite_contact_train.py'),'--stage','movement','--variant','A','--content','fifo',
        '--map-family','simple_rooms_v1','--map-seed',str(args.map_seed),'--train-maps','100',
        '--validation-maps','20','--test-maps','20','--map-order','shuffled','--seed',str(args.seed),
        '--sensor-implementation',args.sensor_implementation,
        '--success-imitation-weight','1','--success-memory-mode','per_map','--success-episode-limit','100',
        '--replay-size','12000','--policy-learning-rate','.0001','--alpha-learning-rate','.0001',
        '--horizon-target','terminal','--pointer-mode','centered','--frontier-shaping-scale','0',
        '--coverage-reward','union','--episodes',str(args.sweeps*100),
        '--max-updates',str(source_updates+args.max_additional_updates),'--max-hours',str(args.max_hours),
        '--eval-train','--eval-train-maps','10','--eval-every','50','--eval-maps','5',
        '--final-eval','--device','cuda','--run-dir',str(movement)]
    if source: train+=['--initialize-from',str(source)]
    jobs=[('dataset',python+[str(ROOT/'finite_contact_dataset.py'),'--output',str(output/'dataset'),
                             '--map-seed',str(args.map_seed)]),('movement',train)]
    for label,checkpoint in [('validation_initial','initial.pth'),('validation_final','final.pth')]:
        jobs.append((label,python+[str(ROOT/'finite_contact_train.py'),'--stage','evaluate',
            '--initialize-from',str(movement/checkpoint),'--split','validation','--eval-maps','20',
            '--max-hours','.3','--device','cuda','--run-dir',str(output/label)]))
    if args.dry_run:
        for name,command in jobs: print(name+': '+shlex.join(command))
        return
    output.mkdir(parents=True)
    plan={'scope':'100-map movement feasibility pilot, not AB innovation validation; no test policy evaluation.',
          'seed':args.seed,'dataset_seed':args.map_seed,'training_sweeps':args.sweeps,
          'source':str(source) if source else None,'source_sha256':
          hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
          'source_updates':source_updates,'steps':[], 'status':'running'}
    def save(): (output/'pipeline.json').write_text(json.dumps(plan,indent=2))
    save()
    try:
        for name,command in jobs:
            record={'name':name,'command':command,'status':'running'};plan['steps'].append(record);save()
            print('Starting '+name,flush=True)
            with (output/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                try:
                    for line in child.stdout:
                        log.write(line);log.flush()
                        try: item=json.loads(line)
                        except json.JSONDecodeError: continue
                        brief={k:item[k] for k in ('episode','learner_updates','success','coverage','audited_maps','evaluated_rollouts','complete') if k in item}
                        if brief: print(json.dumps({'step':name,**brief}),flush=True)
                    code=child.wait()
                except KeyboardInterrupt:
                    child.send_signal(signal.SIGTERM);child.wait();raise
            if code: raise RuntimeError(f'{name} failed; see {name}.log')
            if name=='movement':
                training=rows(movement/'training.jsonl')
                complete=[r for r in training if not r['interrupted']]
                assert len(complete)==args.sweeps*100,'Budget ended before complete training sweeps'
                assert {r['map']['index'] for r in complete}==set(range(100)),'Missing training maps'
                plan['completed_training_episodes']=len(complete)
                plan['successful_training_episodes']=sum(r['success'] for r in complete)
                plan['final_updates']=training[-1]['learner_updates']
            if name.startswith('validation_'):
                results=rows(output/name/'evaluation.jsonl')
                assert len(results)==20 and {r['map']['index'] for r in results}==set(range(20))
                assert all(not r['interrupted'] for r in results)
                plan[name]={'successes':sum(r['success'] for r in results),'count':20,
                    'mean_minimum_coverage':sum(min(r['coverage']) for r in results)/20}
            record['status']='complete';save()
        plan['status']='pilot_complete';save()
        print(json.dumps(plan,indent=2),flush=True)
    except (Exception,KeyboardInterrupt):
        plan['status']='interrupted_or_failed';save();raise


if __name__=='__main__': main()
