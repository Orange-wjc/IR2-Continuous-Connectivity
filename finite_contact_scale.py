#!/usr/bin/env python3
"""Run one declared 1000-map adaptation sweep with bounded process concurrency."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import torch
from finite_contact.config import Config
from finite_contact_dataset import audit_dataset
from finite_contact_multimap_joint import prepare, report, sha256
from finite_contact_multiseed import run_worker_queue
from finite_contact_stability import read, write, recover_initialization

ROOT=Path(__file__).resolve().parent
SOURCES=('finite_contact_scale.py','finite_contact_multimap_joint.py','finite_contact_train.py',
         'finite_contact_multiseed.py','finite_contact_stability.py','finite_contact/config.py',
         'finite_contact/learning.py','finite_contact/environment.py','finite_contact/maps.py',
         'finite_contact/sensing.py','finite_contact_dataset.py')


def execute(args):
    root=args.output.resolve()
    if not args.resume:
        prepare(root,args)
        shutil.copy2(__file__,root/'orchestration_source.py')
        shutil.copy2(ROOT/'finite_contact/THOUSAND_MAP_EXPERIMENT.md',root/'pre_run_design.md')
        identity={name:sha256(ROOT/name) for name in SOURCES}
        write(root/'source_identity_before.json',identity)
        plan=read(root/'pipeline.json')
        plan.update(scope='1000-map simple-room movement adaptation; one shared pretrained chain, one adaptation seed, frozen 100-map communication head.',
                    workers=1,test_policy_evaluations=0,
                    worker_order=['AB_learned','A_rule','A_fifo'],
                    reference_root=None,
                    success_sampling='pooled',compact_replay=True,
                    training_history='Source movement/head used original 100 training maps; no new test policy executions.')
        write(root/'pipeline.json',plan)
        saved=torch.load(root/'common_start.pth',map_location='cpu')
        cfg=Config.from_checkpoint(saved['config']);del saved
        audit_dataset(cfg,root/'dataset')
        write(root/'dataset_ready.json',{'passed':True,'manifest_sha256':sha256(root/'dataset/manifest.json')})
    else:
        plan=read(root/'pipeline.json')
        identity=read(root/'source_identity_before.json')
        if any(sha256(ROOT/name)!=digest for name,digest in identity.items()):
            raise ValueError('Executed source changed; explicit recovery review is required')
        if not (root/'dataset_ready.json').exists():
            if (root/'dataset').exists():
                (root/'dataset').rename(root/('dataset_incomplete_'+str(__import__('time').time_ns())))
            saved=torch.load(root/'common_start.pth',map_location='cpu')
            cfg=Config.from_checkpoint(saved['config']);del saved
            audit_dataset(cfg,root/'dataset')
            write(root/'dataset_ready.json',{'passed':True,'manifest_sha256':sha256(root/'dataset/manifest.json')})
    assert sha256(root/'common_start.pth')==plan['common_sha256']
    assert sha256(root/'dataset/manifest.json')==read(root/'dataset_ready.json')['manifest_sha256']
    assert plan['training_maps']==1000 and plan['new_sac_updates_per_group']==20000
    jobs=[]
    for group in plan['worker_order']:
        d=root/group
        if (d/'status.json').exists() and read(d/'status.json')['status']=='complete': continue
        if args.resume: recover_initialization(d)
        command=[sys.executable,str(ROOT/'finite_contact_multimap_joint.py'),'--stage','group',
                 '--group',group,'--output',str(root),'--device',args.device]
        if (d/'latest.pth').exists():command+=['--resume',str(d/'latest.pth')]
        jobs.append((root,group,command))
    plan['status']='running';write(root/'pipeline.json',plan)
    try:
        run_worker_queue(jobs,1,args.resume)
        report(root,plan)
        after={name:sha256(ROOT/name) for name in identity}
        assert after==identity
        write(root/'source_identity_after.json',after)
        plan['status']='complete'
    except BaseException:
        plan['status']='interrupted_or_failed'
        raise
    finally:write(root/'pipeline.json',plan)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=p.parse_args()
    if Path(sys.prefix)!=Path('/home/wyz/anaconda3/envs/ir2_train'):p.error('Use ir2_train')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    args.movement_state=ROOT/'experiments/multimap_20261011_vectorized_seed1/movement_A/latest.pth'
    args.content_checkpoint=ROOT/'experiments/multimap_content_20261012_seed1/head/final.pth'
    args.reference_root=None
    args.train_maps=1000;args.compact_replay=True
    args.policy_learning_rate=1e-5;args.seed=1;args.sweeps=1;args.updates_per_map=20;args.max_hours=12
    def stop(signum,frame):raise KeyboardInterrupt('Scale experiment interrupted')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    execute(args)


if __name__=='__main__':main()
