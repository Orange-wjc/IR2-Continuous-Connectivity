"""Behavioral integration checks; isolated outputs, never formal training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('IR2_METHOD','finite_learned')
os.environ.setdefault('IR2_RUN_NAME','contact_module_validation')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-contact-validation-mpl')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
import copy
from dataclasses import replace
import hashlib
import json
import random
import subprocess
import numpy as np
import torch
import parameter as config
from model import PolicyNet,QNet
from baseline_replay import PackedMask
from baseline_support import prepare_maps,write_json
from contact_module.config import ContactConfig
from contact_module.radio import Radio,MAP,STATE,CONFIRM,ACK
from contact_module.learning import ContentTrainer
from contact_module.adapter import ContactAdapter
import multi_robot_worker as workers


def seeds(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def protocol_checks():
    cfg=ContactConfig(method='finite_fifo',bytes_per_second=10.)
    beliefs=[np.full((4,4),127,np.uint8) for _ in range(3)]
    beliefs[0][0,0]=255
    positions=np.zeros((3,2))
    links=np.ones((3,3),bool);np.fill_diagonal(links,False)
    radio=Radio((4,4),3,cfg)
    for tick in range(6):
        before=radio.sent_bytes
        radio.service(beliefs,positions,tick,1.,links)
        assert np.all(radio.last_tx<=10) and np.all(radio.last_rx<=10)
        assert radio.sent_bytes-before<=30
        if radio.completed_maps==0:
            assert beliefs[1][0,0]==127 and beliefs[2][0,0]==127
    assert radio.sent_bytes>0
    zero=np.zeros((3,3),bool)
    before=radio.sent_bytes
    radio.service(beliefs,positions,6.,100.,zero)
    assert radio.sent_bytes==before and np.all(radio.credit==0)
    # First A->B delivery cannot already be forwarded to C in the same slice.
    chain=np.array([[0,1,0],[1,0,1],[0,1,0]],bool)
    beliefs=[np.full((4,4),127,np.uint8) for _ in range(3)];beliefs[0][0,0]=255
    r=Radio((4,4),3,replace(cfg,bytes_per_second=1000.))
    r.service(beliefs,positions,0.,1.,chain)
    assert beliefs[1][0,0]==255 and beliefs[2][0,0]==127
    assert r.known_peer[(0,1)][0,0]==127  # ACK not free/same slice.
    r.service(beliefs,positions,1.,1.,chain)
    assert r.known_peer[(0,1)][0,0]==255 and beliefs[2][0,0]==255
    assert r.control_bytes>0
    # Interrupted map packet must finish before the ranker chooses another.
    cfg=replace(cfg,bytes_per_second=8.)
    b=[np.full((4,4),127,np.uint8) for _ in range(2)];b[0][:]=255
    r=Radio((4,4),2,cfg);r.enqueue((0,1),b[0],0.)
    r.last_pose_sent={(0,1):0.,(1,0):0.}
    r.service(b,np.zeros((2,2)),0.,1.,np.array([[0,1],[1,0]],bool))
    active=r.active[(0,1)];assert active is not None and active.kind==MAP
    old_offset=active.offset
    r.service(b,np.zeros((2,2)),1.,1.,np.zeros((2,2),bool))
    assert active.offset==old_offset and np.all(b[1]==127)
    r.service(b,np.zeros((2,2)),2.,1.,np.array([[0,1],[1,0]],bool))
    assert r.active[(0,1)] is active and active.offset==old_offset+8


def original_equivalence():
    # Load the pushed pre-module baseline classes without writing a checkout.
    namespaces={}
    for name in ('env','multi_robot_worker'):
        source=subprocess.check_output(['git','show','dcea459d:'+name+'.py'],text=True)
        scope={'__name__':'reference_'+name}
        exec(compile(source,'reference_'+name+'.py','exec'),scope)
        namespaces[name]=scope
    old=namespaces['multi_robot_worker'];old['Env']=namespaces['env']['Env'];old['MAX_EPS_STEPS']=2
    seeds(101);p=PolicyNet(6,128);q=QNet(6,128)
    seeds(202);a=old['Worker'](0,3,copy.deepcopy(p),copy.deepcopy(q),1,device='cpu')
    assert a.run_episode(1)
    previous=workers.MAX_EPS_STEPS;workers.MAX_EPS_STEPS=2
    try:
        seeds(202);b=workers.Worker(0,3,copy.deepcopy(p),copy.deepcopy(q),1,device='cpu')
        assert b.run_episode(1)
    finally:workers.MAX_EPS_STEPS=previous
    assert a.perf_metrics==b.perf_metrics
    for left,right in zip(a.episode_buffer,b.episode_buffer):
        assert len(left)==len(right)
        for x,y in zip(left,right):
            if isinstance(x,PackedMask):x=x.unpack()
            if isinstance(y,PackedMask):y=y.unpack()
            assert x.dtype==y.dtype and torch.equal(x,y)
    return True


def integrated_checks():
    seeds(303)
    p=PolicyNet(6,128,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim);q=QNet(6,128,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim)
    trainer=ContentTrainer(p,config.CONTACT_CONFIG)
    before={k:v.clone() for k,v in trainer.head.state_dict().items()}
    contact=ContactAdapter(config.CONTACT_CONFIG,p,trainer.weights(),1)
    prev=workers.MAX_EPS_STEPS;workers.MAX_EPS_STEPS=2
    try:
        w=workers.Worker(0,3,p,q,1,device='cpu',contact=contact)
        assert w.run_episode(1)
    finally:workers.MAX_EPS_STEPS=prev
    assert len(w.episode_buffer[0])==6
    assert contact.time>0 and contact.radio.sent_bytes>0 and len(contact.labels)>0
    for obs_i in (0,9):
        for a in w.episode_buffer[obs_i]:assert a.shape[-1]==6+config.CONTACT_CONFIG.state_dim
    obs,ok=w.get_observations(w.robot_list[0].robot_position,0,1,2)
    assert ok and obs[1][0,0,0].item()==obs[2].item()
    logp=p(*[v.clone() for v in obs]);assert torch.isfinite(logp).all()
    assert logp.exp()[0,0]>0 and abs(float(logp.exp().sum())-1)<1e-5
    # Bootstrap cannot rank with a random head; joint stage must use the head.
    assert not contact.content_ready
    trainer.ready = True
    contact.content_ready = True
    # Legal ranking and utility do not consult the receiver's current map.
    pair=(0,1);packets=contact.radio.pending(pair)[:8]
    if not packets:
        # Create known sender-owned cells; this is an isolated fixture only.
        contact.radio.offered[pair].fill(127);contact.radio.known_peer[pair].fill(127)
        contact.radio.enqueue(pair,w.env.all_robot_belief[0][0],contact.time)
        packets=contact.radio.pending(pair)[:8]
    assert packets
    score=contact.scores(pair,packets).copy();delta,utility,_=contact.utility(w.env,0)
    target=w.env.all_robot_belief[1][1];saved=target.copy();target.fill(255)
    assert np.array_equal(score,contact.scores(pair,packets))
    assert np.array_equal(utility,contact.utility(w.env,0)[1]);target[:]=saved
    labels=contact.labels
    loss=trainer.update((labels*128)[:256]);assert loss is not None and np.isfinite(loss)
    assert trainer.updates==4
    assert any(not torch.equal(v,before[k]) for k,v in trainer.head.state_dict().items())
    resumed=ContentTrainer(p,config.CONTACT_CONFIG,trainer.state_dict())
    assert resumed.updates==4 and len(resumed.replay)==len(trainer.replay)
    for k,v in trainer.head.state_dict().items():assert torch.equal(v,resumed.head.state_dict()[k])
    # Same replay/RNG/Adam continuation produces exactly the same next update.
    trainer.update([]);resumed.update([])
    for k,v in trainer.head.state_dict().items():assert torch.equal(v,resumed.head.state_dict()[k])
    return dict(metrics=contact.metrics(),labels=len(labels),head_updates=trainer.updates,
                loss=loss,movement_transitions=len(w.episode_buffer[0]))


def control_mode_checks():
    previous=workers.MAX_EPS_STEPS;workers.MAX_EPS_STEPS=1
    try:
        for mode in ('finite_fifo','finite_rule'):
            seeds(404)
            p=PolicyNet(6,128,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim);q=QNet(6,128,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim)
            contact=ContactAdapter(ContactConfig(method=mode),p,None,1)
            w=workers.Worker(0,3,p,q,1,device='cpu',contact=contact)
            assert w.run_episode(1) and len(w.episode_buffer[0])==3
            assert contact.radio.sent_bytes>0 and not contact.labels
    finally:workers.MAX_EPS_STEPS=previous


def main():
    torch.set_num_threads(1)
    protected=[Path('model/2/checkpoint.pth'),Path('experiments/2/progress.jsonl')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    prepare_maps(config);protocol_checks();print('Protocol behavior PASS',flush=True)
    original_equivalence();print('Original disabled-mode equivalence PASS',flush=True)
    detail=integrated_checks();control_mode_checks()
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    result=dict(passed=True,production_files_unchanged=hashes,protocol_budget_and_no_shortcuts=True,
                original_mode_15_columns_and_metrics_exact=True,
                sender_score_and_utility_no_receiver_truth=True,wait_has_positive_time=True,
                auxiliary_head_gradient_and_resume_exact=True,integrated=detail,
                finite_fifo_and_rule_sampling_passed=True,
                scope='Isolated engineering rollouts and auxiliary updates; not a learning/effectiveness experiment')
    write_json(Path('baseline_checks/contact_module_verified.json'),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
