"""Sender credit conservation and own-decision replay continuity.

Optional frozen real-map comparison also verifies identical physical actions,
policy observations and wire traffic against the archived v5 collector. No
optimization, and no diagnostic success contributes to formal acceptance.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
os.environ['IR2_METHOD']='finite_learned'
os.environ['IR2_RUN_NAME']='contact_transition_validation'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-transition-check-mpl')
from pathlib import Path
import argparse, hashlib, json, random, subprocess, sys, types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from contact_module.transitions import DecisionTransitions
from contact_module.config import ContactConfig
from contact_module.radio import Radio
from baseline_support import write_json


class Sink:
    def __init__(self): self.episode_buffer=[[] for _ in range(15)];self.rewards=[];self.next=[]
    def save_next_observations(self, obs): self.next.append(obs)
    def save_reward_done(self, reward, done):self.rewards.append((reward,done))


def fixtures():
    # Robot 0 is disconnected; sender 1 delivers to 2 during robot 0's slot.
    cfg=ContactConfig(method='finite_fifo')
    radio=Radio((4,4),3,cfg)
    beliefs=[np.full((4,4),127,np.uint8) for _ in range(3)]
    beliefs[1][0,0]=255
    links=np.zeros((3,3),bool);links[1,2]=links[2,1]=True
    gains=radio.service(beliefs,np.zeros((3,2)),0.,1.,links)
    assert gains.tolist()==[0,1,0] and radio.unique_cells==1
    assert gains[0]==0  # The previous implementation would lose this event.
    adapter=types.SimpleNamespace(delivery_cells=np.zeros(3),claimed_delivery_cells=np.zeros(3),
        last_delivery_reward=0.,reward_sums={'delivery':0.},radio=radio)
    robots=[Sink() for _ in range(3)];c=DecisionTransitions(adapter,robots)
    for i in range(3):c.begin(i,'initial');c.action_reward(i,-.02)
    adapter.delivery_cells+=gains
    c.team_reward(0.)
    for i in range(3):c.begin(i,'next');c.action_reward(i,-.02)
    assert robots[1].rewards[0]==(-.019,False)
    assert robots[0].rewards[0]==(-.02,False)
    c.team_reward(40.);c.finish(['terminal']*3,True)
    assert all(r.rewards[-1]==(39.98,True) for r in robots)
    assert c.settled_delivery_cells==1 and c.initial_delivery_cells==0
    assert adapter.delivery_ledger_residual==0
    # A genuine delivery before an agent has chosen anything is an explicit
    # initial condition, never reward for a future action.
    adapter.delivery_cells=np.array([0.,1.,0.]);adapter.claimed_delivery_cells=np.zeros(3)
    adapter.reward_sums={'delivery':0.}
    c=DecisionTransitions(adapter,[Sink() for _ in range(3)])
    for i in range(3):c.begin(i,'initial');c.action_reward(i,-.02)
    c.team_reward(0.);c.finish(['timeout']*3,False)
    assert c.initial_delivery_cells==1 and c.settled_delivery_cells==0
    assert adapter.delivery_ledger_residual==0
    return dict(off_slot_delivery_credited_once=True,terminal_and_timeout_preserved=True,
                initial_condition_not_credited_to_future_action=True)


def frozen(checkpoint,steps):
    import parameter as config
    from baseline_support import prepare_maps
    from contact_module.adapter import ContactAdapter
    from model import PolicyNet,QNet
    import multi_robot_worker as workers
    prepare_maps(config)
    state=torch.load(checkpoint,map_location='cpu')
    legacy=types.ModuleType('v5_worker')
    exec(compile(subprocess.check_output(['git','show','420794dd:multi_robot_worker.py'],text=True),
                 'archived_v5_worker','exec'),legacy.__dict__)
    results=[]
    for module in (legacy,workers):
        random.seed(20261019);np.random.seed(20261019);torch.manual_seed(20261019)
        p=PolicyNet(6,128,allow_wait=True);q=QNet(6,128,allow_wait=True)
        p.load_state_dict(state['policy_model']);q.load_state_dict(state['q_net1_model']);p.eval();q.eval()
        contact=ContactAdapter(config.CONTACT_CONFIG,p,state['contact_learning'],0)
        w=module.Worker(0,3,p,q,0,device='cpu',contact=contact)
        decisions=[];events=[]
        old_select=w.select_node;old_service=contact.radio.service
        def select(obs,robot):
            position,action=old_select(obs,robot)
            decisions.append((robot,position.tolist(),action.tolist(),
                              [hashlib.sha256(t.detach().numpy().tobytes()).hexdigest() for t in obs]))
            return position,action
        def service(*args,**kwargs):
            gains=old_service(*args,**kwargs);events.append(gains.tolist());return gains
        w.select_node=select;contact.radio.service=service;module.MAX_EPS_STEPS=steps
        assert w.run_episode(0)
        mismatches=0
        for robot in w.robot_list:
            b=robot.episode_buffer;assert len({len(c) for c in b})==1
            for k in range(len(b[0])-1):
                equal=True
                for col in range(6):
                    a,n=b[9+col][k],b[col][k+1]
                    if hasattr(a,'unpack'):a,n=a.unpack(),n.unpack()
                    equal=equal and torch.equal(a,n)
                mismatches+=not equal
        results.append(dict(decisions=decisions,events=events,
            maps=[hashlib.sha256(contact.env.all_robot_belief[i][i].tobytes()).hexdigest() for i in range(3)],
            bytes=contact.radio.sent_bytes,control_bytes=contact.radio.control_bytes,
            unique=contact.radio.unique_cells,reward_delivery=contact.reward_sums['delivery'],
            initial=contact.initial_delivery_cells,residual=contact.delivery_ledger_residual,
            mismatches=mismatches,transitions=sum(len(r.episode_buffer[0]) for r in w.robot_list),
            total_reward=sum(float(v.item()) for r in w.robot_list for v in r.episode_buffer[7])))
    before,after=results
    for k in ('decisions','events','maps','bytes','control_bytes','unique'):assert before[k]==after[k],k
    assert after['mismatches']==0 and before['mismatches']>0
    assert after['residual']==0
    assert abs(after['reward_delivery']*1000+after['initial']-after['unique'])<1e-7
    assert abs(after['total_reward']-before['total_reward']-
               (after['reward_delivery']-before['reward_delivery']))<1e-4
    return dict(frozen_source=str(checkpoint),steps=steps,
        physical_actions_observations_deliveries_and_wire_bytes_identical=True,
        previous_next_state_mismatches=before['mismatches'],corrected_next_state_mismatches=0,
        unique_delivery_cells=after['unique'],previous_credited_cells=before['reward_delivery']*1000,
        corrected_credited_cells=after['reward_delivery']*1000,initial_cells=after['initial'],
        reward_residual=after['residual'],all_15_columns_complete=True,
        only_actual_missing_delivery_added_to_total_reward=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');parser.add_argument('--steps',type=int,default=12)
    args=parser.parse_args();torch.set_num_threads(1)
    report=dict(passed=True,fixtures=fixtures(),scope='Engineering verification, no optimization or acceptance')
    if args.checkpoint:report['frozen_comparison']=frozen(args.checkpoint,args.steps)
    write_json('baseline_checks/contact_transition_verified.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
