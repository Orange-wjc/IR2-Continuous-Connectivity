"""Real Env regression: no delivery/new sensing must not pay an idle WAIT."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['IR2_METHOD']='finite_learned';os.environ['IR2_RUN_NAME']='contact_wait_validation'
os.environ['MPLCONFIGDIR']='/tmp/ir2-contact-validation-mpl'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
import numpy as np
import parameter as cfg
from baseline_support import prepare_maps,write_json
from model import PolicyNet,QNet
from contact_module.adapter import ContactAdapter
from contact_module.learning import ContentTrainer
from contact_module.preparation import load_initialization,load_content_initialization
from multi_robot_worker import Worker


def main():
    torch.set_num_threads(1);manifest=prepare_maps(cfg)
    source,init=load_initialization('model/2/checkpoint.pth',manifest)
    saved,audit=load_content_initialization(
        'experiments/contact_easy10_v2_20261019/prepared_initialization.pth',
        cfg.CONTACT_CONFIG,init,source['policy_model'])
    assert saved['ready'] and saved['updates']==1000
    p=PolicyNet(6,128,allow_wait=True);p.load_state_dict(source['policy_model'])
    q=QNet(6,128,allow_wait=True);q.load_state_dict(source['q_net1_model'])
    c=ContactAdapter(cfg.CONTACT_CONFIG,p,ContentTrainer(p,cfg.CONTACT_CONFIG,saved).weights(),4)
    w=Worker(0,3,p,q,4,device='cpu',contact=c)
    # Fixed predicted gain can be wrong when peers are disconnected. It must
    # still not yield an idle reward, while other nodes retain guidance.
    c.expected_delivery=lambda pair:500.
    w.env.ss_realistic_model.is_within_signal_strength=lambda *a,**k:False
    pos=w.all_robot_positions[0]
    rewards=[]
    for step in range(3):
        w.env.all_robot_positions_belief[0][0]=pos
        obs,valid=w.get_observations(pos,0,4,step);assert valid
        assert float(obs[0][0,int(obs[2].item()),3])==0.
        valid,reward,_=w.env.single_robot_step(0,w.all_robot_positions,4,step,0.)
        assert valid;rewards.append(float(reward))
    assert np.allclose(rewards[1:],[-.02,-.02])
    assert c.positive_empty_waits==0 and c.empty_waits>=2
    assert c.radio.sent_bytes==0 and c.last_delivery_novel==0
    # A useful communication WAIT is still eligible for actual delivery reward.
    before=c.empty_waits;c.last_delivery_novel=100;c.last_delivery_reward=.1
    c.record_reward(0.,100,.08,0.,0.,0.)
    assert c.empty_waits==before and c.positive_empty_waits==0
    result=dict(passed=True,repeated_empty_wait_rewards=rewards[1:],
        empty_wait_positive_rewards=c.positive_empty_waits,useful_wait_not_flagged=True,
        content_init_sha256=audit['sha256'],shared_prepared_head_exact=True,
        scope='Isolated deterministic regression, not formal training')
    write_json('baseline_checks/contact_wait_reward_verified.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
