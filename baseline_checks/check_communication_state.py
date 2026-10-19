"""Local information, exact protocol, zero-init/RNG and adapter gradients.

CPU engineering checks only. Never optimizes or writes a formal model.
"""
from pathlib import Path
import sys,os,subprocess,types,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch
from contact_module.config import ContactConfig
from contact_module.radio import Radio
from contact_module.state import local_state,attach_state
from model import PolicyNet,QNet
from baseline_support import write_json


def legacy(name):
    module=types.ModuleType('v6_reference_'+name);sys.modules[module.__name__]=module
    exec(compile(subprocess.check_output(['git','show','1cfa3779:'+name+'.py'],text=True),name+'_v6','exec'),module.__dict__)
    return module


def canonical(value):
    if isinstance(value,np.ndarray):return value.dtype.str,value.shape,value.tobytes()
    if hasattr(value,'wire'):return tuple(getattr(value,k) for k in ('kind','sender','receiver','seq','created','wire','offset'))
    if isinstance(value,dict):return tuple((k,canonical(v)) for k,v in value.items())
    if isinstance(value,(list,tuple,deque)):return tuple(canonical(v) for v in value)
    return value


def protocol():
    reference=legacy('contact_module/radio');metadata={'last_receive_at','last_novel_receive_at','last_novel_receive_cells','last_novel_receive_free'}
    events=0
    for bandwidth in (10.,100.):
        cfg=ContactConfig(method='finite_fifo',bytes_per_second=bandwidth)
        old=reference.Radio((4,4),3,cfg);new=Radio((4,4),3,cfg)
        beliefs=[[np.full((4,4),127,np.uint8) for _ in range(3)] for _ in range(2)]
        for b in beliefs:b[0][0,0]=255;b[0][0,1]=1
        links=np.ones((3,3),bool);np.fill_diagonal(links,False)
        for tick in range(30):
            active=links if tick not in (4,5) else np.zeros((3,3),bool)
            before=new.last_receive_at.copy()
            a=old.service(beliefs[0],np.zeros((3,2)),float(tick),1.,active)
            b=new.service(beliefs[1],np.zeros((3,2)),float(tick),1.,active)
            assert np.array_equal(a,b) and all(np.array_equal(x,y) for x,y in zip(*beliefs))
            left=canonical(old.__dict__);right=canonical({k:v for k,v in new.__dict__.items() if k not in metadata})
            assert left==right,(bandwidth,tick)
            if tick==0 and bandwidth==10.:assert not np.isfinite(new.last_receive_at).any()
            if tick in (4,5):assert np.array_equal(before,new.last_receive_at)
            assert np.all(new.last_novel_receive_free<=new.last_novel_receive_cells)
        if bandwidth==100.:
            assert np.isfinite(new.last_novel_receive_at).any()
            assert new.last_novel_receive_free.sum()>0
            assert new.last_novel_receive_cells.sum()>new.last_novel_receive_free.sum()
        events+=30
    return events


class Own:
    def __init__(self,robot,value):self.robot,self.value=robot,value
    def __getitem__(self,robot):assert robot==self.robot,'Peer receiver state leak';return self.value


class OwnPairs(dict):
    def __init__(self,robot,*args):self.robot=robot;super().__init__(*args)
    def __getitem__(self,pair):assert pair[0]==self.robot,'Peer sender state leak';return super().__getitem__(pair)
    def get(self,pair,default=None):assert pair[0]==self.robot,'Peer pose state leak';return super().get(pair,default)


def locality():
    cfg=ContactConfig(method='finite_fifo');robot=1;pairs=[(i,j) for i in range(3) for j in range(3) if i!=j]
    own=[p for p in pairs if p[0]==robot]
    radio=SimpleNamespace(pairs=pairs,active=OwnPairs(robot,{p:None for p in own}),
        queues=OwnPairs(robot,{p:[] for p in own}),controls=OwnPairs(robot,{p:deque() for p in own}),
        peer_pose_time=OwnPairs(robot,{p:1. for p in own}),last_receive_at=Own(robot,-np.inf),
        last_novel_receive_at=Own(robot,-np.inf),last_novel_receive_cells=Own(robot,0.),last_novel_receive_free=Own(robot,0.))
    # No GT, peer maps, positions or link arrays exist on this fixture.
    env=SimpleNamespace(all_robot_map_belief_area_diff=Own(robot,np.array([100.,0.,200.])))
    contact=SimpleNamespace(config=cfg,radio=radio,time=10.)
    stale=local_state(contact,env,robot)
    radio.last_receive_at=Own(robot,10.);radio.last_novel_receive_at=Own(robot,10.)
    radio.last_novel_receive_cells=Own(robot,100.);radio.last_novel_receive_free=Own(robot,60.)
    fresh=local_state(contact,env,robot)
    assert stale[2]==stale[3]==stale[4]==0 and fresh[2]==1 and fresh[3]>0 and fresh[4]>fresh[3]
    contact.time=14.;aged=local_state(contact,env,robot);assert np.isclose(aged[2],np.exp(-1))
    assert np.isfinite(fresh).all() and np.all((fresh>=0)&(fresh<=4))
    return fresh


def networks(values):
    old=legacy('model');result=[]
    for ctor,previous in ((PolicyNet,old.PolicyNet),(QNet,old.QNet)):
        torch.manual_seed(101);a=previous(6,128,allow_wait=True);old_rng=torch.get_rng_state().clone()
        torch.manual_seed(101);b=ctor(6,128,allow_wait=True,contact_state_dim=6)
        assert torch.equal(old_rng,torch.get_rng_state())
        assert list(a.state_dict())==list(b.state_dict())[:-1]
        assert list(b.named_parameters())[-1][0]=='communication_embedding.weight'
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert torch.count_nonzero(b.communication_embedding.weight)==0
        nodes=torch.randn(1,5,6);mask=torch.zeros(1,1,5,dtype=torch.int64)
        obs=(nodes,torch.tensor([[[2,0,1,3,4]]]),torch.tensor([[[2]]]),torch.zeros(1,5),mask,torch.zeros(1,5,5))
        extended=attach_state(obs,values)
        assert torch.equal(extended[0][...,:6],nodes)
        assert torch.count_nonzero(extended[0][0,[0,1,3,4],6:])==0
        assert all(x is y for x,y in zip(obs[1:],extended[1:]))
        a.eval();b.eval()
        first=a(*[x.clone() for x in obs]);second=b(*[x.clone() for x in extended])
        left=first[0] if isinstance(first,tuple) else first
        right=second[0] if isinstance(second,tuple) else second
        assert torch.equal(left,right)
        assert torch.equal(a.encode_graph(nodes,obs[3],obs[5]),b.encode_graph(nodes,obs[3],obs[5]))
        try:b(*obs)
        except ValueError:pass
        else:raise AssertionError('Missing communication state was silently accepted')
        (right.square().sum() if isinstance(second,tuple) else -right[:,1].sum()).backward()
        gradient=b.communication_embedding.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)>0
        with torch.no_grad():b.communication_embedding.weight.fill_(.03)
        assert torch.equal(a.encode_graph(nodes,obs[3],obs[5]),b.encode_graph(nodes,obs[3],obs[5]))
        result.append(dict(model=ctor.__name__,zero_init_output_bit_exact=True,original_weights_parameter_order_rng_exact=True,
            graph_only_context_bit_exact=True,new_projection_gradients_finite_nonzero=True,missing_state_rejected=True))
    return result


def main():
    torch.set_num_threads(1);events=protocol();features=locality();proof=networks(features)
    report=dict(passed=True,protocol_service_ticks=events,original_wire_queue_ack_pose_credit_and_belief_state_exact=True,
        paid_receipts_only_after_complete_packet=True,receiver_free_vs_occupied_novelty_distinguished=True,
        own_information_only_fixture=True,current_node_only_packing=True,features=features.tolist(),models=proof,
        scope='CPU engineering prototype; no formal optimization or learning/99% claim')
    write_json('baseline_checks/communication_state_verified.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
