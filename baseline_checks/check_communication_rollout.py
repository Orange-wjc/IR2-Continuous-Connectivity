"""Real-map CPU v6/new-zero-adapter paired sampling, no optimization."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['IR2_METHOD']='finite_learned';os.environ['IR2_RUN_NAME']='communication_rollout_validation'
os.environ['MPLCONFIGDIR']='/tmp/ir2-comm-state-check-mpl'
from pathlib import Path
import argparse,copy,hashlib,json,random,sys,types,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
import parameter as config
from baseline_support import prepare_maps,write_json
from contact_module.adapter import ContactAdapter
from contact_module.state import zero_projection_import
from model import PolicyNet,QNet
import multi_robot_worker as workers


def module(name):
    m=types.ModuleType('communication_legacy_'+name);sys.modules[m.__name__]=m
    exec(compile(subprocess.check_output(['git','show','1cfa3779:'+name+'.py'],text=True),name+'_v6','exec'),m.__dict__)
    return m


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--steps',type=int,default=12)
    args=parser.parse_args();torch.set_num_threads(1);prepare_maps(config)
    path=Path(args.checkpoint);sha=hashlib.sha256(path.read_bytes()).hexdigest();source=torch.load(path,map_location='cpu')
    assert source['contact_config']['schema_version']==6
    oldmodel,oldworker=module('model'),module('multi_robot_worker');results=[]
    for old in (True,False):
        random.seed(20261019);np.random.seed(20261019);torch.manual_seed(20261019)
        p=(oldmodel.PolicyNet(6,128,allow_wait=True) if old else PolicyNet(6,128,allow_wait=True,contact_state_dim=6))
        q=(oldmodel.QNet(6,128,allow_wait=True) if old else QNet(6,128,allow_wait=True,contact_state_dim=6))
        p.load_state_dict(source['policy_model'] if old else zero_projection_import(p,source['policy_model']))
        q.load_state_dict(source['q_net1_model'] if old else zero_projection_import(q,source['q_net1_model']))
        p.eval();q.eval();aux=copy.deepcopy(source['contact_learning'])
        if not old:aux['encoder']=zero_projection_import(p,aux['encoder'])
        contact=ContactAdapter(config.CONTACT_CONFIG,p,aux,4)
        code=oldworker if old else workers;code.MAX_EPS_STEPS=args.steps
        w=code.Worker(0,3,p,q,4,device='cpu',contact=contact)
        actions=[];previous=w.select_node
        def select(obs,robot):
            position,action=previous(obs,robot);actions.append((robot,position.tolist(),int(action.item())))
            return position,action
        w.select_node=select;assert w.run_episode(4)
        contexts={robot:(coords.copy(),embedding.copy()) for robot,(coords,embedding) in contact.contexts.items()}
        results.append((w,actions,contexts))
    before,after=results
    assert before[1]==after[1] and before[0].perf_metrics==after[0].perf_metrics
    for robot in before[2]:
        assert all(np.array_equal(a,b) for a,b in zip(before[2][robot],after[2][robot]))
    for old,new in zip(before[0].episode_buffer,after[0].episode_buffer):assert len(old)==len(new)
    for column,(old,new) in enumerate(zip(before[0].episode_buffer,after[0].episode_buffer)):
        for index,(a,b) in enumerate(zip(old,new)):
            if hasattr(a,'unpack'):a,b=a.unpack(),b.unpack()
            if column in (0,9):
                assert a.shape[-1]==6 and b.shape[-1]==12 and torch.equal(a,b[:,:6])
                current_column=2 if column==0 else 11
                current=int(after[0].episode_buffer[current_column][index].item())
                others=torch.ones(b.shape[0],dtype=torch.bool);others[current]=False
                assert torch.count_nonzero(b[others,6:])==0 and torch.isfinite(b).all()
            else:assert torch.equal(a,b),(column,index)
    for robot in after[0].robot_list:
        b=robot.episode_buffer
        for i in range(len(b[0])-1):
            for col in range(6):
                a,n=b[9+col][i],b[col][i+1]
                if hasattr(a,'unpack'):a,n=a.unpack(),n.unpack()
                assert torch.equal(a,n),(robot.robot_id,i,col)
    assert sha==hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(passed=True,steps=args.steps,robots=3,actual_actions=len(after[1]),
        zero_adapter_physical_actions_rewards_wire_and_metrics_exact=True,
        frozen_context_bit_exact=True,original_replay_columns_exact_except_added_features=True,
        actual_and_next_state12_fields_recorded=True,all_next_decision_states_exact=True,
        current_node_only_communication_state=True,source_sha256=sha,
        scope='Paired CPU engineering rollout, no optimization or formal success counting')
    write_json('baseline_checks/communication_rollout_verified.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
