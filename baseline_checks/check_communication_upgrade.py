"""Audit old tensors/moments and new Adam IDs on a CPU copy of real v6 state."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from model import PolicyNet,QNet
from contact_module.config import ContactConfig
from contact_module.state import attach_state,zero_projection_import
from contact_module.state_upgrade import expand_checkpoint,PROJECTION
from baseline_support import write_json


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args()
    torch.set_num_threads(1);path=Path(args.checkpoint);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    before=torch.load(path,map_location='cpu');cfg=ContactConfig(method='finite_learned')
    maps=list(before['baseline_progress']['visits']);audit=dict(source=str(path.resolve()),source_sha256=sha)
    after=expand_checkpoint(before,cfg,maps,audit)
    for model_key,opt_key,ctor in (('policy_model','policy_optimizer',PolicyNet),
            ('q_net1_model','q_net1_optimizer',QNet),('q_net2_model','q_net2_optimizer',QNet)):
        assert all(equal(v,after[model_key][k]) for k,v in before[model_key].items())
        assert set(after[model_key])-set(before[model_key])=={PROJECTION}
        assert torch.count_nonzero(after[model_key][PROJECTION])==0
        old,new=before[opt_key],after[opt_key]
        assert equal(old['state'],new['state'])
        assert old['param_groups'][0]['params']==new['param_groups'][0]['params'][:-1]
        assert len(new['param_groups'][0]['params'])==len(old['param_groups'][0]['params'])+1
        for k,v in old['param_groups'][0].items():
            if k!='params':assert equal(v,new['param_groups'][0][k])
        model=ctor(6,128,allow_wait=True,contact_state_dim=6);model.load_state_dict(after[model_key])
        assert list(model.named_parameters())[-1][0]==PROJECTION
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-5);optimizer.load_state_dict(copy.deepcopy(new))
        assert len(optimizer.param_groups[0]['params'])==len(new['param_groups'][0]['params'])
        if ctor is PolicyNet:
            nodes=torch.randn(1,5,6);obs=(nodes,torch.tensor([[[2,0,1,3,4]]]),torch.tensor([[[2]]]),
                torch.zeros(1,5),torch.zeros(1,1,5,dtype=torch.int64),torch.zeros(1,5,5))
            extended=attach_state(obs,np.array([1.,1.,1.,.5,.7,2.],dtype=np.float32))
            loss=-model(*extended)[:,1].sum();optimizer.zero_grad();loss.backward();optimizer.step()
            parameter=model.communication_embedding.weight
            assert torch.count_nonzero(parameter)>0 and optimizer.state[parameter]['step'].item()==1
            for bad in (dict(after[model_key]),dict(before[model_key])):
                bad.pop(PROJECTION,None)
                try:model.load_state_dict(bad)
                except RuntimeError:pass
                else:raise AssertionError('Ordinary model load silently filled missing state')
    for key in ('log_alpha','log_alpha_optimizer','policy_lr_decay','q_net1_lr_decay','q_net2_lr_decay','log_alpha_lr_decay','training_updates'):
        assert equal(before[key],after[key]),key
    for key,value in before['contact_learning'].items():
        if key!='encoder':assert equal(value,after['contact_learning'][key]),key
    assert all(equal(v,after['contact_learning']['encoder'][k]) for k,v in before['contact_learning']['encoder'].items())
    assert after['completed_episodes']==after['episode']==after['invalid_episodes']==0
    assert after['baseline_progress']['acceptance_recent']==[] and not any(after['baseline_progress']['visits'].values())
    cfg.validate_checkpoint(after)
    for bad in (before,dict(after,policy_model=before['policy_model'])):
        try:cfg.validate_checkpoint(bad)
        except ValueError:pass
        else:raise AssertionError('Old/missing state contract accepted')
    assert sha==hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(passed=True,source_sha256=sha,inherited_sac_updates=after['training_updates'],
        inherited_head_updates=after['contact_learning']['updates'],old_weights_moments_schedulers_temperature_exact=True,
        old_auxiliary_context_head_replay_rng_exact=True,new_projection_zero_with_fresh_adam_id=True,
        cpu_copy_step_updates_projection=True,new_condition_acceptance_empty=True,missing_projection_or_old_schema_rejected=True,
        scope='CPU copy engineering only; one local test optimizer step, no formal training/model write')
    write_json('baseline_checks/communication_upgrade_verified.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
