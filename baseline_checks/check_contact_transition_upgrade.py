"""Exact audit of the explicit v5->v6 continuation, without optimization."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from baseline_support import write_json
from contact_module.config import ContactConfig


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(1)
    before=torch.load(ROOT/'experiments/contact_transition_audit_20261019/archive/checkpoint_v5_527.pth',map_location='cpu')
    after=torch.load(ROOT/'model/contact_easy10_v6_20261019/checkpoint.pth',map_location='cpu')
    keys=['policy_model','q_net1_model','q_net2_model','log_alpha','policy_optimizer','q_net1_optimizer',
        'q_net2_optimizer','log_alpha_optimizer','policy_lr_decay','q_net1_lr_decay','q_net2_lr_decay',
        'log_alpha_lr_decay','contact_learning','training_updates']
    assert all(equal(before[k],after[k]) for k in keys)
    assert after['training_updates']==7880 and after['contact_learning']['updates']==5036
    assert after['episode']==after['completed_episodes']==after['invalid_episodes']==0
    assert after['baseline_progress']['acceptance_recent']==[]
    assert len(after['baseline_progress']['visits'])==10 and not any(after['baseline_progress']['visits'].values())
    cfg=ContactConfig(method='finite_learned');cfg.validate_checkpoint(after)
    try:cfg.validate_checkpoint(before)
    except ValueError:pass
    else:raise AssertionError('Old experience contract silently resumed')
    result=dict(passed=True,all_weights_temperature_optimizers_schedulers_exact=True,
        entire_auxiliary_state_replay_rng_exact=True,inherited_sac_updates=7880,inherited_head_updates=5036,
        new_condition_acceptance_empty=True,old_contract_resume_rejected=True,automatic_review=after['automatic_review_control'])
    write_json(ROOT/'baseline_checks/contact_transition_upgrade_verified.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
