"""Read-only identity audit of the prepared, explicit v3->v4 continuation."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from baseline_support import write_json


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(1)
    parent=torch.load(ROOT/'experiments/contact_pose_audit_20261019/archive/checkpoint_77.pth',map_location='cpu')
    child=torch.load(ROOT/'experiments/contact_pose_audit_20261019/archive/prepared_v4_verified.pth',map_location='cpu')
    keys=['policy_model','q_net1_model','q_net2_model','log_alpha','policy_optimizer',
          'q_net1_optimizer','q_net2_optimizer','log_alpha_optimizer','policy_lr_decay',
          'q_net1_lr_decay','q_net2_lr_decay','log_alpha_lr_decay']
    assert all(equal(parent[k],child[k]) for k in keys)
    for k in ['head','encoder','optimizer','rng','updates','ready']:
        assert equal(parent['contact_learning'][k],child['contact_learning'][k]),k
    assert child['completed_episodes']==0 and child['episode']==0
    assert child['training_updates']==parent['training_updates']==552
    assert child['contact_config']['schema_version']==4
    assert child['contact_learning']['replay']==[]
    assert child['baseline_progress']['acceptance_recent']==[]
    assert all(v==0 for v in child['baseline_progress']['visits'].values())
    assert child['initialization']['sha256']==child['checkpoint_upgrade']['source_sha256']
    result=dict(passed=True,weights_temperature_optimizers_and_schedulers_exact=True,
                head_context_optimizer_rng_exact=True,inherited_sac_updates=552,
                inherited_head_updates=child['contact_learning']['updates'],
                prior_exposure_recorded=True,new_condition_acceptance_empty=True,
                stale_auxiliary_replay_cleared=True,scope='Prepared checkpoint audit, no optimization')
    write_json(ROOT/'baseline_checks/contact_pose_upgrade_verified.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
