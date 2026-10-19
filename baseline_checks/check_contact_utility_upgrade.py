"""Read-only identity audit of the explicit v4->v5 cache-repair continuation."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from baseline_support import write_json
from contact_module.config import ContactConfig


def equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    torch.set_num_threads(1)
    archive = ROOT/'experiments/contact_utility_audit_20261019/archive'
    parent = torch.load(archive/'checkpoint_v4_405.pth', map_location='cpu')
    child = torch.load(archive/'prepared_v5_verified.pth', map_location='cpu')
    keys = ['policy_model', 'q_net1_model', 'q_net2_model', 'log_alpha', 'policy_optimizer',
            'q_net1_optimizer', 'q_net2_optimizer', 'log_alpha_optimizer', 'policy_lr_decay',
            'q_net1_lr_decay', 'q_net2_lr_decay', 'log_alpha_lr_decay']
    assert all(equal(parent[k], child[k]) for k in keys)
    for key in ['head', 'encoder', 'optimizer', 'rng', 'updates', 'ready']:
        assert equal(parent['contact_learning'][key], child['contact_learning'][key]), key
    assert child['completed_episodes'] == child['episode'] == child['invalid_episodes'] == 0
    assert child['training_updates'] == parent['training_updates'] == 3728
    assert child['contact_config']['schema_version'] == 5
    assert child['contact_learning']['updates'] == 2928
    assert child['contact_learning']['replay'] == []
    assert child['baseline_progress']['acceptance_recent'] == []
    assert all(v == 0 for v in child['baseline_progress']['visits'].values())
    assert child['initialization']['sha256'] == child['checkpoint_upgrade']['source_sha256']
    assert child['automatic_review_control']
    cfg = ContactConfig(method='finite_learned')
    cfg.validate_checkpoint(child)
    try:
        cfg.validate_checkpoint(parent)
    except ValueError:
        pass
    else:
        raise AssertionError('An old cache contract silently resumed as v5')
    result = dict(passed=True, weights_temperature_optimizers_and_schedulers_exact=True,
                  head_context_optimizer_and_auxiliary_rng_exact=True, inherited_sac_updates=3728,
                  inherited_head_updates=2928, new_condition_acceptance_empty=True,
                  stale_auxiliary_replay_cleared=True, prior_exposure_recorded=True,
                  old_schema_resume_rejected=True,
                  scope='Prepared checkpoint audit, no optimization; sampler streams rebuild on driver restart')
    write_json(ROOT/'baseline_checks/contact_utility_upgrade_verified.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
