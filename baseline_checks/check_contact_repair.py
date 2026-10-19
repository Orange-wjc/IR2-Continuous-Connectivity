"""Regression fixtures for the observed v1 migration defects."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['IR2_METHOD'] = 'finite_learned'
os.environ['IR2_RUN_NAME'] = 'contact_repair_validation'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-contact-validation-mpl')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
import torch
import parameter as config
from baseline_support import prepare_maps, write_json
from env import Env
from graph import Edge
from model import PolicyNet
from contact_module.adapter import ContactAdapter
from contact_module.radio import MAP, CELL
from contact_module.config import ContactConfig
from contact_module.learning import ContentTrainer
from contact_module.preparation import load_initialization


def main():
    torch.set_num_threads(1)
    manifest = prepare_maps(config)
    policy = PolicyNet(6, 128, allow_wait=True)
    source, provenance = load_initialization('model/2/checkpoint.pth', manifest)
    policy.load_state_dict(source['policy_model'])
    trainer = ContentTrainer(policy, config.CONTACT_CONFIG, provenance=provenance)
    assert all(torch.equal(v.cpu(), trainer.encoder[k]) for k, v in source['policy_model'].items())
    contact = ContactAdapter(config.CONTACT_CONFIG, policy, trainer.weights(), 4)
    env = Env(4, 3, contact=contact)
    # A one-cell packet exactly on the real Env.find_frontier output must be near.
    frontier = env.all_frontiers[0]
    assert len(frontier)
    point = frontier[0]
    data = np.empty(1, dtype=CELL)
    data['index'] = int(point[1])*env.ground_truth_size[1]+int(point[0])
    data['value'] = 255
    packet = contact.radio.packet(MAP, (0, 1), 0., data.tobytes(), 1)
    features, _ = contact.features((0, 1), [packet])
    assert features[0, 9] == 1.
    # Walls/detours/direction: straight-line distance would be 1, local path is 201.
    coords = np.array([[0., 0.], [0., 100.], [1., 0.], [2., 0.]])
    edges = {(0., 0.): {1: Edge((0., 100.), 100.)},
             (0., 100.): {2: Edge((1., 0.), 101.)}}
    distances = contact.path_distances(coords, edges, coords[2])
    assert np.array_equal(distances[:3], [201., 101., 0.]) and np.isinf(distances[3])
    reverse = contact.path_distances(coords, edges, coords[0])
    assert np.isinf(reverse[1:]).all()
    # Same gain produces the same signal at its goal, independent of map area.
    old_coords, old_graph, old_start = env.all_node_coords[0], env.all_graph[0], env.start_position
    env.all_node_coords[0] = coords; env.all_graph[0] = edges; env.start_position = coords[2]
    contact.expected_delivery = lambda pair: 500. if pair == (0, 1) else 0.
    signal = contact.utility(env, 0)[1][:, 0]
    assert signal[2] == 1. and signal[3] == 0. and 0. < signal[0] < signal[1] < signal[2]
    own = env.all_robot_positions_belief[0][0]
    env.all_robot_positions_belief[0][0] = coords[2].copy()
    masked = contact.utility(env,0)[1][:,0]
    assert masked[2] == 0. and masked[0] == 0.
    env.all_robot_positions_belief[0][0] = own
    env.all_node_coords[0], env.all_graph[0], env.start_position = old_coords, old_graph, old_start
    # Random head cannot control bootstrap; readiness/provenance survive resume.
    assert not trainer.weights()['ready'] and not contact.content_ready
    trainer.ready = True
    restored = ContentTrainer(policy, config.CONTACT_CONFIG, trainer.state_dict())
    assert restored.ready and restored.provenance == provenance
    current = dict(contact_config=config.CONTACT_CONFIG.to_dict(), contact_learning=trainer.state_dict())
    config.CONTACT_CONFIG.validate_checkpoint(current)
    current['contact_config'] = dict(current['contact_config'], schema_version=2)
    try:
        config.CONTACT_CONFIG.validate_checkpoint(current)
    except ValueError:
        pass
    else:
        raise AssertionError('v2 reward semantics accepted by v3 resume')
    ContactConfig().validate_checkpoint(source)
    report = dict(passed=True, frontier_point=point.tolist(), near_frontier_feature=float(features[0, 9]),
                  directed_path_distances=[201., 101., 0., 'unreachable'],
                  gain_500_signal=signal.tolist(), trained_context_sha256=provenance['sha256'],
                  old_finite_resume_rejected=True, original_checkpoint_still_accepted=True,
                  readiness_and_provenance_restored=True)
    write_json('baseline_checks/contact_repair_verified.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
