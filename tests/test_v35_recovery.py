import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules['TRAINING'] = True

import env as env_module
from env import Env
from model import (PolicyNet, expand_policy_input_state_dict,
                   require_checkpoint_metadata)


class V35RecoveryTests(unittest.TestCase):
    def test_policy_expansion_preserves_outputs(self):
        torch.manual_seed(7)
        source = PolicyNet(11, 128).eval()
        target = PolicyNet(16, 128).eval()
        expanded = expand_policy_input_state_dict(
            source.state_dict(), source_input_dim=11, target_input_dim=16)
        target.load_state_dict(expanded)

        node_inputs = torch.randn(1, 12, 16)
        edge_inputs = torch.tensor([[[3, 1, 5, 7, 9]]])
        current_index = torch.tensor([[[3]]])
        edge_padding_mask = torch.zeros((1, 1, 5), dtype=torch.long)
        edge_mask = torch.zeros((1, 12, 12))

        with torch.no_grad():
            source_output = source(
                node_inputs[:, :, :11], edge_inputs, current_index,
                None, edge_padding_mask, edge_mask)
            target_output = target(
                node_inputs, edge_inputs, current_index,
                None, edge_padding_mask, edge_mask)

        self.assertEqual(
            float(target.initial_embedding.weight[:, 11:].abs().max()), 0.0)
        self.assertTrue(torch.equal(source_output, target_output))

    def test_checkpoint_metadata_validation_reports_mismatch(self):
        checkpoint = {'input_dim': 16, 'observation_version': 'wrong'}
        with self.assertRaisesRegex(ValueError, 'observation_version'):
            require_checkpoint_metadata(
                checkpoint,
                {'input_dim': 16, 'observation_version': 'local_recovery_v1'},
                'test checkpoint')

    def test_disconnected_rssi_trend_keeps_updating_from_local_estimate(self):
        environment = self._make_feature_environment()
        original_feature_dim = env_module.CONNECTIVITY_FEATURE_DIM
        env_module.CONNECTIVITY_FEATURE_DIM = 10
        try:
            environment.update_local_connectivity_state()
            self.assertEqual(float(environment.rssi_margin_trend[0]), 0.0)

            environment.all_robot_positions_belief[0][0] = np.array([5.0, 0.0])
            environment.update_local_connectivity_state()
            self.assertAlmostEqual(float(environment.rssi_margin_trend[0]), 0.5)

            environment.all_robot_positions_belief[0][0] = np.array([8.0, 0.0])
            environment.update_local_connectivity_state()
            self.assertAlmostEqual(float(environment.rssi_margin_trend[0]), 0.3)
        finally:
            env_module.CONNECTIVITY_FEATURE_DIM = original_feature_dim

    def test_ten_features_preserve_v34_prefix(self):
        environment = self._make_feature_environment()
        node_coords = np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]])
        original_feature_dim = env_module.CONNECTIVITY_FEATURE_DIM
        try:
            env_module.CONNECTIVITY_FEATURE_DIM = 5
            legacy_features = environment.get_connectivity_node_features(0, node_coords)
            env_module.CONNECTIVITY_FEATURE_DIM = 10
            recovery_features = environment.get_connectivity_node_features(0, node_coords)
        finally:
            env_module.CONNECTIVITY_FEATURE_DIM = original_feature_dim

        self.assertEqual(legacy_features.shape, (3, 5))
        self.assertEqual(recovery_features.shape, (3, 10))
        np.testing.assert_allclose(legacy_features, recovery_features[:, :5])
        self.assertTrue(np.isfinite(recovery_features).all())

    @staticmethod
    def _make_feature_environment():
        environment = Env.__new__(Env)
        environment.n_agent = 2
        environment.group_ids_list = [[0], [1]]
        environment.local_reachability_history = [[], []]
        environment.local_recent_reachability = np.ones(2, dtype=np.float32)
        environment.previous_best_local_margin = [None, None]
        environment.rssi_margin_trend = np.zeros(2, dtype=np.float32)
        environment.last_fully_connected_positions = [
            np.array([0.0, 0.0]), np.array([10.0, 0.0])]
        environment.all_robot_positions_gt = [
            np.array([0.0, 0.0]), np.array([10.0, 0.0])]
        environment.all_robot_positions_belief = [
            [np.array([0.0, 0.0]), np.array([10.0, 0.0])],
            [np.array([0.0, 0.0]), np.array([10.0, 0.0])],
        ]
        belief = np.full((32, 32), 255, dtype=np.uint8)
        environment.all_robot_belief = [
            [belief.copy(), belief.copy()],
            [belief.copy(), belief.copy()],
        ]
        environment.disconnect_steps = np.array([5, 0], dtype=np.int32)
        environment.all_robot_positions_step_updated = [[5, 0], [0, 5]]
        environment.estimate_link = lambda robot_belief, source, target: (
            -70.0 + float(source[0]), float(source[0]), float(source[0]) > 0)
        return environment


if __name__ == '__main__':
    unittest.main()
