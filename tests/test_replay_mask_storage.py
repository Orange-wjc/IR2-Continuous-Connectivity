"""Lossless replay storage checks; no training or evaluation episodes are run."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import PolicyNet, QNet
from robot import Robot, pack_graph_mask, stack_graph_masks


class ReplayMaskStorageTests(unittest.TestCase):
    def test_snapshot_roundtrip_and_storage(self):
        mask = torch.randint(2, (1, 360, 360)).float()
        original = mask.clone()
        observations = (torch.zeros(1, 360, 16), torch.zeros(1, 1, 30).long(),
                        torch.zeros(1, 1, 1).long(), torch.zeros(1, 1, 360).long(),
                        torch.zeros(1, 1, 30).long(), mask)
        robot = Robot(0, None)
        robot.save_observations(observations)
        robot.save_next_observations(observations)
        mask.fill_(1)
        for field in (5, 14):
            stored = robot.episode_buffer[field][0]
            self.assertEqual(stored.dtype, torch.bool)
            self.assertEqual(stored.numel() * stored.element_size(),
                             original.numel() * original.element_size() // 4)
            self.assertTrue(torch.equal(stack_graph_masks([stored], 'cpu'), original))
        robot.episode_buffer[5][0].logical_not_()
        self.assertTrue(torch.equal(
            stack_graph_masks(robot.episode_buffer[14], 'cpu'), original))

    def test_reject_nonbinary_values(self):
        for value in (0.5, -1.0, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pack_graph_mask(torch.tensor([value]))

    def test_model_outputs_losses_and_gradients_are_unchanged(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for cls in (PolicyNet, QNet):
                with self.subTest(device=device, network=cls.__name__):
                    torch.manual_seed(13)
                    network = cls(16, 128).to(device)
                    nodes = torch.randn(2, 12, 16, device=device)
                    edges = torch.tensor([[[3, 1, 5, 7, 3]],
                                          [[3, 1, 5, 7, 3]]], device=device)
                    current = torch.full((2, 1, 1), 3, dtype=torch.long, device=device)
                    padding = torch.tensor([[[0, 0, 0, 0, 1]],
                                            [[0, 0, 0, 0, 1]]], device=device)
                    mask = torch.randint(2, (2, 12, 12)).float()
                    mask[:, torch.arange(12), torch.arange(12)] = 0
                    legacy = stack_graph_masks(list(mask), device)
                    restored = stack_graph_masks(list(pack_graph_mask(mask)), device)
                    self.assertTrue(torch.equal(legacy, restored))
                    results = []
                    for model_mask in (legacy, restored):
                        network.zero_grad(set_to_none=True)
                        output = network(nodes, edges, current, None, padding, model_mask)
                        if cls is PolicyNet:
                            loss = (output.exp() * output).sum(-1).mean()
                        else:
                            output = output[0]
                            loss = (output[:, :4] - 0.25).square().mean()
                        loss.backward()
                        gradients = {n: p.grad.detach().clone()
                                     for n, p in network.named_parameters()
                                     if p.grad is not None}
                        results.append((output.detach().clone(), loss.detach().clone(), gradients))
                    self.assertTrue(torch.equal(results[0][0], results[1][0]))
                    self.assertTrue(torch.equal(results[0][1], results[1][1]))
                    self.assertEqual(results[0][2].keys(), results[1][2].keys())
                    for name in results[0][2]:
                        self.assertTrue(torch.equal(results[0][2][name], results[1][2][name]), name)


if __name__ == '__main__':
    unittest.main()
