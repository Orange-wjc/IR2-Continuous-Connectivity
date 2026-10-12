"""Uniform history retention, equal map weighting and genuine frozen-head updates."""
import unittest
import pickle
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.learning import Learner
from finite_contact_multimap_content import Reservoir, balanced_batch, weight_change


class BalancedContentChecks(unittest.TestCase):
    def test_zero_constant_window_control_is_fifo_and_does_not_mutate_state(self):
        from finite_contact.environment import Environment
        from finite_contact_content_check import state_check
        torch.set_num_threads(2)
        cfg = Config(variant='A', sensor_implementation='vectorized')
        learner = Learner(cfg)
        env = Environment(cfg, 'train', 0, scorer=learner.score)
        radio_before = pickle.dumps(env.radio)
        beliefs_before = [b.copy() for b in env.beliefs]
        result = state_check(env, learner, 0., constant_control=True)
        self.assertEqual(result['forced_contact_gains']['bounded_constant'],
                         result['forced_contact_gains']['fifo'])
        self.assertEqual(radio_before, pickle.dumps(env.radio))
        for before, after in zip(beliefs_before, env.beliefs):
            np.testing.assert_array_equal(before, after)

    def test_reservoir_keeps_early_and_late_samples_without_touching_global_rng(self):
        np.random.seed(77)
        before = np.random.get_state()
        sample = [(np.array([i]), np.array([i + 1]), np.float32(i)) for i in range(10000)]
        whole, chunks = Reservoir(1000, 5), Reservoir(1000, 5)
        whole.add(sample)
        for offset in range(0, len(sample), 100):
            chunks.add(sample[offset:offset + 100])
        for a, b in zip(whole.data, chunks.data):
            np.testing.assert_array_equal(a, b)
        self.assertGreater(np.sum(chunks.data[2] < 1000), 60)
        self.assertGreater(np.sum(chunks.data[2] >= 9000), 60)
        self.assertEqual(chunks.seen, 10000)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_small_and_large_map_datasets_have_equal_sampling_probability(self):
        datasets = [(np.full((n, 12), i, np.float32), np.zeros((n, 64), np.float32),
                     np.full(n, i, np.float32)) for i, n in enumerate(range(1, 101))]
        batch, indices = balanced_batch(datasets, np.random.RandomState(1), 80000)
        counts = np.bincount(indices, minlength=100)
        self.assertGreater(counts.min(), 650)
        self.assertLess(counts.max(), 950)
        self.assertTrue(all(item[2] == i for item, i in zip(batch, indices)))

    def test_explicit_head_batch_changes_only_head_and_reduces_supervised_error(self):
        torch.set_num_threads(2)
        torch.manual_seed(33)
        cfg = Config(batch_size=16)
        learner = Learner(cfg)
        before = {k: ({name: value.clone() for name, value in v.items()} if isinstance(v, dict)
                       else [{name: value.clone() for name, value in part.items()} for part in v])
                  for k, v in learner.state().items()
                  if k in ('policy', 'critics', 'targets', 'context_encoder', 'content')}
        batch = [(np.zeros(12, np.float32), np.zeros(64, np.float32), np.float32(0.2))] * 16
        first = learner.update_content(batch)['content_loss']
        for _ in range(100):
            last = learner.update_content(batch)['content_loss']
        self.assertLess(last, first)
        self.assertEqual(learner.updates, 0)
        for name in ('policy', 'critics', 'targets', 'context_encoder'):
            self.assertEqual(weight_change(before, learner.state(), name), 0)
        self.assertGreater(weight_change(before, learner.state(), 'content'), 0)
        with self.assertRaises(ValueError):
            learner.update_content(batch[:2])


if __name__ == '__main__':
    unittest.main()
