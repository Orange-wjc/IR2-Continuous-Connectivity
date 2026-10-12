import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from finite_contact.config import Config
from finite_contact.learning import Learner
from finite_contact_train import checkpoint, rollout
from finite_contact_multimap_joint import common_state, identical, train_group, archive_after_checkpoint
from finite_contact_multimap_content import weight_change
from finite_contact_multimap_report import rows


class MatchedAdaptationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self, directory):
        cfg = Config(map_family='simple_rooms_v1', sensor_implementation='vectorized',
                     train_maps=2, validation_maps=20, test_maps=20, seed=1,
                     variant='A', batch_size=2, warmup=2, horizon_seconds=4.,
                     success_memory_mode='per_map', success_episode_limit=2,
                     success_imitation_weight=1.)
        torch.manual_seed(1)
        learner = Learner(cfg)
        rollout(learner, cfg, 'train', 0, 'movement')
        self.assertGreater(learner.updates, 0)
        checkpoint(directory / 'source.pth', learner, cfg, 'movement', 1, None, 0.)
        movement = torch.load(directory / 'source.pth', map_location='cpu')
        batch = [(np.zeros(12, np.float32), np.zeros(64, np.float32), np.float32(.2))] * 2
        learner.update_content(batch)
        checkpoint(directory / 'head.pth', learner, cfg, 'content', 1, None, 0., full=False)
        return movement, torch.load(directory / 'head.pth', map_location='cpu')

    def test_collect_then_update_finishes_map_instead_of_budget_interrupt(self):
        cfg = Config(horizon_seconds=4., batch_size=2, warmup=2)
        learner = Learner(cfg)
        result, _ = rollout(learner, cfg, 'train', 0, 'finetune', episode_update_limit=0)
        self.assertFalse(result['interrupted'])
        self.assertEqual(result['time'], 4.)
        self.assertEqual(learner.updates, 0)
        self.assertGreaterEqual(len(learner.replay), 2)
        self.assertTrue(learner.update())
        self.assertTrue(learner.update())
        self.assertEqual(learner.updates, 2)
        before = learner.updates
        result, _ = rollout(learner, cfg, 'train', 0, 'finetune', episode_update_limit=1)
        self.assertFalse(result['interrupted'])
        self.assertEqual(result['time'], 4.)
        self.assertEqual(learner.updates, before + 1)

    def test_common_start_preserves_real_adam_and_transfers_head_without_fifo_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            movement, head = self.source(Path(temporary))
            before = copy.deepcopy(movement)
            saved = common_state(movement, head, 1e-5, 1)
            self.assertNotIn('replay', saved)
            self.assertNotIn('success_episodes', saved)
            self.assertTrue(movement['learner']['policy_optimizer']['state'])
            self.assertTrue(identical(saved['learner']['policy_optimizer']['state'],
                                      movement['learner']['policy_optimizer']['state']))
            self.assertEqual(saved['learner']['policy_optimizer']['param_groups'][0]['lr'], 1e-5)
            for name in ('q_optimizers', 'alpha_optimizer'):
                self.assertTrue(identical(saved['learner'][name], movement['learner'][name]))
            self.assertEqual(weight_change(saved['learner'], head['learner'], 'content'), 0)
            self.assertGreater(weight_change(saved['learner'], movement['learner'], 'content'), 0)
            self.assertTrue(identical(movement['learner'], before['learner']))
            head['learner']['policy'][next(iter(head['learner']['policy']))].add_(1.)
            with self.assertRaises(ValueError):
                common_state(movement, head, 1e-5, 1)

    def test_real_worker_completes_matched_maps_freezes_head_and_resumes_final_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            movement, head = self.source(root)
            common = common_state(movement, head, 1e-5, 1)
            torch.save(common, root / 'common_start.pth')
            plan = {'sweeps': 1, 'updates_per_map': 2, 'max_hours_per_group': .1,
                    'source_sac_updates': common['learner']['updates'],
                    'new_sac_updates_per_group': 4}
            train_group(root, plan, 'AB_learned', 'cpu')
            directory = root / 'AB_learned'
            latest = torch.load(directory / 'latest.pth', map_location='cpu')
            self.assertEqual(latest['episode'], 2)
            self.assertEqual(latest['learner']['updates'], common['learner']['updates'] + 4)
            records = rows(directory / 'training.jsonl')
            self.assertEqual({r['map']['index'] for r in records}, {0, 1})
            self.assertTrue(all(r['post_episode_sac_updates'] == 2 and not r['interrupted'] for r in records))
            for name in ('content', 'context_encoder'):
                self.assertEqual(weight_change(common['learner'], latest['learner'], name), 0)
            self.assertGreater(weight_change(common['learner'], latest['learner'], 'policy'), 0)
            original = rows(directory / 'validation_final/evaluation.jsonl')
            self.assertEqual(len(original), 20)
            train_group(root, plan, 'AB_learned', 'cpu', resume=directory / 'latest.pth')
            self.assertEqual(rows(directory / 'validation_final/evaluation.jsonl'), original)
            self.assertEqual(len(rows(directory / 'updates.jsonl')), 4)
            self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'complete')

    def test_resume_preserves_superseded_logs_and_removes_uncommitted_update_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'training.jsonl').write_text('{"episode": 10}\n{"episode": 11}\n')
            (directory / 'updates.jsonl').write_text('{"updates": 100}\n{"updates": 101}\n')
            archive_after_checkpoint(directory, 10, 100)
            self.assertEqual(rows(directory / 'training.jsonl'), [{'episode': 10}])
            self.assertEqual(rows(directory / 'updates.jsonl'), [{'updates': 100}])
            archive = next(directory.glob('superseded_*'))
            self.assertEqual(len(rows(archive / 'updates.jsonl')), 2)


if __name__ == '__main__':
    unittest.main()
