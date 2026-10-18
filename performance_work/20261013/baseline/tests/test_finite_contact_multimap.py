"""Map isolation, diversity, scheduling and real multi-map training checks."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from scipy.ndimage import label
from finite_contact.config import Config
from finite_contact.maps import simple_map, configured_map, manifest, training_map_index
from finite_contact.environment import Environment
from finite_contact.learning import Learner


class MultiMapChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **changes):
        return Config(**dict(dict(map_family='simple_rooms_v1', train_maps=100,
            validation_maps=20, test_maps=20, map_order='shuffled'), **changes))

    def test_140_maps_are_connected_distinct_and_reproducible(self):
        cfg = self.config()
        entries = manifest(cfg)
        self.assertEqual(len(entries), 140)
        self.assertEqual(len({e['sha256'] for e in entries}), 140)
        self.assertEqual(len({e['canonical_sha256'] for e in entries}), 140)
        self.assertEqual({e['rooms'] for e in entries}, {2, 3})
        self.assertEqual({e['rotation'] for e in entries}, {0, 1, 2, 3})
        for info in entries:
            truth, starts, again = configured_map(cfg, info['split'], info['index'])
            self.assertEqual(info, again)
            self.assertEqual(label(truth == 255)[1], 1)
            self.assertTrue(np.all(truth[starts[:, 1].astype(int), starts[:, 0].astype(int)] == 255))
        cfg.seed = 99  # Changing a training seed must not silently change the dataset.
        self.assertEqual(entries, manifest(cfg))

    def test_generator_and_map_schedule_leave_global_rng_unchanged(self):
        cfg = self.config()
        random.seed(47); np.random.seed(48); torch.manual_seed(49)
        py, n, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        configured_map(cfg, 'train', 99)
        training_map_index(cfg, 132)
        self.assertEqual(py, random.getstate())
        after = np.random.get_state()
        self.assertEqual(n[0], after[0]); np.testing.assert_array_equal(n[1], after[1])
        self.assertEqual(n[2:], after[2:]); self.assertTrue(torch.equal(th, torch.get_rng_state()))

    def test_each_shuffled_sweep_visits_all_maps_and_resumes_by_episode(self):
        cfg = self.config()
        first = [training_map_index(cfg, e) for e in range(1, 101)]
        second = [training_map_index(cfg, e) for e in range(101, 201)]
        self.assertEqual(sorted(first), list(range(100)))
        self.assertEqual(sorted(second), list(range(100)))
        self.assertNotEqual(first, second)
        restored = Config.from_checkpoint(cfg.to_dict())
        self.assertEqual(second[31], training_map_index(restored, 132))
        cfg.map_order = 'cyclic'
        self.assertEqual([training_map_index(cfg, e) for e in range(1, 101)], list(range(100)))

    def test_old_checkpoints_keep_exact_legacy_maps_and_limits(self):
        cfg = Config.from_checkpoint({'train_maps': 10})
        self.assertEqual(cfg.map_family, 'legacy')
        for info in manifest(cfg):
            actual = configured_map(cfg, info['split'], info['index'])
            expected = simple_map(info['split'], info['index'])
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertEqual(actual[2], expected[2])
        with self.assertRaises(ValueError):
            Config(train_maps=100).validate()
        with self.assertRaises(ValueError):
            configured_map(self.config(), 'validation', 20)

    def test_new_layouts_build_legal_local_observations(self):
        cfg = self.config(variant='A')
        for split, index in [('train', 0), ('train', 99), ('validation', 19), ('test', 19)]:
            env = Environment(cfg, split, index)
            self.assertLess(env.coverage()[1], .85)
            for robot in range(2):
                obs = env.observation(robot)
                self.assertTrue(all(np.isfinite(x).all() for x in obs))
                self.assertGreater(np.sum(obs[4] == 0), 1)

    def test_per_map_success_memory_retains_harder_layouts(self):
        cfg = self.config(success_memory_mode='per_map', success_episode_limit=100)
        learner = Learner(cfg)
        fake = [(None, 0, 0., 1., None, 1., False)] * 2
        learner.remember_success(fake, 10., 'easy')
        learner.remember_success(fake, 240., 'hard')
        learner.remember_success(fake, 15., 'easy')
        self.assertEqual([(e['map_key'],e['duration']) for e in learner.success_episodes],
                         [('easy',10.),('hard',240.)])
        learner.remember_success(fake, 200., 'hard')
        self.assertEqual(learner.success_episodes[1]['duration'], 200.)
        with self.assertRaises(ValueError):
            self.config(success_memory_mode='per_map', success_episode_limit=3).validate()

    def test_real_cli_multi_map_checkpoint_and_strict_resume(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            command = [sys.executable, str(root/'finite_contact_train.py'), '--stage','movement',
                '--map-family','simple_rooms_v1','--train-maps','100','--validation-maps','20',
                '--test-maps','20','--map-order','shuffled','--success-memory-mode','per_map',
                '--success-episode-limit','100','--horizon','4','--batch-size','2','--warmup','2',
                '--max-updates','2','--episodes','1','--device','cpu','--run-dir',str(run)]
            subprocess.run(command,cwd=root,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            saved = torch.load(run/'latest.pth',map_location='cpu')
            self.assertGreater(saved['learner']['updates'],0)
            self.assertEqual(len(json.loads((run/'maps.json').read_text())),140)
            subprocess.run([sys.executable,str(root/'finite_contact_train.py'),'--stage','movement',
                '--resume',str(run/'latest.pth'),'--episodes','2','--max-updates','4','--device','cpu'],
                cwd=root,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            after = torch.load(run/'latest.pth',map_location='cpu')
            self.assertEqual(saved['config'],after['config'])
            self.assertGreater(after['learner']['updates'],saved['learner']['updates'])

    def test_vectorized_sensor_exactly_matches_legacy_full_beliefs(self):
        from finite_contact.sensing import sensor_work_vectorized
        from sensor import sensor_work
        for index in range(8):
            truth,starts,_=configured_map(self.config(),'train',index)
            rng=np.random.RandomState(index)
            free=np.argwhere(truth==255)
            poses=list(starts.astype(int))+[p[::-1] for p in free[rng.choice(len(free),2,replace=False)]]
            for pose in poses:
                for radius in [8,35,70]:
                    belief=rng.choice(np.array([1,127,255],np.uint8),size=truth.shape)
                    expected=sensor_work(pose,radius,belief.copy(),truth)
                    actual=sensor_work_vectorized(pose,radius,belief.copy(),truth)
                    np.testing.assert_array_equal(actual,expected)

    def test_vectorized_sensor_preserves_real_rollout_and_protocol(self):
        from finite_contact_train import rollout
        cfg=self.config(variant='A',horizon_seconds=24)
        learner=Learner(cfg)
        before,env=rollout(learner,cfg,'train',4,'evaluate')
        cfg.sensor_implementation='vectorized'
        after,fast=rollout(learner,cfg,'train',4,'evaluate')
        self.assertEqual(before,after)
        self.assertEqual(env.trajectory,fast.trajectory)
        self.assertEqual(env.radio.events,fast.radio.events)
        np.testing.assert_array_equal(env.beliefs,fast.beliefs)


if __name__ == '__main__':
    unittest.main()
