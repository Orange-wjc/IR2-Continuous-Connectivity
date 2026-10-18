import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

WORK=Path(__file__).resolve().parents[1]
REPO=WORK.parents[1]
sys.path.insert(0,str(WORK/'source'))
from native_driver import configuration,save_state,restore_state,restore_rng
from native_data import load_original,dataset
from native_environment import segment_clear
from native_learning import NativeLearner
from native_worker import cpu_weights
from finite_contact.environment import Environment
from finite_contact.learning import tensors
from finite_contact.memory import pack_observation,compact_state,semantic_digest
from env import Env

torch.set_num_threads(1)

class NativeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=configuration(WORK/'checks/native_dataset.json',preflight=True)
        cls.env=Environment(cls.cfg,'train',0)
        cls.observations=[cls.env.observation(i) for i in range(2)]

    def test_original_gray_and_rgba_decode_and_spawn_exact(self):
        for name in ('train/easy/img_1.png','train/easy/img_10.png','train/difficult/img_1.png','train/difficult/img_1449.png'):
            path=REPO/'DungeonMaps'/name
            expected,start=Env.import_ground_truth(None,str(path))
            actual,actual_start=load_original(path)
            np.testing.assert_array_equal(expected,actual)
            np.testing.assert_array_equal(start,actual_start)

    def test_full_original_counts_and_geometry_group_separation(self):
        d=dataset(str(WORK/'checks/native_dataset.json'))
        for name,expected in [('easy',9999),('difficult',6000)]:
            splits=d['families'][name]['splits']
            self.assertEqual(sum(len(x) for x in splits.values()),expected)
            a={x['geometry_group'] for x in splits['train']}
            b={x['geometry_group'] for x in splits['validation']}
            self.assertFalse(a&b)
            self.assertGreater(len(splits['train']),1000)
        with self.assertRaises(ValueError):Environment(self.cfg,'test',0)

    def test_sender_observation_does_not_read_true_peer_map(self):
        before=self.env.observation(0)
        saved=self.env.beliefs[1].copy()
        self.env.beliefs[1][:]=255
        after=self.env.observation(0)
        self.env.beliefs[1][:]=saved
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_continuous_actions_reject_wall_interiors(self):
        b=np.full((10,10),255,np.uint8);b[5,5]=1
        self.assertFalse(segment_clear(b,[1,1],[8,8]))
        self.assertTrue(segment_clear(b,[1,4.5],[8,4.5]))
        self.assertFalse(segment_clear(b,[1,4.51],[8,4.51]))
        self.assertTrue(segment_clear(b,[1,1],[1,8]))

    def test_vectorized_sensor_matches_upstream_on_native_map(self):
        from sensor import sensor_work
        from finite_contact.sensing import sensor_work_vectorized
        truth=self.env.truth
        for position in self.env.positions:
            a=np.full(truth.shape,127,np.uint8);b=a.copy()
            sensor_work(np.rint(position).astype(int),80,a,truth)
            sensor_work_vectorized(np.rint(position).astype(int),80,b,truth)
            np.testing.assert_array_equal(a,b)

    def test_dynamic_padding_matches_individual_forward(self):
        learner=NativeLearner(self.cfg,'cpu',microbatch=4)
        with torch.no_grad():
            batch=learner.policy(*tensors(self.observations,'cpu'))
            individual=torch.cat([learner.policy(*tensors([o],'cpu')) for o in self.observations])
        torch.testing.assert_close(batch,individual,rtol=1e-5,atol=2e-6)
        self.assertNotEqual(len(self.observations[0][0]),len(self.observations[1][0]))

    def learners(self):
        cfg=copy.deepcopy(self.cfg);cfg.batch_size=4;cfg.warmup=4
        torch.manual_seed(41)
        a=NativeLearner(cfg,'cpu',4)
        b=NativeLearner(cfg,'cpu',2);b.load(copy.deepcopy(a.state()),optimizers=True)
        obs=[pack_observation(o) for o in self.observations]
        transitions=[(obs[i%2],1,float(i/20),1.+i/10,obs[(i+1)%2],float(i==7),False) for i in range(8)]
        a.replay.extend(transitions);b.replay.extend(transitions)
        return a,b,cfg

    def test_microbatch_preserves_effective_batch_update(self):
        import random
        a,b,cfg=self.learners()
        random.seed(7);ra=a.update()
        random.seed(7);rb=b.update()
        self.assertEqual(ra['updates'],rb['updates'])
        for name in ('policy_loss','q1_loss','q2_loss','alpha_loss'):
            self.assertAlmostEqual(ra[name],rb[name],delta=1e-4)
        for x,y in zip(a.policy.parameters(),b.policy.parameters()):
            torch.testing.assert_close(x,y,rtol=1e-4,atol=2e-5)

    def test_full_checkpoint_restores_rng_adam_replay_and_next_update(self):
        a,_,cfg=self.learners();a.update()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'latest.pth'
            save_state(path,a,cfg,1,'test')
            before=a.state()['policy']['initial_embedding.weight'].clone()
            a.update();expected=semantic_digest(a.state())
            b,_,episode,_=restore_state(path,'cpu',4)
            self.assertEqual(episode,1)
            self.assertEqual(len(b.replay),8)
            b.update()
            self.assertEqual(semantic_digest(b.state()),expected)
            self.assertFalse(torch.equal(before,b.policy.initial_embedding.weight))

    def test_hard_target_copy_at_original_64_update_boundary(self):
        a,_,_=self.learners();a.updates=63;a.update()
        self.assertEqual(a.updates,64)
        for q,target in zip(a.critics,a.targets):
            for p,t in zip(q.parameters(),target.parameters()):self.assertTrue(torch.equal(p,t))

if __name__=='__main__':unittest.main(verbosity=2)
