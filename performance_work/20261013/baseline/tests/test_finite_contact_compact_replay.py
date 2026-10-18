import copy
import random
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner, compact_observation, dense_observation, tensors
from finite_contact_train import checkpoint, rollout
from finite_contact_multimap_joint import identical
from finite_contact.maps import configured_map, training_map_index


class CompactReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_observation_roundtrip_and_mixed_batches_are_exact(self):
        cfg = Config()
        observation = Environment(cfg, 'train', 0).observation(0)
        packed = compact_observation(observation)
        self.assertLess(packed[-1].nbytes, observation[-1].nbytes / 7)
        for original, restored in zip(observation, dense_observation(packed)):
            np.testing.assert_array_equal(original, restored)
            self.assertEqual(original.dtype, restored.dtype)
        self.assertIs(compact_observation(packed), packed)
        for dense, mixed in zip(tensors([observation,observation], 'cpu'), tensors([packed,observation], 'cpu')):
            self.assertTrue(torch.equal(dense,mixed))

    def test_real_sac_and_checkpoint_resume_match_uncompressed(self):
        cfg=Config(batch_size=2,warmup=2,horizon_seconds=4.,completion=.2,
                   success_imitation_weight=1.)
        torch.manual_seed(11)
        dense=Learner(cfg)
        initial=copy.deepcopy(dense.state())
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        rollout(dense,cfg,'train',0,'finetune',episode_update_limit=0)
        packed=Learner(Config(**{**cfg.to_dict(),'compact_replay':True}))
        packed.load(initial,optimizers=True)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        rollout(packed,packed.config,'train',0,'finetune',episode_update_limit=0)
        self.assertTrue(packed.success_episodes)
        self.assertTrue(all(t[0][-1].ndim==1 for t in packed.replay))
        for learner in (dense,packed):
            random.seed(23)
            torch.manual_seed(23)
            learner.update()
        self.assertTrue(identical(dense.state(),packed.state()))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'latest.pth'
            checkpoint(p,packed,packed.config,'finetune',1,None,0.)
            state=torch.load(p,map_location='cpu')
            restored=Learner(Config.from_checkpoint(state['config']))
            restored.load(state['learner'],optimizers=True)
            restored.replay.extend(state['replay'])
            restored.success_episodes=state['success_episodes']
            for learner in (dense,restored):
                random.seed(31)
                torch.manual_seed(31)
                learner.update()
            self.assertTrue(identical(dense.state(),restored.state()))

    def test_thousand_map_schedule_preserves_old_split_and_covers_every_map(self):
        cfg=Config(map_family='simple_rooms_v1',train_maps=1000,validation_maps=20,test_maps=20,
                   map_order='shuffled',seed=1,success_memory_mode='per_map',success_episode_limit=1000,
                   compact_replay=True)
        cfg.validate()
        self.assertEqual({training_map_index(cfg,e) for e in range(1,1001)},set(range(1000)))
        old=Config(map_family='simple_rooms_v1',train_maps=100,validation_maps=20,test_maps=20)
        for split,index in (('train',0),('train',99),('validation',19),('test',19)):
            self.assertEqual(configured_map(old,split,index)[2],configured_map(cfg,split,index)[2])
        self.assertFalse(Config.from_checkpoint({}).compact_replay)


if __name__=='__main__': unittest.main()
