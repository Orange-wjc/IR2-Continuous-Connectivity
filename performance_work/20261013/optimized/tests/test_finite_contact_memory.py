import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import json
import hashlib
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner, compact_observation, dense_observation, tensors
from finite_contact.memory import pack_observation, compact_state, load_checkpoint, semantic_digest
from finite_contact_train import rollout, checkpoint
from finite_contact_multimap_joint import identical
from finite_contact_identity import active_identity


class LosslessMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_observations_restore_every_value_and_dtype(self):
        o=compact_observation(Environment(Config(), 'train', 0).observation(0))
        packed=pack_observation(o)
        self.assertLess(len(packed.payload),sum(x.nbytes for x in o))
        for a,b in zip(o,packed.unpack()):
            self.assertEqual(a.dtype,b.dtype)
            np.testing.assert_array_equal(a,b)
        for a,b in zip(tensors([o,o],'cpu'),tensors([packed,o],'cpu')):
            self.assertTrue(torch.equal(a,b))

    def test_changed_source_requires_review_and_cannot_change_after_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checks').mkdir();(root/'worker.py').write_text('original')
            (root/'pipeline.json').write_text(json.dumps({'source_identity_file':'resume.json'}))
            sha=hashlib.sha256((root/'worker.py').read_bytes()).hexdigest()
            (root/'resume.json').write_text(json.dumps({'worker.py':sha}))
            with patch('finite_contact_identity.ROOT',root):
                with self.assertRaises(FileNotFoundError):active_identity(root)
                review={'status':'passed','source_identity_file':'resume.json',
                        'source_identity_sha256':hashlib.sha256((root/'resume.json').read_bytes()).hexdigest()}
                (root/'checks/memory_recovery_review.json').write_text(json.dumps(review))
                self.assertEqual(active_identity(root),{'worker.py':sha})
                (root/'worker.py').write_text('unreviewed change')
                with self.assertRaises(AssertionError):active_identity(root)

    def test_stream_loader_matches_torch_loader_including_rng_and_shared_storage(self):
        cfg=Config(batch_size=2,warmup=2,horizon_seconds=4,completion=.2,compact_replay=True,
                   success_imitation_weight=1.)
        learner=Learner(cfg)
        rollout(learner,cfg,'train',0,'finetune',episode_update_limit=0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'legacy.pth';checkpoint(p,learner,cfg,'finetune',1,None,0)
            normal=torch.load(p,map_location='cpu');streamed=load_checkpoint(p)
            self.assertEqual(semantic_digest(normal),semantic_digest(streamed))
            before=semantic_digest(streamed);compact_state(streamed)
            self.assertEqual(before,semantic_digest(streamed))
            q=Path(tmp)/'compact.pth';torch.save(streamed,q,pickle_protocol=4)
            self.assertEqual(before,semantic_digest(load_checkpoint(q)))

    def test_actual_rollout_sac_and_full_resume_are_exact(self):
        self.check_actual_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not visible in this process')
    def test_cuda_actual_rollout_sac_and_full_resume_are_exact(self):
        self.check_actual_resume('cuda')

    def check_actual_resume(self,device):
        cfg=Config(batch_size=2,warmup=2,horizon_seconds=4,completion=.2,compact_replay=True,
                   success_imitation_weight=1.)
        torch.manual_seed(12);a=Learner(cfg,device);initial=copy.deepcopy(a.state())
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None
        rollout(a,cfg,'train',0,'finetune',episode_update_limit=0)
        b=Learner(cfg,device);b.load(initial,optimizers=True);b.lossless_observations=True
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        rollout(b,cfg,'train',0,'finetune',episode_update_limit=0)
        self.assertTrue(a.success_episodes and b.success_episodes)
        for learner in (a,b):
            random.seed(43);torch.manual_seed(43);learner.update()
        self.assertTrue(identical(a.state(),b.state()))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'latest.pth';checkpoint(p,b,cfg,'finetune',1,None,0)
            state=load_checkpoint(p);c=Learner(cfg,device);c.load(state['learner'],optimizers=True)
            c.replay.extend(state['replay']);c.success_episodes=state['success_episodes']
            for learner in (a,c):
                random.seed(57);torch.manual_seed(57);learner.update()
            self.assertTrue(identical(a.state(),c.state()))


if __name__=='__main__':unittest.main()
