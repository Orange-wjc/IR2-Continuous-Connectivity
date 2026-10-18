"""Check mechanism isolation, information permissions and real learning for new controls."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
import json
from unittest.mock import patch
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner
from finite_contact.memory import load_checkpoint
from finite_contact_train import rollout, checkpoint
from finite_contact_multimap_joint import identical
from finite_contact_ablation import group_config, NEW, OLD
import finite_contact_multimap_joint as joint


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_only_declared_mechanism_switches_change(self):
        common={'config':Config().to_dict()}
        for group,switches in {**OLD,**NEW}.items():
            cfg=group_config(common,group).to_dict()
            self.assertEqual((cfg['variant'],cfg['content'],cfg['coupled']),switches)
            self.assertEqual({k:v for k,v in cfg.items() if k not in ('variant','content','coupled')},
                             {k:v for k,v in common['config'].items() if k not in ('variant','content','coupled')})

    def test_b0_and_b_use_same_surplus_independent_of_learned_scores(self):
        a=Environment(Config(variant='B0',content='fifo'))
        b=Environment(Config(variant='B',content='learned'),scorer=lambda f,c:np.zeros(len(f)))
        with patch.object(a,'expected_delivery',side_effect=AssertionError('A leaked into B0')), \
             patch.object(b,'expected_delivery',side_effect=AssertionError('A leaked into B')):
            for i in (0,1):
                for x,y in zip(a.observation(i),b.observation(i)):np.testing.assert_array_equal(x,y)

    def test_uncoupled_uses_fifo_benefit_but_keeps_learned_transmission(self):
        env=Environment(Config(variant='AB',content='learned',coupled=False))
        before=env.observation(0)
        fifo_gain=env.expected_delivery(0)
        self.assertGreater(fifo_gain,0)
        env.config.content='fifo'
        for a,b in zip(before,env.observation(0)):np.testing.assert_array_equal(a,b)
        env.config.content='learned'
        with patch.object(env,'packet_scores',return_value=np.arange(len(env.radio.pending(0)))) as scorer:
            index=env.rank_packet(0,env.radio.pending(0))
            self.assertGreater(index,0)
            self.assertEqual(len(scorer.call_args[0]),2) # default mode resolves learned
        env.config.coupled=True
        env.scorer=lambda f,c:np.zeros(len(f))
        env.feature_provider=lambda sender,indices:np.zeros((len(indices),64),np.float32)
        self.assertEqual(env.expected_delivery(0),0)
        self.assertFalse(np.array_equal(before[0],env.observation(0)[0]))

    def test_new_controls_cannot_read_hidden_receiver_or_truth(self):
        for g in NEW:
            cfg=group_config({'config':Config().to_dict()},g)
            env=Environment(cfg,scorer=lambda f,c:np.ones(len(f)))
            env.feature_provider=lambda sender,indices:np.zeros((len(indices),64),np.float32)
            before=env.observation(0)
            packets=env.radio.pending(0)
            scores=env.packet_scores(0,packets)
            env.beliefs[1][:]=255;env.positions[1][:]=220;env.truth[:]=1
            for a,b in zip(before,env.observation(0)):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(scores,env.packet_scores(0,packets))

    def test_actual_rollout_update_and_resume_for_all_new_modes(self):
        for g in NEW:
            cfg=group_config({'config':Config(batch_size=2,warmup=2,horizon_seconds=4,
                 completion=.2,compact_replay=True,success_imitation_weight=1.).to_dict()},g)
            random.seed(17);np.random.seed(17);torch.manual_seed(17)
            learner=Learner(cfg);learner.lossless_observations=True
            head=copy.deepcopy(learner.state()['content'])
            metrics,_=rollout(learner,cfg,'train',0,'finetune',episode_update_limit=0)
            self.assertFalse(metrics['interrupted']);self.assertEqual(learner.updates,0)
            self.assertTrue(learner.update());self.assertEqual(learner.updates,1)
            self.assertTrue(identical(head,learner.state()['content']))
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'latest.pth';checkpoint(p,learner,cfg,'finetune',1,None,0)
                state=load_checkpoint(p);restored=Learner(cfg);restored.load(state['learner'],optimizers=True)
                restored.replay.extend(state['replay']);restored.success_episodes=state['success_episodes']
                for item in (learner,restored):
                    random.seed(43);torch.manual_seed(43);item.update()
                self.assertTrue(identical(learner.state(),restored.state()))

    def test_all_new_workers_commit_complete_training_and_endpoint_evaluation(self):
        cfg=Config(batch_size=2,warmup=2,horizon_seconds=4,completion=.2,
                   compact_replay=True,train_maps=2,validation_maps=20,test_maps=20,
                   map_family='simple_rooms_v1',success_imitation_weight=1.)
        common={'config':cfg.to_dict(),'learner':Learner(cfg).state()}
        plan={'max_hours_per_group':.1,'sweeps':1,'updates_per_map':1,
              'source_sac_updates':0,'new_sac_updates_per_group':2}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);torch.save(common,root/'common_start.pth')
            def provenance(directory,config):
                (directory/'config.json').write_text(json.dumps(config.to_dict()))
            with patch.object(joint,'group_config',group_config), patch.object(joint,'provenance',provenance), \
                 patch('finite_contact_train.provenance',provenance):
                for g in NEW:
                    joint.train_group(root,plan,g,'cpu')
                    d=root/g
                    status=json.loads((d/'status.json').read_text())
                    self.assertEqual(status['status'],'complete')
                    self.assertEqual(status['completed_episodes'],2)
                    self.assertEqual(status['final_sac_updates'],2)
                    dev=[json.loads(x) for x in (d/'validation_final/evaluation.jsonl').read_text().splitlines()]
                    self.assertEqual(len(dev),20)
                    self.assertEqual([r['map']['index'] for r in dev],list(range(20)))
                    self.assertTrue(all(r['learner_updates']==2 for r in dev))
                    self.assertEqual(load_checkpoint(d/'latest.pth')['episode'],2)


if __name__=='__main__':unittest.main()
