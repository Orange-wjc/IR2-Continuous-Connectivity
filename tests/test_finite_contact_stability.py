import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from finite_contact.config import Config
from finite_contact.learning import Learner, sample_success_demonstrations, success_tail_start
from finite_contact_train import checkpoint, rollout
from finite_contact_multimap_joint import identical
from finite_contact_multimap_report import rows
from finite_contact_stability import train_worker, write, recover_initialization, prepare


class CompletionSamplingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def episode(self, name, durations):
        transitions=[]
        for i, duration in enumerate(durations):
            for robot in range(2):
                transitions.append((name, i*2+robot, 0., duration, None,
                                    i==len(durations)-1, False))
        return {'map_key':name,'duration':sum(durations),'transitions':transitions}

    def test_pooled_keeps_historical_draws_and_rng(self):
        episodes=[self.episode('short',[1,1]),self.episode('long',[1]*20)]
        random.seed(27)
        expected=random.sample([x for e in episodes for x in e['transitions']],16)
        rng=random.getstate()
        random.seed(27)
        actual=sample_success_demonstrations(episodes,16,'pooled')
        self.assertEqual(expected,actual)
        self.assertEqual(rng,random.getstate())

    def test_tail_uses_physical_time_including_crossing_decision(self):
        episode=self.episode('variable',[1,1,1,7])
        self.assertEqual(success_tail_start(episode),6)
        self.assertEqual(success_tail_start(self.episode('one',[10])),0)
        self.assertEqual(success_tail_start(self.episode('uniform',[1]*4)),6)

    def test_maps_are_balanced_and_tail_draws_increase(self):
        episodes=[self.episode('short',[1]*4),self.episode('long',[1]*40)]
        random.seed(818)
        batch=sample_success_demonstrations(episodes,12000,'map_tail')
        fraction_short=sum(x[0]=='short' for x in batch)/len(batch)
        self.assertTrue(.48 < fraction_short < .52)
        limits={'short':success_tail_start(episodes[0]),'long':success_tail_start(episodes[1])}
        fraction_tail=sum(x[1]>=limits[x[0]] for x in batch)/len(batch)
        self.assertTrue(.60 < fraction_tail < .65)
        originals={x for e in episodes for x in e['transitions']}
        self.assertTrue(all(x in originals for x in batch))
        self.assertEqual(sample_success_demonstrations([],16,'map_tail'),[])

    def test_old_config_migrates_and_invalid_combination_is_rejected(self):
        self.assertEqual(Config.from_checkpoint({}).success_sampling,'pooled')
        with self.assertRaises(ValueError):
            Config(success_sampling='map_tail').validate()
        with self.assertRaises(ValueError):
            Config(success_sampling='unknown').validate()

    def test_interrupted_initialization_preserves_files_and_never_discards_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'A_fifo'
            directory.mkdir()
            (directory/'latest.tmp').write_bytes(b'partial checkpoint')
            archived=recover_initialization(directory)
            self.assertFalse(directory.exists())
            self.assertEqual((archived/'latest.tmp').read_bytes(),b'partial checkpoint')
            directory.mkdir()
            (directory/'latest.pth').write_bytes(b'committed checkpoint')
            self.assertIsNone(recover_initialization(directory))
            (directory/'latest.pth').unlink()
            (directory/'updates.jsonl').write_text('{"updates": 1}\n')
            with self.assertRaises(RuntimeError):recover_initialization(directory)
            self.assertTrue(directory.exists())

    def test_real_full_state_paired_continuation_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cfg=Config(seed=2,map_family='simple_rooms_v1',train_maps=2,validation_maps=1,test_maps=1,
                       sensor_implementation='vectorized',completion=.2,horizon_seconds=4.,
                       batch_size=2,warmup=2,success_imitation_weight=1.,success_episode_limit=2,
                       success_memory_mode='per_map',variant='A',content='fifo')
            torch.manual_seed(2)
            learner=Learner(cfg)
            rollout(learner,cfg,'train',0,'movement')
            self.assertGreater(learner.updates,0)
            source=root/'source.pth'
            checkpoint(source,learner,cfg,'finetune',2,None,0.)
            write(root/'plan.json',{'sources':{'A_fifo':{'path':str(source),'sha256':'toy-source',
                'episode':2,'sac_updates':learner.updates}},'training_maps':2,'updates_per_map':1,
                'max_hours_per_worker':.1,'monitor_maps':1,'monitor_at':[1,2]})
            saved=torch.load(source,map_location='cpu')
            for condition in ('pooled','map_tail'):
                train_worker(root,condition,'A_fifo','cpu')
                directory=root/condition/'A_fifo'
                initial=torch.load(directory/'initial.pth',map_location='cpu')
                final=torch.load(directory/'latest.pth',map_location='cpu')
                self.assertTrue(identical(initial['learner'],saved['learner']))
                self.assertEqual(final['episode'],4)
                self.assertEqual(final['learner']['updates'],learner.updates+2)
                self.assertTrue(identical(final['learner']['content'],saved['learner']['content']))
                self.assertTrue(identical(final['learner']['context_encoder'],saved['learner']['context_encoder']))
                self.assertEqual(final['config']['success_sampling'],condition)
                updates=rows(directory/'updates.jsonl')
                evaluation=rows(directory/'validation_final/evaluation.jsonl')
                train_worker(root,condition,'A_fifo','cpu',resume=True)
                self.assertEqual(updates,rows(directory/'updates.jsonl'))
                self.assertEqual(evaluation,rows(directory/'validation_final/evaluation.jsonl'))
                restored=torch.load(directory/'latest.pth',map_location='cpu')
                self.assertTrue(identical(final['learner'],restored['learner']))
                self.assertEqual(json.loads((directory/'status.json').read_text())['status'],'complete')
            self.assertEqual(rows(root/'pooled/A_fifo/train_at_000/evaluation.jsonl'),
                             rows(root/'map_tail/A_fifo/train_at_000/evaluation.jsonl'))

    def test_preparation_infers_seed_and_rejects_mixed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for seed in (1,3):
                source=root/('sources_%d'%seed)
                for group,variant,content in [('A_fifo','A','fifo'),('A_rule','A','rule'),
                                              ('AB_learned','AB','learned')]:
                    directory=source/group;directory.mkdir(parents=True)
                    cfg=Config(seed=seed,map_family='simple_rooms_v1',train_maps=100,
                               validation_maps=20,test_maps=20,variant=variant,content=content,
                               success_memory_mode='per_map',success_episode_limit=100)
                    torch.save({'config':cfg.to_dict(),'episode':100,
                                'learner':{'updates':10895,'content_updates':5000}},directory/'latest.pth')
                output=root/('seed_%d'%seed)
                prepare(output,source)
                self.assertEqual(json.loads((output/'plan.json').read_text())['seed'],seed)
                with self.assertRaises(ValueError):prepare(output,source)
                path=source/'AB_learned/latest.pth'
                saved=torch.load(path,map_location='cpu');saved['config']['seed']=seed+1
                torch.save(saved,path)
                with self.assertRaises(AssertionError):prepare(root/('mixed_%d'%seed),source)


if __name__=='__main__':unittest.main()
