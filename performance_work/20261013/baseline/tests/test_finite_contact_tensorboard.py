import json
from pathlib import Path
import tempfile
import unittest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from finite_contact_tensorboard import JsonlMirror


class TensorboardMirrorChecks(unittest.TestCase):
    def test_incremental_partial_append_and_source_rewind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'updates.jsonl';output=root/'events'
            source.write_text('{"episode":1,"updates":101,"policy_loss":1}\n')
            mirror=JsonlMirror(source,output,'sac')
            self.assertEqual(mirror.poll(),1)
            self.assertEqual(mirror.poll(),0)
            with source.open('a') as f:f.write('{"episode":2,"updates":102,"policy_loss":2}')
            self.assertEqual(mirror.poll(),0)
            with source.open('a') as f:f.write('\n')
            self.assertEqual(mirror.poll(),1)
            source.rename(root/'old.jsonl')
            source.write_text('{"episode":1,"updates":101,"policy_loss":3}\n')
            self.assertEqual(mirror.poll(),1)
            mirror.close()
            events=EventAccumulator(str(output));events.Reload()
            values=events.Scalars('loss/policy')
            self.assertEqual([(r.step,r.value) for r in values],[(101,3.)])
            self.assertEqual(source.read_text(),'{"episode":1,"updates":101,"policy_loss":3}\n')

    def test_training_window_and_episode_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'training.jsonl'
            records=[{'episode':n,'success':n==1,'time':100*n,'coverage':[.98,.99],
                      'union_coverage':.99,'coverage_deficit':.1,'return':[1.,3.],
                      'sent_bytes':100,'learner_updates':1000+n*20} for n in (1,2)]
            source.write_text(''.join(json.dumps(r)+'\n' for r in records))
            mirror=JsonlMirror(source,root/'events','train');self.assertEqual(mirror.poll(),2);mirror.close()
            events=EventAccumulator(str(root/'events'));events.Reload()
            self.assertEqual([(r.step,r.value) for r in events.Scalars('train/success_rate_last_50')],[(1,1.),(2,.5)])
            self.assertEqual(events.Scalars('train/time_mean_last_50')[-1].value,150.)


if __name__=='__main__':unittest.main()
