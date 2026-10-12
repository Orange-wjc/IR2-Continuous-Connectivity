import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from finite_contact_multiseed import seed_statistics, run_worker_queue
from finite_contact_multimap_joint import GROUPS


class AdaptationSeedStatisticsChecks(unittest.TestCase):
    def records(self):
        return [{'seed': i+1, 'groups': {g: {'maps': 20, 'successes': success,
                    'mean_capped_time': duration + (10 if g != 'AB_learned' else 0),
                    'mean_minimum_coverage': .99, 'mean_coverage_deficit': .1}
                for g in GROUPS}} for i, (success, duration) in enumerate(((18, 100), (20, 120), (16, 140)))]

    def test_statistics_use_model_seed_and_preserve_paired_differences(self):
        result = seed_statistics(self.records())
        learned = result['groups']['AB_learned']
        self.assertEqual(learned['successes']['mean'], 18)
        self.assertEqual(learned['successes']['sample_std'], 2)
        self.assertEqual(learned['mean_capped_time']['mean'], 120)
        self.assertEqual(learned['mean_capped_time']['sample_std'], 20)
        self.assertEqual(len(learned['mean_capped_time']['per_seed']), 3)
        self.assertTrue(all(r['time_saved_seconds'] == 10 for r in result['paired_learned_vs']['A_rule']))
        self.assertEqual([r['seed'] for r in result['paired_learned_vs']['A_rule']], [1, 2, 3])

    def test_incomplete_duplicate_or_nonfinite_runs_cannot_be_aggregated(self):
        original = self.records()
        for field, value in [('maps', 19), ('mean_capped_time', float('nan')), ('successes', 21)]:
            records = copy.deepcopy(original)
            records[2]['groups']['A_rule'][field] = value
            with self.assertRaises(ValueError):
                seed_statistics(records)
        duplicate = copy.deepcopy(original)
        duplicate[2]['seed'] = 1
        with self.assertRaises(ValueError):
            seed_statistics(duplicate)

    def test_real_worker_queue_limits_concurrency_and_finishes_all_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            code = "import json,time,sys; p=sys.argv[1]; a=time.monotonic(); time.sleep(.15); open(p,'w').write(json.dumps([a,time.monotonic()]))"
            jobs = [(directory, str(i), [sys.executable, '-c', code, str(directory/('%d.json'%i))]) for i in range(5)]
            run_worker_queue(jobs, 2)
            intervals = [json.loads((directory/('%d.json'%i)).read_text()) for i in range(5)]
            for start, _ in intervals:
                self.assertLessEqual(sum(a <= start < b for a,b in intervals), 2)

    def test_failed_queued_job_prevents_launching_remaining_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            marker = directory/'should_not_exist'
            jobs = [(directory, 'fail', [sys.executable, '-c', 'raise SystemExit(1)']),
                    (directory, 'later', [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).touch()', str(marker)])]
            with self.assertRaises(RuntimeError):
                run_worker_queue(jobs, 1)
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
