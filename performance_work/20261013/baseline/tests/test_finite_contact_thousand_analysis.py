"""Guard against premature results, dropped failures and inflated update counts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from finite_contact_thousand_analysis import finalize, iter_rows, optimizer_audit, paired


class EvidenceTests(unittest.TestCase):
    def test_failures_remain_in_paired_mean_and_success_losses_are_reported(self):
        def record(index, success, seconds):
            return {'map': {'index': index}, 'success': success, 'time': seconds}
        rule = [record(0, True, 100), record(1, True, 200), record(2, False, 256)]
        learned = [record(0, True, 80), record(1, False, 256), record(2, True, 160)]
        value = paired(rule, learned)
        self.assertEqual(value['mean_capped_time_saved_seconds'], 20)
        self.assertEqual(value['both_successful_mean_time_saved_seconds'], 20)
        self.assertEqual(value['gained_success_maps'], [2])
        self.assertEqual(value['lost_success_maps'], [1])
        self.assertEqual(paired(learned, rule)['mean_capped_time_saved_seconds'], -20)

    def test_map_identity_mismatch_is_rejected(self):
        with self.assertRaises(AssertionError):
            paired([{'map': {'index': 0}}], [{'map': {'index': 1}}])

    def test_incomplete_pipeline_cannot_generate_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'pipeline.json').write_text(json.dumps({'status': 'running'}))
            with self.assertRaisesRegex(AssertionError, 'not complete'):
                finalize(root)
            self.assertFalse((root/'report').exists())

    def test_partial_jsonl_append_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'training.jsonl'
            path.write_bytes(b'{"episode":1}\n{"episode":2')
            self.assertEqual(list(iter_rows(path)), [{'episode': 1}])
            with path.open('ab') as handle:
                handle.write(b'}\n')
            self.assertEqual(len(list(iter_rows(path))), 2)

    def test_one_missing_parameter_update_cannot_pass_budget_audit(self):
        def optimizer(step):
            return {'state': {0: {'step': torch.tensor(step)}, 1: {'step': torch.tensor(step)}}}
        initial = {'policy_optimizer': optimizer(7), 'alpha_optimizer': optimizer(7),
                   'q_optimizers': [optimizer(7), optimizer(7)], 'content_optimizer': optimizer(5)}
        final = copy.deepcopy(initial)
        for name in ('policy_optimizer', 'alpha_optimizer'):
            final[name] = optimizer(27)
        final['q_optimizers'] = [optimizer(27), optimizer(27)]
        self.assertEqual(optimizer_audit(initial, final, 20)['policy_optimizer']['increment_per_parameter'], 20)
        final['q_optimizers'][1]['state'][1]['step'] = torch.tensor(26)
        with self.assertRaises(AssertionError):
            optimizer_audit(initial, final, 20)


if __name__ == '__main__':
    unittest.main()
