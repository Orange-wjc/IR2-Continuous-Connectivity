import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import supervisor_new as s


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.root = base / 'repo/experiments/run'
        self.root.mkdir(parents=True)
        self.watch = base / 'watch'
        self.watch.mkdir()
        self.worker = s.Supervisor(self.root, self.watch)
        (self.root / 'source').mkdir()
        (self.root / 'source/graph_generator.py').write_text('def graph(values):\n    return values[-1]\n')
        (self.root / 'source/model.py').write_text('PROTECTED = True\n')
        (self.root / 'dataset').mkdir()
        s.write(self.root / 'dataset/native_dataset.json', {'fixture': True})
        for group in ('IR2_FIFO', 'AB_learned'):
            (self.root / group).mkdir()
            (self.root / group / 'latest.pth').write_bytes(b'x' * 1000001)
            s.write(self.root / group / 'status.json', {'episodes': 8})
        self.plan = dict(status='interrupted_or_failed', stage='easy_paired_training',
                         automatic_development_evaluation=False, groups=['IR2_FIFO', 'AB_learned'],
                         workers=8, effective_batch=128, updates_per_completed_episode=8,
                         preflight=False, test_policy_evaluations=0, wall_guard_hours=48,
                         microbatch=64, source_identity=s.source_identity(self.root / 'source'),
                         dataset_sha256=s.sha(self.root / 'dataset/native_dataset.json'),
                         completed_episodes=dict(IR2_FIFO=8, AB_learned=8), error='IndexError: empty candidates',
                         updated=s.now())
        s.write(self.root / 'pipeline.json', self.plan)
        s.write(self.root / 'source_identity.json', self.plan['source_identity'])
        s.write(self.root / 'coordinator.json', {'time': s.now(), 'pid': 999999})
        self.match = patch.object(s, 'matching_processes', return_value=[])
        self.match_mock = self.match.start()
        self.worker.resources_ok = Mock(return_value=True)
        self.worker.restart = Mock()

    def tearDown(self):
        self.match.stop()
        self.temp.cleanup()

    def test_twelve_worker_policy_is_accepted(self):
        self.plan.update(workers=12,NUM_META_AGENT=12)
        s.fixed_policy(self.plan)

    def test_inconsistent_worker_alias_is_rejected(self):
        self.plan.update(workers=12,NUM_META_AGENT=8)
        with self.assertRaises(AssertionError):s.fixed_policy(self.plan)

    def test_invalid_worker_counts_are_rejected(self):
        for value in (0,-1,True):
            self.plan['workers']=value
            with self.assertRaises(AssertionError):s.fixed_policy(self.plan)

    def test_healthy_training_never_calls_repair_or_restart(self):
        self.match_mock.return_value = [123]
        self.worker.propose = Mock(side_effect=AssertionError('must not call model'))
        self.worker.tick()
        self.assertEqual(self.worker.state['status'], 'healthy')
        self.worker.propose.assert_not_called()
        self.worker.restart.assert_not_called()

    def test_training_stop_marker_is_respected(self):
        (self.root / 'STOP').touch()
        self.worker.tick()
        self.assertEqual(self.worker.state['status'], 'user_paused')
        self.worker.restart.assert_not_called()

    def test_supervisor_pause_does_not_stop_training(self):
        (self.watch / 'PAUSE').touch()
        self.match_mock.return_value = [123]
        self.worker.tick()
        self.assertEqual(self.worker.state['status'], 'user_paused')
        self.assertFalse((self.root / 'STOP').exists())

    def test_manual_signal_pause_is_not_restarted(self):
        self.plan['status'] = 'paused_for_easy_review'
        s.write(self.root / 'pipeline.json', self.plan)
        self.worker.tick()
        self.worker.restart.assert_not_called()
        self.assertEqual(self.worker.state['status'], 'user_paused')

    def test_48_hour_guard_reviews_resources_then_continues(self):
        self.plan['status'] = 'paused_for_easy_review'
        self.plan['updated'] = (datetime.now(timezone.utc) + timedelta(hours=49)).isoformat()
        s.write(self.root / 'pipeline.json', self.plan)
        self.worker.tick()
        self.worker.restart.assert_called_once()

    def test_insufficient_resources_wait_without_model_call(self):
        self.worker.resources_ok.return_value = False
        self.worker.tick()
        self.assertEqual(self.worker.state['status'], 'waiting_resources')
        self.worker.restart.assert_not_called()

    def test_cuda_oom_reduces_microbatch_preserving_effective_batch(self):
        self.plan['error'] = 'CUDA out of memory'
        s.write(self.root / 'pipeline.json', self.plan)
        self.worker.tick()
        plan = self.worker.restart.call_args[0][0]
        self.assertEqual(plan['microbatch'], 32)
        self.assertEqual(plan['effective_batch'], 128)
        self.assertEqual(plan['workers'], 8)

    def test_held_training_lock_prevents_repair(self):
        with s.training_locks(self.root):
            self.worker.tick()
        self.assertEqual(self.worker.state['status'], 'waiting_for_training_lock')
        self.worker.restart.assert_not_called()

    def test_orphan_sampling_workers_are_cleaned_before_recovery(self):
        with patch.object(s, 'orphan_workers', return_value=[998877]), patch.object(s.os, 'kill') as kill:
            self.worker.tick()
        kill.assert_called_once_with(998877, s.signal.SIGTERM)
        self.assertEqual(self.worker.state['status'], 'waiting_for_orphan_exit')
        self.worker.restart.assert_not_called()

    def test_explicit_resume_clears_blocked_repair_state(self):
        self.match_mock.return_value = [123]
        self.worker.state.update(blocked_fingerprint='old', retry_after=99999999999, attempts={'old': 3})
        (self.watch / 'RETRY').touch()
        self.worker.tick()
        self.assertNotIn('blocked_fingerprint', self.worker.state)
        self.assertEqual(self.worker.state['attempts'], {})
        self.assertFalse((self.watch / 'RETRY').exists())

    def test_failed_repairs_back_off_then_block_without_blind_restart(self):
        self.worker.propose = Mock(side_effect=RuntimeError('reproduction not established'))
        for attempt in range(3):
            self.worker.state.pop('retry_after', None)
            self.worker.tick()
        self.assertEqual(self.worker.state['status'], 'needs_attention')
        self.assertTrue((self.watch / 'NEEDS_ATTENTION.md').exists())
        self.worker.tick()
        self.assertEqual(self.worker.propose.call_count, 3)
        self.worker.restart.assert_not_called()

    def proposal_fixture(self):
        directory = self.worker.incident(self.plan, 'fixture')
        (directory / 'source/graph_generator.py').write_text('def graph(values):\n    return values[-1] if values else None\n')
        (directory / 'validate.py').write_text('pass\n')
        proposal = dict(changed_files=['graph_generator.py'], validation_script='validate.py', summary='empty candidate fix')
        validation = dict(passed=True, regression_tests=1, failed_case_reproduced_before=True,
                          failed_case_completed_after=True, test_policy_executions=0, sac_updates=0)
        def fake_child(*args, **kwargs):
            s.write(directory / 'validation_result.json', validation)
        self.worker.run_child = fake_child
        return directory, proposal, validation

    def test_model_changes_are_rejected_before_activation(self):
        directory, proposal, _ = self.proposal_fixture()
        (directory / 'source/model.py').write_text('PROTECTED = False\n')
        with self.assertRaises(AssertionError):
            self.worker.validate(directory, self.plan, proposal)
        self.assertEqual((self.root / 'source/model.py').read_text(), 'PROTECTED = True\n')

    def test_validation_then_activation_preserves_checkpoints_and_archives_source(self):
        directory, proposal, validation = self.proposal_fixture()
        changed, after, result = self.worker.validate(directory, self.plan, proposal)
        hashes = {g: s.sha(self.root / g / 'latest.pth') for g in self.plan['groups']}
        self.worker.activate(directory, self.plan, proposal, changed, after, result)
        saved = s.read(self.root / 'pipeline.json')
        self.assertEqual(saved['source_identity'], s.source_identity(self.root / 'source'))
        self.assertFalse(saved['automatic_development_evaluation'])
        self.assertEqual(hashes, {g: s.sha(self.root / g / 'latest.pth') for g in self.plan['groups']})
        self.assertTrue((directory / 'repair.patch').exists())
        self.assertFalse((self.watch / 'activation_transaction.json').exists())

    def test_interrupted_activation_rolls_back_to_recorded_identity(self):
        archive = self.root / 'old_source'
        import shutil
        shutil.copytree(self.root / 'source', archive)
        (self.root / 'source/graph_generator.py').write_text('raise RuntimeError("partial swap")\n')
        s.write(self.watch / 'activation_transaction.json',
                dict(archive=str(archive), before=self.plan['source_identity'], after={'wrong': 'identity'}))
        restored = self.worker.recover_transaction(copy.deepcopy(self.plan))
        self.assertEqual(s.source_identity(self.root / 'source'), restored['source_identity'])

    def test_restart_uses_saved_run_and_manual_evaluation_flag(self):
        self.worker.restart = s.Supervisor.restart.__get__(self.worker)
        old_hashes = {g: s.sha(self.root / g / 'latest.pth') for g in self.plan['groups']}
        with patch.object(s.subprocess, 'Popen', return_value=Mock(pid=789)) as spawn:
            self.worker.restart(self.plan, 'test external interruption')
        command = spawn.call_args[0][0]
        self.assertIn('--resume', command)
        self.assertIn('--no-auto-evaluation', command)
        self.assertEqual(old_hashes, {g: s.sha(self.root / g / 'latest.pth') for g in self.plan['groups']})
        self.assertEqual(self.worker.state['status'], 'restarting')

    def test_process_identity_requires_exact_run_and_non_zombie(self):
        self.match.stop()
        try:
            proc = self.watch / 'proc'
            proc.mkdir()
            for pid, output, state in ((123, self.root, 'R'), (124, self.root, 'Z'), (125, self.watch, 'R')):
                d = proc / str(pid)
                d.mkdir()
                args = [s.PYTHON, str(self.root / 'source/driver.py'), '--finite-contact', '--output', str(output), '--resume']
                (d / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())
                (d / 'stat').write_text('%d (python) %s 1 2 3' % (pid, state))
            self.assertEqual(s.matching_processes(self.root, proc), [123])
        finally:
            self.match.start()


if __name__ == '__main__':
    unittest.main(verbosity=2)
