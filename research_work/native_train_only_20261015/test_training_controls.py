"""Exercise the real coordinator loop without maps, GPU work or checkpoints.

The fake learner records training updates; a forbidden evaluation raises, so
crossing an evaluation boundary must actually continue committing training.
"""
import ast
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

SOURCE = Path(__file__).parent / 'source/native_driver.py'


class TrainingControls(unittest.TestCase):
    def namespace(self):
        tree = ast.parse(SOURCE.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('automatic_review_due', 'train')]
        ns = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), ns)
        return ns

    def test_review_policy_defaults_and_alignment(self):
        due = self.namespace()['automatic_review_due']
        plan = dict(review_interval=512)
        self.assertFalse(due(plan, dict(a=504, b=504)))
        self.assertFalse(due(plan, dict(a=512, b=504)))
        self.assertTrue(due(plan, dict(a=512, b=512)))
        plan['automatic_development_evaluation'] = False
        for episode in (512, 1024, 2048, 100000):
            self.assertFalse(due(plan, dict(a=episode, b=episode)))

    def simulate(self, stored_policy, cli_disable):
        ns = self.namespace()
        evaluations, commits, snapshots = [], [], []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups = {'IR2_FIFO': ('B0', 'fifo'), 'AB_learned': ('AB', 'learned')}
            plan = dict(source_identity={}, dataset_sha256='data', workers=8,
                        microbatch=64, preflight=False, wall_guard_hours=48,
                        paired_learning_enabled=True, cross_method_prefetch=True,
                        review_interval=512)
            if stored_policy is not None:
                plan['automatic_development_evaluation'] = stored_policy

            def write(path, value):
                path = Path(path)
                path.write_text(json.dumps(value))
                if path.name == 'pipeline.json':
                    snapshots.append(json.loads(json.dumps(value)))

            write(root / 'pipeline.json', plan)
            write(root / 'head_complete.json', dict(sha256='data', source_sac_updates=216))
            for group in groups:
                (root / group).mkdir()
                (root / group / 'latest.pth').touch()

            class Frozen:
                def requires_grad_(self, value):
                    assert value is False
                    return self

                def eval(self):
                    return self

            class Learner:
                def __init__(self):
                    self.updates = 4184
                    self.content_updates = 5000
                    self.replay = [0, 1]
                    self.content = Frozen()
                    self.context_encoder = Frozen()

            class Future:
                def __init__(self, job):
                    self.job = job

                def result(self):
                    return dict(metric=dict(episode=self.job['episode'], map=dict(source_file='train.png')),
                                worker_rss_mib=550)

                def cancelled(self):
                    return False

                def exception(self):
                    return None

            class Pool:
                def __init__(self, **kwargs):
                    pass

                def submit(self, function, job):
                    return Future(job)

                def shutdown(self, wait):
                    assert wait

            def restore(path, microbatch):
                group = Path(path).parent.name
                variant, content = groups[group]
                cfg = SimpleNamespace(variant=variant, content=content, warmup=2,
                                      dataset_manifest=str((root / 'dataset/native_dataset.json').resolve()))
                return Learner(), cfg, 504, {}

            def commit(learner, cfg, directory, item, episode, writer, enable_updates):
                assert enable_updates
                learner.updates += 8
                commits.append((directory.name, episode, learner.updates))
                if directory.name == 'AB_learned' and episode == 520:
                    (root / 'STOP').touch()

            def evaluate(root, plan, pool, learners, configs, completed):
                if stored_policy is False or cli_disable:
                    raise AssertionError('Automatic evaluation ran in manual-only mode')
                evaluations.append(dict(completed))
                plan['last_review_episode'] = min(completed.values())

            class Writer:
                def __init__(self, *args, **kwargs):
                    pass

                def close(self):
                    pass

            lock_handles = []

            def flock(handle, operation):
                lock_handles.append(handle)
                fcntl.flock(handle, operation)

            ns.update(Path=Path, os=os, json=json,
                      fcntl=SimpleNamespace(flock=flock, LOCK_EX=fcntl.LOCK_EX, LOCK_NB=fcntl.LOCK_NB),
                      GROUPS=groups,
                      identity=lambda: {}, digest=lambda path: 'data',
                      read=lambda path: json.loads(Path(path).read_text()), write=write,
                      now=lambda: 'simulation', torch=SimpleNamespace(set_num_threads=lambda n: None),
                      signal=SimpleNamespace(SIGTERM=15, SIGINT=2, signal=lambda *a: None),
                      time=SimpleNamespace(monotonic=lambda: 0),
                      multiprocessing=SimpleNamespace(get_context=lambda name: None),
                      ProcessPoolExecutor=Pool, initialize_worker=None,
                      restore_state=restore, recover_logs=lambda *a: None,
                      SummaryWriter=Writer, restore_rng=lambda rng: None,
                      jobs_for=lambda cfg, learner, directory, first, count:
                          [dict(episode=ep) for ep in range(first, first + count)],
                      run_episode=None, commit_episode=commit,
                      save_state=lambda *a: {}, evaluate_pair=evaluate)
            try:
                ns['train'](root, SimpleNamespace(no_auto_evaluation=cli_disable))
            finally:
                for handle in lock_handles:
                    handle.close()
            final = json.loads((root / 'pipeline.json').read_text())
            self.assertEqual(final['completed_episodes'], dict(IR2_FIFO=520, AB_learned=520))
            self.assertEqual(final['sac_updates'], dict(IR2_FIFO=4312, AB_learned=4312))
            self.assertEqual(len(commits), 32)
            return evaluations, final, snapshots

    def test_manual_only_crosses_boundary_and_persists_on_resume(self):
        evaluations, plan, snapshots = self.simulate(False, False)
        self.assertEqual(evaluations, [])
        self.assertFalse(plan['automatic_development_evaluation'])
        self.assertNotIn('easy_development_review', [p.get('stage') for p in snapshots])

    def test_cli_disables_existing_automatic_policy(self):
        evaluations, plan, snapshots = self.simulate(True, True)
        self.assertEqual(evaluations, [])
        self.assertFalse(plan['automatic_development_evaluation'])

    def test_legacy_default_still_evaluates_once_at_512(self):
        evaluations, plan, snapshots = self.simulate(None, False)
        self.assertEqual(evaluations, [dict(IR2_FIFO=512, AB_learned=512)])
        self.assertEqual(plan['last_review_episode'], 512)


if __name__ == '__main__':
    unittest.main(verbosity=2)
