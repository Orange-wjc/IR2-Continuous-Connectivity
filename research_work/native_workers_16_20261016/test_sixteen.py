import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).parent / 'source'


def helpers():
    tree = ast.parse((SOURCE / 'native_driver.py').read_text())
    names = {'positive_worker_count', 'apply_runtime_workers', 'next_wave_size', 'build_parser'}
    namespace = dict(argparse=argparse, Path=Path, now=lambda: 'fixture')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'parallel-controls', 'exec'), namespace)
    return namespace


class SixteenWorkers(unittest.TestCase):
    def test_driver_default_is_sixteen(self):
        tree = ast.parse((SOURCE / 'driver.py').read_text())
        values = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'NUM_META_AGENT' for t in n.targets)]
        self.assertEqual(values, [16])

    def test_both_cli_names_accept_sixteen_on_resume(self):
        namespace = helpers()
        for name in ('--num-meta-agent', '--workers'):
            args = namespace['build_parser']().parse_args(['--output', '/tmp/example', '--resume', name, '16'])
            plan = dict(workers=12)
            namespace['apply_runtime_workers'](plan, args)
            self.assertEqual(plan['workers'], 16)
            self.assertEqual(plan['NUM_META_AGENT'], 16)
            self.assertEqual(plan['parallelism_changes'][0]['previous_workers'], 12)

    def test_resume_without_flag_preserves_saved_setting(self):
        plan = dict(workers=12)
        helpers()['apply_runtime_workers'](plan, SimpleNamespace(workers=None))
        self.assertEqual(plan['workers'], 12)

    def test_old_twelve_episode_gap_is_caught_up_before_sixteen_wave(self):
        size = helpers()['next_wave_size']
        counts = dict(IR2_FIFO=5160, AB_learned=5148)
        self.assertEqual(size(counts, 'AB_learned', 16), 12)
        counts['AB_learned'] += 12
        self.assertEqual(size(counts, 'IR2_FIFO', 16), 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
