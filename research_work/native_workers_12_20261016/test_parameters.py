import argparse
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).parent / 'source'


def namespace():
    tree = ast.parse((SOURCE / 'native_driver.py').read_text())
    names = {'positive_worker_count', 'apply_runtime_workers', 'next_wave_size', 'build_parser', 'main'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(argparse=argparse, Path=Path, now=lambda: 'fixture')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'native-parameter-controls', 'exec'), ns)
    return ns


class ParameterTests(unittest.TestCase):
    def test_driver_constant_is_twelve_and_is_passed_to_finite_entry(self):
        tree = ast.parse((SOURCE / 'driver.py').read_text())
        values = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'NUM_META_AGENT' for t in n.targets)]
        self.assertEqual(values, [12])
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'finite_contact_main']
        self.assertEqual(calls[0].keywords[0].arg, 'default_num_meta_agent')
        self.assertEqual(calls[0].keywords[0].value.id, 'NUM_META_AGENT')

    def test_cli_aliases(self):
        parser = namespace()['build_parser']()
        for flag in ('--num-meta-agent', '--workers'):
            args = parser.parse_args(['--output', '/tmp/run', '--resume', flag, '12'])
            self.assertEqual(args.workers, 12)

    def test_invalid_parallelism_is_rejected(self):
        parser = namespace()['build_parser']()
        for value in ('0', '-1', '1.5', 'wrong'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(['--output', '/tmp/run', '--num-meta-agent', value])

    def test_resume_preserves_saved_parallelism_without_explicit_override(self):
        plan = dict(workers=12)
        namespace()['apply_runtime_workers'](plan, SimpleNamespace(workers=None))
        self.assertEqual(plan, dict(workers=12, NUM_META_AGENT=12))

    def test_explicit_resume_override_is_recorded(self):
        plan = dict(workers=8)
        namespace()['apply_runtime_workers'](plan, SimpleNamespace(workers=12))
        self.assertEqual(plan['workers'], 12)
        self.assertEqual(plan['NUM_META_AGENT'], 12)
        self.assertEqual(plan['parallelism_changes'][0]['previous_workers'], 8)

    def test_new_run_default_comes_from_driver_constant(self):
        ns = namespace()
        ns['prepare'], ns['train'] = Mock(), Mock()
        with patch('sys.argv', ['driver.py', '--output', '/tmp/example', '--dataset', '/tmp/manifest']):
            ns['main'](default_num_meta_agent=12)
        args = ns['prepare'].call_args[0][1]
        self.assertEqual(args.workers, 12)

    def test_switch_catches_up_old_partial_wave_before_using_twelve(self):
        size = namespace()['next_wave_size']
        counts = dict(IR2_FIFO=5136, AB_learned=5128)
        self.assertEqual(size(counts, 'AB_learned', 12), 8)
        counts['AB_learned'] += 8
        self.assertEqual(size(counts, 'IR2_FIFO', 12), 12)
        counts['IR2_FIFO'] += 12
        self.assertEqual(size(counts, 'AB_learned', 12), 12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
