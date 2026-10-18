import importlib.util
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-native-mpl')
from pathlib import Path
import sys
import unittest
import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path.insert(0, str(WORK / 'source'))
import parameter
from graph_generator import Graph_generator

spec = importlib.util.spec_from_file_location('original_graph_before_fix', WORK / 'graph_generator_before.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def setup(cls, narrow=False, peer=None):
    graph = cls(1, (480, 640), 30, 80, 'synthetic-regression', plot=False)
    own = graph.uniform_points[np.argmin(np.linalg.norm(graph.uniform_points - [104, 98], axis=1))].copy()
    belief = np.full((480, 640), 127, np.uint8)
    if narrow:
        own = np.array([101, 101])
        belief[own[1], own[0]] = 255
    else:
        belief[50:150, 50:180] = 255
    if peer is not None:
        belief[peer[1], peer[0]] = 255
    frontiers = np.empty((0, 2), int)
    graph.route_node = [own.copy()]
    graph.generate_graph(own, belief, frontiers)
    positions = [peer, own]
    history = [[[], []], [[own.copy()], []]]
    return graph, (belief, frontiers, frontiers, positions, history, [[0, 0], [0, 0]])


class GraphRecovery(unittest.TestCase):
    def test_empty_peer_candidates_reproduces_old_error_and_new_completes(self):
        peer = np.array([282, 177])
        graph, args = setup(original.Graph_generator, peer=peer)
        with self.assertRaises(IndexError):
            graph.update_graph(*args, step=0, eps=0)
        graph, args = setup(Graph_generator, peer=peer)
        result = graph.update_graph(*args, step=0, eps=0)
        self.assertTrue(result[0])
        self.assertTrue(any(np.array_equal(p, peer) for p in result[1]))

    def test_both_local_windows_without_grid_nodes_keep_pose_and_wait(self):
        graph, args = setup(Graph_generator, narrow=True)
        result = graph.update_graph(*args, step=0, eps=0)
        self.assertTrue(result[0])
        self.assertEqual(result[1].shape[1], 2)
        self.assertTrue(any(np.array_equal(p, args[3][1]) for p in result[1]))

    def test_existing_uniform_pose_case_matches_old_graph_exactly(self):
        old, args_old = setup(original.Graph_generator)
        new, args_new = setup(Graph_generator)
        a, b = old.update_graph(*args_old, step=0, eps=0), new.update_graph(*args_new, step=0, eps=0)
        self.assertEqual(a[0], b[0])
        for i in (1, 3, 4):
            np.testing.assert_array_equal(a[i], b[i])
        def edges(graph):
            return sorted((tuple(source), tuple(destination), edge.length)
                          for source, neighbors in graph.items()
                          for destination, edge in neighbors.items())
        self.assertEqual(edges(a[2]), edges(b[2]))

    def test_received_pose_between_nodes_uses_its_own_component(self):
        # Extract the actual local-dense block to isolate component selection.
        # The cached last graph node lies in a different disconnected component.
        import ast
        tree = ast.parse((WORK / 'source/graph_generator.py').read_text())
        update = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                      for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'update_graph')
        start = next(i for i, n in enumerate(update.body)
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == 'uniform_points_to_check')
        end = next(i for i, n in enumerate(update.body[start:], start)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'robots_local_nodes_combined')
        graph, _ = setup(Graph_generator)
        graph.uniform_points = np.array([[98, 98], [110, 110]])
        graph.node_coords = np.array([[98, 98]])
        belief = np.full((480, 640), 127, np.uint8)
        belief[97:100, 97:100] = 255
        belief[109:112, 109:112] = 255
        namespace = dict(self=graph, robot_belief=belief,
                         robot_location_belief=[np.array([111, 111]), None],
                         np=np, label=original.label, CUR_AGENT_KNN_RAD=80, OTHER_AGENT_KNN_RAD=20)
        exec(compile(ast.Module(body=update.body[start:end], type_ignores=[]), 'local-component', 'exec'), namespace)
        self.assertEqual(namespace['robots_local_nodes'][0], {(110, 110)})


if __name__ == '__main__':
    unittest.main(verbosity=2)
