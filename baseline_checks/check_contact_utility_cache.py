"""Captured v4 local-map counterexample: an unvisited reachable goal is hidden.

No simulator truth or peer maps exist on the fixture. The repair must use only
the captured own map/frontiers and must not resurrect an actually visited node.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['IR2_METHOD'] = 'finite_learned'
os.environ['IR2_RUN_NAME'] = 'contact_utility_validation'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import json
import hashlib
from types import SimpleNamespace
import numpy as np
import parameter as config
from node import Node
from contact_module.adapter import ContactAdapter
from baseline_support import write_json


def fixture(saved, visited_goal=False):
    coords, belief, frontiers = saved['coords'], saved['belief'].copy(), saved['frontiers']
    nodes = [Node(c, frontiers, belief) for c in coords]
    cached = saved['cached_utility'].copy()
    for node, value in zip(nodes, cached):
        node.utility = int(value)
        node.zero_utility_node = value == 0
        if value == 0:
            node.observable_frontiers = set()
    route = [c.copy() for c, visited in zip(coords, saved['exact_visited']) if visited]
    if visited_goal:
        route.append(coords[72].copy())
    generator = SimpleNamespace(nodes_list=nodes, route_node=route, node_utility=cached.copy())
    # Deliberately no ground_truth, peer map, radio, or policy on this fixture.
    env = SimpleNamespace(all_robot_belief=[[belief]], all_frontiers=[frontiers],
                          all_graph_generator=[generator], all_node_utility=[cached.copy()])
    adapter = ContactAdapter.__new__(ContactAdapter)
    adapter._utility_known_counts = {}
    adapter.utility_recoveries = 0
    return adapter, env


def main():
    path = ROOT/'baseline_checks/fixtures/utility_cache_v4_robot2.npz'
    saved = np.load(path)
    index = 72
    assert saved['coords'][index].tolist() == [52, 156]
    assert saved['cached_utility'][index] == 0 and saved['fresh_utility'][index] == 7
    assert not saved['exact_visited'][index] and not saved['guide_visited'][index]
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    edges = csr_matrix((saved['edge_lengths'], (saved['edge_sources'], saved['edge_targets'])),
                       shape=(len(saved['coords']), len(saved['coords'])))
    distance = float(dijkstra(edges, directed=True, indices=int(saved['current']))[index])
    assert np.isfinite(distance)
    adapter, env = fixture(saved)
    before = env.all_robot_belief[0][0].copy()
    count = adapter.refresh_frontier_utility(env, 0)
    assert count == 1 and env.all_node_utility[0][index] == 7
    assert env.all_graph_generator[0].nodes_list[index].observable_frontiers
    assert env.all_graph_generator[0].node_utility[index] == 7
    assert np.array_equal(before, env.all_robot_belief[0][0])
    assert adapter.refresh_frontier_utility(env, 0) == 0
    assert adapter.utility_recoveries == 1
    visited, env_visited = fixture(saved, visited_goal=True)
    assert visited.refresh_frontier_utility(env_visited, 0) == 0
    assert env_visited.all_node_utility[0][index] == 0
    # Re-use the same callback after genuinely new locally known cells arrive.
    growing, env_growing = fixture(saved)
    belief = env_growing.all_robot_belief[0][0]
    center = saved['coords'][index]
    ring = [(int(center[1]+dy), int(center[0]+dx)) for dy in (-1, 0, 1)
            for dx in (-1, 0, 1) if dx or dy]
    original = [int(belief[y, x]) for y, x in ring]
    for y, x in ring:
        belief[y, x] = 127
    assert Node(center, saved['frontiers'], belief).utility == 0
    assert growing.refresh_frontier_utility(env_growing, 0) == 0
    for (y, x), value in zip(ring, original):
        belief[y, x] = value
    assert growing.refresh_frontier_utility(env_growing, 0) == 1
    assert env_growing.all_node_utility[0][index] == 7
    result = dict(passed=True, fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  node= saved['coords'][index].tolist(), cached_before=0, fresh_after=7,
                  reachable_graph_distance=distance, actual_visited_zero_preserved=True,
                  known_map_growth_reactivates_existing_unvisited_node=True,
                  no_truth_or_peer_map_access=True, own_belief_unchanged=True,
                  repeated_unchanged_map_is_idempotent=True,
                  scope='Captured local-state regression, not learned-policy performance')
    write_json('baseline_checks/contact_utility_cache_verified.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
