"""Frozen training-map diagnosis. Never optimizes or writes production models.

The frontier controller is a local-information feasibility witness, not a
learned baseline. Runtime ablations do not modify production source/config.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('IR2_METHOD', 'finite_learned')
os.environ.setdefault('IR2_RUN_NAME', 'contact_reward_diagnostic_20261019')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-frozen-diagnostic-mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import argparse
import hashlib
import heapq
import json
import random
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
import torch
import parameter as config
from baseline_support import prepare_maps, write_json
from contact_module.adapter import ContactAdapter
from model import PolicyNet, QNet
import multi_robot_worker as workers


def goal_distances(coords, edges, goals):
    lookup = {tuple(p): i for i, p in enumerate(coords)}
    reverse = [[] for _ in coords]
    for source, neighbors in edges.items():
        if source not in lookup:
            continue
        for edge in neighbors.values():
            if tuple(edge.to_node) in lookup:
                reverse[lookup[tuple(edge.to_node)]].append((lookup[source], float(edge.length)))
    distance = np.full(len(coords), np.inf)
    queue = []
    for goal in goals:
        distance[goal] = 0.
        heapq.heappush(queue, (0., int(goal)))
    while queue:
        value, node = heapq.heappop(queue)
        if value != distance[node]:
            continue
        for previous, cost in reverse[node]:
            candidate = value + cost
            if candidate < distance[previous]:
                distance[previous] = candidate
                heapq.heappush(queue, (candidate, previous))
    return distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--controller', choices=('policy', 'frontier','oracle_feasibility'), default='policy')
    parser.add_argument('--mask-self', action='store_true')
    parser.add_argument('--steps', type=int, default=196)
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    prepare_maps(config)
    state = torch.load(args.checkpoint, map_location='cpu')
    p = PolicyNet(6, 128, allow_wait=True); q = QNet(6, 128, allow_wait=True)
    p.load_state_dict(state['policy_model']); q.load_state_dict(state['q_net1_model'])
    p.eval(); q.eval()
    contact = ContactAdapter(config.CONTACT_CONFIG, p, state['contact_learning'], args.episode)
    w = workers.Worker(0, 3, p, q, args.episode, device='cpu', contact=contact)
    oracle = None
    if args.controller == 'oracle_feasibility':
        # PRIVILEGED OFFLINE FEASIBILITY ONLY. These actions/data must never
        # enter training or count as learned-policy success. Real sensing and
        # finite packet service remain unchanged in the simulator.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        from scipy.spatial import cKDTree
        truth = w.env.ground_truth
        yy,xx = np.mgrid[2:truth.shape[0]:4,2:truth.shape[1]:4]
        free = truth[yy,xx] == 255
        grid = np.full(free.shape,-1,dtype=int);grid[free]=np.arange(int(free.sum()))
        points = np.stack([xx[free],yy[free]],axis=1)
        sources,targets=[],[]
        for dy,dx in ((0,1),(1,0)):
            start=grid[:grid.shape[0]-dy or None,:grid.shape[1]-dx or None]
            end=grid[dy:,dx:];ok=(start>=0)&(end>=0)
            sources.extend(start[ok]);targets.extend(end[ok])
        all_sources=np.r_[sources,targets];all_targets=np.r_[targets,sources]
        graph=csr_matrix((np.full(len(all_sources),4.),(all_sources,all_targets)),shape=(len(points),len(points)))
        oracle=(points,cKDTree(points),graph,dijkstra)
    if args.mask_self:
        oldutility = contact.utility
        def utility(self, env, robot):
            delta, values, valid = oldutility(env, robot)
            own = env.all_robot_positions_belief[robot][robot]
            values[np.linalg.norm(env.all_node_coords[robot]-own, axis=1) < 30.] = 0.
            return delta, values, valid
        contact.utility = types.MethodType(utility, contact)
    actions, records = [], []
    previous_select = w.select_node
    def select(self, observations, robot):
        nodes, edge, current, _, pad, _ = observations
        coords = self.env.all_node_coords[robot]; here = int(current.item())
        goals = np.flatnonzero((self.env.all_node_utility[robot] > 0) &
                              (np.linalg.norm(coords-coords[here], axis=1) > 1.))
        distances = goal_distances(coords, self.env.all_graph[robot], goals)
        candidate = edge[0, 0].numpy(); valid = pad[0, 0].numpy() == 0
        with torch.no_grad():
            probs = p(*observations).exp()[0].numpy()
        advancing = valid & (distances[candidate] < distances[here]-1e-6)
        if args.controller == 'policy':
            position, action = previous_select(observations, robot)
        elif args.controller == 'oracle_feasibility':
            points,tree,graph,dijkstra=oracle
            unknown=np.flatnonzero(self.env.all_robot_belief[robot][robot][points[:,1],points[:,0]]!=255)
            if len(unknown):
                distances_oracle=dijkstra(graph,directed=False,indices=unknown,min_only=True)
                candidate_grid=tree.query(coords[candidate])[1]
                distance_candidate=distances_oracle[candidate_grid]
                choice=np.flatnonzero(valid)
                index=int(choice[np.argmin(distance_candidate[choice])])
                if index==0 and len(choice)>1:
                    advancing_oracle=choice[distance_candidate[choice]<distance_candidate[0]-1e-6]
                    if len(advancing_oracle):index=int(advancing_oracle[np.argmin(distance_candidate[advancing_oracle])])
            else:
                index=0
            action=torch.tensor([index],dtype=torch.long);position=coords[candidate[index]]
        else:
            if np.isfinite(distances[here]):
                choice = np.flatnonzero(advancing)
                index = int(choice[np.argmin(distances[candidate[choice]])]) if len(choice) else 0
            else:
                # Gather at the common deployment prior after local exploration.
                gather = contact.path_distances(coords, self.env.all_graph[robot], self.env.start_position)
                choice = np.flatnonzero(valid & (gather[candidate] < gather[here]-1e-6))
                index = int(choice[np.argmin(gather[candidate[choice]])]) if len(choice) else 0
            action = torch.tensor([index], dtype=torch.long)
            position = coords[candidate[index]]
        actions.append(dict(step=len(actions)//3, robot=robot,
            own_rv=float(nodes[0, here, 3]), max_rv=float(nodes[0, :len(coords), 3].max()),
            frontier_count=len(self.env.all_frontiers[robot]), positive_nodes=len(goals),
            reachable_positive=bool(np.isfinite(distances[here])),
            wait_probability=float(probs[0]), selected_advance=bool(advancing[int(action.item())]),
            advance_probability=float(probs[advancing].sum())))
        return position, action
    w.select_node = types.MethodType(select, w)
    previous_exchange = contact.exchange
    communication = [0.]
    def exchange(self, *a, **k):
        result = previous_exchange(*a, **k)
        communication[0] = float(result[1]); return result
    contact.exchange = types.MethodType(exchange, contact)
    previous_step = w.env.single_robot_step
    def step(self, robot, positions, episode, index, distance):
        target = self.find_index_from_coords(positions[robot], robot)
        rv = float(self.all_rendezvous_utility_inputs[robot][target])
        exploration = float(self.all_node_utility[robot][target])/50.
        before = np.array([np.count_nonzero(self.all_robot_belief[i][i] == 255) for i in range(3)])
        before_union = np.any(np.stack([self.all_robot_belief[i][i] == 255 for i in range(3)]), axis=0)
        novel = contact.radio.unique_cells
        result = previous_step(robot, positions, episode, index, distance)
        after = np.array([np.count_nonzero(self.all_robot_belief[i][i] == 255) for i in range(3)])
        after_union = np.any(np.stack([self.all_robot_belief[i][i] == 255 for i in range(3)]), axis=0)
        time_cost = contact.config.time_penalty*(distance/contact.config.speed+contact.config.wait_seconds)
        reward = float(result[1]) if result[1] is not None else None
        records.append(dict(step=index, robot=robot, position=positions[robot].tolist(),
            distance=float(distance), wait=bool(distance < 1e-8), rv_reward=rv,
            exploration_utility_reward=exploration, distance_cost=float(distance/512.),
            communication_reward=communication[0], time_cost=float(time_cost), reward=reward,
            frontier_reward=(reward-rv-exploration+distance/512.-communication[0]) if reward is not None else None,
            free_gain=int(np.maximum(after-before, 0).sum()),
            union_free_gain=int(np.count_nonzero(after_union & ~before_union)),
            delivered_novel=contact.radio.unique_cells-novel,
            local_coverage=[float(self.evaluate_exploration_rate(i)) for i in range(3)]))
        if len(records) % 120 == 0:
            print(json.dumps(dict(stage='frozen_reward_probe',controller=args.controller,
                mask_self=args.mask_self,substeps=len(records),map=self.file_path)),flush=True)
        return result
    w.env.single_robot_step = types.MethodType(step, w.env)
    workers.MAX_EPS_STEPS = args.steps
    started = time.monotonic(); valid = w.run_episode(args.episode)
    late = records[-180:]; late_actions = actions[-180:]
    idle_positive = [r for r in records if r['wait'] and not r['free_gain'] and
                     not r['delivered_novel'] and r['reward'] is not None and r['reward'] > 0]
    summary = dict(valid=valid,map=w.env.file_path,controller=args.controller,mask_self=args.mask_self,
        checkpoint_completed=state['completed_episodes'],seed=args.seed,max_team_steps=args.steps,
        actual_substeps=len(records),elapsed_seconds=time.monotonic()-started,
        metrics={k:float(v) for k,v in w.perf_metrics.items()},
        final_local_coverage=[float(w.env.evaluate_exploration_rate(i)) for i in range(3)],
        true_union_coverage=float(np.count_nonzero(np.any(np.stack([w.env.all_robot_belief[i][i] == 255 for i in range(3)]),axis=0)&(w.env.ground_truth==255))/np.count_nonzero(w.env.ground_truth==255)),
        idle_positive_count=len(idle_positive),idle_positive_reward_sum=sum(r['reward'] for r in idle_positive),
        late_wait_fraction=float(np.mean([r['wait'] for r in late])),
        late_no_free_gain_fraction=float(np.mean([r['free_gain']==0 for r in late])),
        late_reachable_positive_fraction=float(np.mean([a['reachable_positive'] for a in late_actions])),
        scope=('PRIVILEGED offline feasibility witness ONLY, never training/learned performance' if oracle is not None else
               'Frozen CPU training-map diagnostic, no optimization; one seed is not a causal performance estimate'),
        records=records,actions=actions)
    write_json(args.output, summary)
    np.savez_compressed(str(args.output)+'.npz',ground_truth=w.env.ground_truth,
        beliefs=np.stack([w.env.all_robot_belief[i][i] for i in range(3)]))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','actions')},indent=2),flush=True)


if __name__ == '__main__':
    main()
