#!/usr/bin/env python3
"""Replay saved actions and inspect rendezvous/WAIT without training or extra actor inputs."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-finite-contact-matplotlib')

import numpy as np
import torch

from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner, tensors
from finite_contact_train import append_json, provenance, bind_context


def diagnose(learner, cfg, record, output, contact_every):
    info = record['map']
    env = Environment(cfg, info['split'], info['index'], scorer=learner.score)
    trajectory = record['trajectory']
    decisions = []
    for index, expected in enumerate(trajectory[1:]):
        bind_context(learner, env)
        observations = [env.observation(i) for i in range(2)]
        with torch.no_grad():
            inputs = tensors(observations, learner.device)
            logp = learner.policy(*inputs)
            q = torch.minimum(*(net(*inputs)[0].squeeze(-1) for net in learner.critics))
            probabilities = logp.exp().cpu().numpy()
            entropy = -(logp.exp() * logp).sum(-1).cpu().numpy()
            q = q.cpu().numpy()
        row = {'time': env.time, 'map': info, 'repeat': record.get('repeat', 0),
               'decision': index, 'actions': expected['actions'],
               'actor_argmax': probabilities.argmax(-1).tolist(),
               'actual_distance': float(np.linalg.norm(env.positions[0] - env.positions[1])),
               'local_coverage': env.coverage()[0].tolist(), 'robots': []}
        for robot, obs in enumerate(observations):
            valid = obs[4][0] == 0
            moves = valid.copy()
            moves[0] = False
            nodes = obs[0][obs[3][0] == 0]
            age = env.time - env.radio.peer_pose_time[robot]
            pending = env.radio.pending(robot)
            score = env.packet_scores(robot, pending, 'fifo')
            chosen = expected['actions'][robot]
            row['robots'].append({
                'pose_age': age, 'pose_error': float(np.linalg.norm(
                    env.radio.peer_pose[robot][:2] - env.positions[1 - robot])),
                'estimated_delivery': env.expected_delivery(robot),
                'estimated_pending_novelty': float(score.sum()),
                'age_multiplier': math.exp(-age / 60.),
                'rendezvous_max': float(nodes[:, 3].max()),
                'rendezvous_current': float(nodes[0, 3]),
                'rendezvous_positive_nodes': int((nodes[:, 3] > 0).sum()),
                'nodes': len(nodes), 'wait_legal': bool(valid[0]),
                'wait_probability': float(probabilities[robot, 0]),
                'entropy': float(entropy[robot]),
                'alpha': float(learner.log_alpha.exp().item()),
                'wait_q': float(q[robot, 0]),
                'best_move_q': float(q[robot, moves].max()) if moves.any() else None,
                'actor_chosen_q': float(q[robot, chosen]),
                'actor_q_regret': float(q[robot, valid].max() - q[robot, chosen]),
                'max_exploration_utility': float(env.graph_data[robot][2].max()),
                'current_exploration_utility': float(env.graph_data[robot][2][0]),
                'chosen_exploration_utility': float(env.graph_data[robot][2][obs[1][0, chosen]]),
                'frontier_potential': env.frontier_potential(robot),
                'wait_q_minus_best_move': float(q[robot, 0] - q[robot, moves].max()) if moves.any() else None,
                'queue_packets': len(pending),
                'active_map': bool(env.radio.active[(robot, 1 - robot)] is not None and
                                   env.radio.active[(robot, 1 - robot)].kind == 0),
            })
        if index % contact_every == 0:
            # Diagnostic oracle only: assume a contact at frozen positions, with
            # no sensing or movement. Actual receiver maps are NEVER actor inputs.
            radio = copy.deepcopy(env.radio)
            beliefs = [b.copy() for b in env.beliefs]
            gains = np.zeros(2)
            seconds = cfg.estimated_contact_seconds
            elapsed = 0.
            before = radio.sent_bytes
            while elapsed < seconds - 1e-9:
                dt = min(cfg.dt, seconds - elapsed)
                gains += radio.service(env.positions.copy(), np.zeros((2, 2)), env.time + elapsed,
                                       dt, True, beliefs, log_events=False)
                elapsed += dt
            row['oracle_frozen_forced_contact_new_cells'] = gains.tolist()
            row['oracle_frozen_forced_contact_bytes'] = radio.sent_bytes - before
            for mode in ['rule', 'oracle']:
                radio = copy.deepcopy(env.radio)
                beliefs = [b.copy() for b in env.beliefs]
                # The rule uses only the original sender's legal frozen state;
                # the oracle uses receiver truth solely as an unattainable bound.
                scores = {sender: {p.seq: value / p.size for p, value in zip(
                    env.radio.pending(sender), env.packet_scores(sender, env.radio.pending(sender), 'rule'))}
                    for sender in range(2)} if mode == 'rule' else None
                def rank(sender, packets):
                    if mode == 'rule':
                        for packet in packets:
                            if packet.seq not in scores[sender]:
                                scores[sender][packet.seq] = env.packet_scores(sender, [packet], 'rule')[0] / packet.size
                        return max(range(len(packets)), key=lambda i: scores[sender][packets[i].seq])
                    target = beliefs[1 - sender].ravel()
                    return max(range(len(packets)), key=lambda i: np.count_nonzero(
                        target[packets[i].cells()['index']] == 127) / packets[i].size)
                gains = np.zeros(2)
                elapsed = 0.
                while elapsed < seconds - 1e-9:
                    dt = min(cfg.dt, seconds - elapsed)
                    gains += radio.service(env.positions.copy(), np.zeros((2, 2)), env.time + elapsed,
                                           dt, True, beliefs, rank=rank, log_events=False)
                    elapsed += dt
                row[f'oracle_frozen_forced_contact_{mode}_new_cells'] = gains.tolist()
        _, _, terminated, truncated = env.step(expected['actions'])
        actual = env.trajectory[-1]
        for key in ['time', 'positions', 'coverage', 'novel_received_cells', 'connected_seconds']:
            np.testing.assert_allclose(actual[key], expected[key], rtol=0., atol=1e-10,
                                       err_msg=f'decision {index}, {key}')
        row.update({'duration': actual['duration'], 'connected_seconds': actual['connected_seconds'],
                    'new_cells': actual['novel_received_cells']})
        decisions.append(row)
        append_json(output / 'decisions.jsonl', row)
    summary = {'map': info, 'repeat': record.get('repeat', 0), 'decisions': len(decisions),
               'replay_matches': True, 'all_argmax_match': all(r['actions'] == r['actor_argmax'] for r in decisions),
               'metrics': env.metrics(terminated), 'robots': []}
    for robot in range(2):
        waits = [r for r in decisions if r['actions'][robot] == 0]
        total_wait = sum(r['duration'] for r in waits)
        stale = sum(r['duration'] for r in waits if r['robots'][robot]['pose_age'] >= 60)
        zero = sum(r['duration'] for r in waits if r['robots'][robot]['rendezvous_max'] <= 1e-6)
        summary['robots'].append({'explicit_wait_seconds': total_wait,
                                 'wait_with_pose_age_ge_60': stale,
                                 'wait_with_all_rendezvous_zero': zero,
                                 'max_pose_error': max(r['robots'][robot]['pose_error'] for r in decisions)})
    missing = (env.truth == 255) & (env.beliefs[0] != 255) & (env.beliefs[1] != 255)
    yy, xx = np.where(missing)
    summary['final_unseen_free_cells'] = int(missing.sum())
    summary['final_unseen_bbox'] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None
    summary['final_frontier_counts'] = [len(env.frontiers(i)) for i in range(2)]
    summary['final_max_utility'] = [float(env.graph_data[i][2].max()) for i in range(2)]
    append_json(output / 'summary.jsonl', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--contact-every', type=int, default=8)
    args = parser.parse_args()
    if Path(sys.prefix) != Path('/home/wyz/anaconda3/envs/ir2_train'):
        parser.error('Use ir2_train Python')
    if args.contact_every < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('Use a positive contact interval and an empty output directory')
    saved = torch.load(args.checkpoint, map_location='cpu')
    cfg = Config.from_checkpoint(saved['config'])
    if cfg.content == 'learned' and saved['learner']['content_updates'] == 0:
        parser.error('Learned replay requires a trained content head')
    torch.set_num_threads(cfg.torch_threads)
    learner = Learner(cfg, args.device)
    learner.load(saved['learner'])
    args.output.mkdir(parents=True, exist_ok=True)
    provenance(args.output, cfg)
    records = [json.loads(x) for x in (args.evaluation_dir / 'evaluation_trajectories.jsonl').read_text().splitlines()]
    for record in records:
        diagnose(learner, cfg, record, args.output, args.contact_every)


if __name__ == '__main__':
    main()
