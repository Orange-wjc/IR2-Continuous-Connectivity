#!/usr/bin/env python3
"""Evaluate frozen content heads on saved states; receiver maps are offline labels only."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-finite-contact-matplotlib')

import numpy as np
import torch

from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner
from finite_contact_train import append_json, provenance


def window_gain(env, mode, score_cache):
    radio = copy.deepcopy(env.radio)
    beliefs = [b.copy() for b in env.beliefs]
    gains = np.zeros(2)
    elapsed = 0.
    seconds = env.config.estimated_contact_seconds
    before = radio.sent_bytes

    def rank(sender, packets):
        if mode == 'oracle':
            target = beliefs[1 - sender].ravel()
            return max(range(len(packets)), key=lambda i: np.count_nonzero(
                target[packets[i].cells()['index']] == 127) / packets[i].size)
        cache = score_cache[mode][sender]
        for packet in packets:
            if packet.seq not in cache:
                if mode == 'bounded_constant':
                    raise RuntimeError('A forced window unexpectedly created a new map packet')
                cache[packet.seq] = env.packet_scores(sender, [packet], mode)[0] / packet.size
        return max(range(len(packets)), key=lambda i: cache[packets[i].seq])

    while elapsed < seconds - 1e-9:
        dt = min(env.config.dt, seconds - elapsed)
        gains += radio.service(env.positions.copy(), np.zeros((2, 2)), env.time + elapsed,
                               dt, True, beliefs, None if mode == 'fifo' else rank, log_events=False)
        elapsed += dt
    assert radio.sent_bytes - before <= 2 * env.config.bytes_per_second * seconds + 2
    return gains.tolist()


def state_check(env, learner, constant, constant_control=False):
    contexts = learner.contexts(env)
    env.feature_provider = lambda sender, indices: contexts[sender][indices]
    scores = {mode: {} for mode in ['rule', 'learned']}
    if constant_control:
        scores['bounded_constant'] = {}
    labels = []
    for sender in range(2):
        packets = env.radio.pending(sender)
        predicted = env.packet_scores(sender, packets, 'learned')
        rule = env.packet_scores(sender, packets, 'rule')
        nominal = env.packet_scores(sender, packets, 'fifo')
        for mode, values in [('rule', rule), ('learned', predicted)]:
            scores[mode][sender] = {p.seq: value / p.size for p, value in zip(packets, values)}
        if constant_control:
            scores['bounded_constant'][sender] = {
                p.seq: min(constant * 100., value) / p.size for p, value in zip(packets, nominal)}
        selection = np.linspace(0, len(packets) - 1, min(64, len(packets))).astype(int)
        target = env.beliefs[1 - sender].ravel()
        for i in selection:
            packet = packets[i]
            y = np.count_nonzero(target[packet.cells()['index']] == 127) / 100.
            labels.append({'sender': sender, 'seq': packet.seq, 'target': float(y),
                           'learned': float(predicted[i] / 100.),
                           'sender_novelty': float(nominal[i] / 100.), 'constant': constant,
                           'packet_cells': len(packet.cells())})
            if constant_control:
                labels[-1]['bounded_constant'] = min(constant, float(nominal[i] / 100.))
    modes = ['fifo', 'rule', 'learned', 'oracle'] + (['bounded_constant'] if constant_control else [])
    return {'time': env.time, 'labels': labels,
            'forced_contact_gains': {mode: window_gain(env, mode, scores)
                                     for mode in modes}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--evaluation-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-every', type=int, default=16)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--content-score-bound', choices=['none', 'sender'])
    parser.add_argument('--training-sampling', type=Path,
                        help='Map-balanced head sampling.json containing the training-only label mean')
    parser.add_argument('--constant-control', action='store_true',
                        help='Also rank packets with the training mean, same local bound and byte cost')
    args = parser.parse_args()
    if Path(sys.prefix) != Path('/home/wyz/anaconda3/envs/ir2_train'):
        parser.error('Use ir2_train Python')
    if args.sample_every < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('Use a positive sample interval and an empty output directory')
    saved = torch.load(args.checkpoint, map_location='cpu')
    if saved['learner']['content_updates'] == 0 or (not saved.get('content_replay') and not args.training_sampling):
        parser.error('Use a trained head and its supervised replay or map-balanced training sampling.json')
    cfg = Config.from_checkpoint(saved['config'])
    if args.content_score_bound is not None:
        cfg.content_score_bound = args.content_score_bound
    torch.set_num_threads(cfg.torch_threads)
    learner = Learner(cfg, args.device)
    learner.load(saved['learner'])
    learner.policy.eval()
    learner.context_encoder.eval()
    constant = (float(json.loads(args.training_sampling.read_text())['dataset_mean_target'])
                if args.training_sampling else float(np.mean([x[2] for x in saved['content_replay']])))
    args.output.mkdir(parents=True, exist_ok=True)
    provenance(args.output, cfg)
    append_json(args.output / 'invocation.jsonl', {'checkpoint': str(args.checkpoint.resolve()),
        'evaluation_dirs': [str(p.resolve()) for p in args.evaluation_dirs],
        'training_sampling': str(args.training_sampling.resolve()) if args.training_sampling else None,
        'constant_control': args.constant_control,
        'sample_every': args.sample_every, 'device': args.device, 'constant_training_mean': constant})
    summaries = []
    for directory in args.evaluation_dirs:
        trace_cfg = Config.from_checkpoint(json.loads((directory / 'config.json').read_text()))
        trace_cfg.content_score_bound = cfg.content_score_bound
        assert trace_cfg.content == 'fifo'
        records = [json.loads(x) for x in (directory / 'evaluation_trajectories.jsonl').read_text().splitlines()]
        for record in records:
            info = record['map']
            env = Environment(trace_cfg, info['split'], info['index'], scorer=learner.score)
            states = []
            for index, expected in enumerate(record['trajectory'][1:]):
                if index % args.sample_every == 0:
                    row = state_check(env, learner, constant, args.constant_control)
                    row.update({'map': info, 'decision': index, 'source': str(directory)})
                    append_json(args.output / 'states.jsonl', row)
                    states.append(row)
                env.step(expected['actions'])
                actual = env.trajectory[-1]
                for key in ['time', 'positions', 'coverage', 'novel_received_cells']:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=0., atol=1e-10)
            labels = [x for row in states for x in row['labels']]
            summary = {'map': info, 'states': len(states), 'labels': len(labels), 'replay_matches': True,
                       'content_updates': learner.content_updates, 'movement_updates': learner.updates,
                       'positive_fraction': float(np.mean([x['target'] > 0 for x in labels])),
                       'mse': {mode: float(np.mean([(x[mode] - x['target']) ** 2 for x in labels]))
                               for mode in ['learned', 'sender_novelty', 'constant'] +
                               (['bounded_constant'] if args.constant_control else [])},
                       'mean_contact_new_cells': {mode: float(np.mean([sum(x['forced_contact_gains'][mode]) for x in states]))
                                                 for mode in ['fifo', 'rule', 'learned', 'oracle'] +
                                                 (['bounded_constant'] if args.constant_control else [])},
                       'bound_violation_fraction': float(np.mean([x['learned'] * 100 > x['packet_cells'] + 1e-5 for x in labels]))}
            summaries.append(summary)
            append_json(args.output / 'maps.jsonl', summary)
            print(json.dumps(summary), flush=True)
    grouped = {}
    for split in ['train', 'validation']:
        group = [x for x in summaries if x['map']['split'] == split]
        if not group:
            continue
        grouped[split] = {'maps': len(group),
            'macro_mse': {mode: float(np.mean([x['mse'][mode] for x in group])) for mode in group[0]['mse']},
            'macro_mean_contact_new_cells': {mode: float(np.mean([x['mean_contact_new_cells'][mode] for x in group])) for mode in group[0]['mean_contact_new_cells']}}
    validation = grouped.get('validation')
    passed = bool(validation and validation['macro_mse']['learned'] <= validation['macro_mse']['constant']
                  and all(x['macro_mean_contact_new_cells']['learned'] > x['macro_mean_contact_new_cells']['fifo'] for x in grouped.values()))
    result = {'groups': grouped, 'gate_passed': passed, 'constant_training_mean': constant,
              'note': 'Frozen hypothetical contacts and offline labels, not deployed task completion.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
