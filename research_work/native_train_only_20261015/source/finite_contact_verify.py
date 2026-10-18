#!/usr/bin/env python3
"""Read-only single-map training acceptance, with a predeclared action-noise seed."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.learning import Learner
from finite_contact_train import evaluate, append_json, provenance


def fingerprint(learner):
    digest = hashlib.sha256()
    state = learner.state()
    for name in ('policy', 'critics', 'targets', 'context_encoder', 'content'):
        networks = state[name] if isinstance(state[name], list) else [state[name]]
        for network in networks:
            for key, tensor in sorted(network.items()):
                digest.update(key.encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
    digest.update(learner.log_alpha.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--sample-seed', type=int, default=101)
    parser.add_argument('--validation-maps', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--max-hours', type=float, default=.3)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/wyz/anaconda3/envs/ir2_train').resolve():
        parser.error('Use ir2_train Python')
    if args.samples < 10 or args.validation_maps < 1 or args.max_hours <= 0:
        parser.error('Use at least ten samples, a positive validation count and wall budget')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use an empty output directory')
    saved = torch.load(args.checkpoint, map_location='cpu')
    cfg = Config.from_checkpoint(saved['config'])
    cfg.validate()
    if args.validation_maps > cfg.validation_maps:
        parser.error('Too many validation maps')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    torch.set_num_threads(cfg.torch_threads)
    seed_everything(cfg.seed)
    learner = Learner(cfg, args.device)
    learner.load(saved['learner'])
    before = fingerprint(learner)
    counters = learner.updates, learner.content_updates, learner.imitation_updates
    args.output.mkdir(parents=True, exist_ok=True)
    provenance(args.output, cfg)
    deadline = time.monotonic() + args.max_hours * 3600
    groups = {}
    for label, split, count, policy, repeats in (
            ('train_greedy', 'train', cfg.train_maps, 'greedy', 1),
            ('train_sampled', 'train', cfg.train_maps, 'sampled', args.samples),
            ('validation', 'validation', args.validation_maps, 'greedy', 1)):
        directory = args.output / label
        directory.mkdir()
        if policy == 'sampled':
            seed_everything(args.sample_seed)
        result = evaluate(learner, cfg, count, split, deadline, directory, policy, repeats)
        for row in result:
            append_json(directory / 'evaluation.jsonl', row)
            assert row['delivered_bytes'] <= row['sent_bytes']
            assert row['control_bytes'] <= row['sent_bytes']
        groups[label] = {'requested': count * repeats, 'completed': len(result),
                         'successes': sum(x['success'] for x in result),
                         'times': [x['time'] for x in result],
                         'minimum_coverage': [min(x['coverage']) for x in result]}
        print(json.dumps({'phase': label, **groups[label]}), flush=True)
    assert before == fingerprint(learner), 'Evaluation changed network weights'
    assert counters == (learner.updates, learner.content_updates, learner.imitation_updates)
    complete = all(x['completed'] == x['requested'] for x in groups.values())
    greedy, sampled = groups['train_greedy'], groups['train_sampled']
    accepted = (complete and greedy['successes'] == greedy['requested'] and
                sampled['successes'] / sampled['requested'] >= .9)
    summary = {'checkpoint': str(args.checkpoint.resolve()), 'config': cfg.to_dict(),
               'sample_seed': args.sample_seed, 'weights_unchanged': True,
               'learner_updates': learner.updates, 'content_updates': learner.content_updates,
               'imitation_updates': learner.imitation_updates, 'complete': complete,
               'accepted_for_single_map_training': accepted, 'groups': groups,
               'scope': 'Same trained model; repeated action samples are not independent training seeds. Test split unused.'}
    (args.output / 'acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({'accepted_for_single_map_training': accepted, 'complete': complete}), flush=True)


if __name__ == '__main__':
    main()
