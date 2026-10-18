#!/usr/bin/env python3
"""Fork full finetuning state into a new directory, changing only Actor LR."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import torch
from finite_contact.config import Config
from finite_contact_train import provenance


def fork_checkpoint(source, output, policy_learning_rate):
    if policy_learning_rate <= 0:
        raise ValueError('Actor learning rate must be positive')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Fork output must be an empty directory')
    saved = torch.load(source, map_location='cpu')
    if saved.get('format') != 'ir2-finite-contact-v1' or saved['stage'] != 'finetune' or 'replay' not in saved:
        raise ValueError('Use a full finetune checkpoint with replay, optimizers and RNG')
    cfg = Config.from_checkpoint(saved['config'])
    before = cfg.to_dict()
    cfg.policy_learning_rate = policy_learning_rate
    cfg.validate()
    saved['config'] = cfg.to_dict()
    for group in saved['learner']['policy_optimizer']['param_groups']:
        group['lr'] = policy_learning_rate
    saved['fork_source_checkpoint'] = str(source.resolve())
    saved['fork_overrides'] = {'policy_learning_rate': policy_learning_rate}
    # Separate experiment wall time; counters, episodes, Adam moments, all replay
    # and every RNG stream retain the source state. No rollout is synthesized.
    saved['training_wall_seconds'] = 0.
    output.mkdir(parents=True, exist_ok=True)
    provenance(output, cfg)
    checkpoint = output / 'fork_source.pth'
    torch.save(saved, checkpoint)
    record = {'source': str(source.resolve()), 'checkpoint': str(checkpoint.resolve()),
              'source_config': before, 'fork_config': cfg.to_dict(),
              'updates': saved['learner']['updates'], 'content_updates': saved['learner']['content_updates'],
              'restored': ['network weights', 'Adam moments', 'replay', 'successful training episodes', 'RNG'],
              'only_training_change': 'Actor learning rate'}
    (output / 'fork.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-learning-rate', type=float, required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/wyz/anaconda3/envs/ir2_train').resolve():
        parser.error('Use ir2_train Python')
    checkpoint = fork_checkpoint(args.checkpoint, args.output, args.policy_learning_rate)
    print(json.dumps({'checkpoint': str(checkpoint), 'next_command_prefix': shlex.join([
        sys.executable, str(Path(__file__).resolve().parent / 'finite_contact_train.py'),
        '--stage', 'finetune', '--resume', str(checkpoint.resolve())])}), flush=True)


if __name__ == '__main__':
    main()
