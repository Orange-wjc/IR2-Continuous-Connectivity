#!/usr/bin/env python3
"""Run the bounded two-robot single-map development recipe in ir2_train."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory')
    parser.add_argument('--profile', choices=['actor_decay', 'union', 'local'], default='actor_decay',
                        help='actor_decay: lower Actor LR after 3000; union/local reproduce diagnostic conditions')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without writing or training')
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/wyz/anaconda3/envs/ir2_train').resolve():
        parser.error('Use /home/wyz/anaconda3/envs/ir2_train/bin/python')
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output must be empty; existing experiments are never overwritten')
    movement, content, joint = (output / name for name in ('movement_A', 'content', 'finetune_AB'))
    trainer = [sys.executable, str(ROOT / 'finite_contact_train.py')]
    validation = ['--eval-every', '5', '--eval-maps', '2', '--eval-train', '--final-eval', '--device', 'cuda']
    targets = ([2000, 2500] if args.profile == 'local' else
               [2000, 2500, 3000] if args.profile == 'actor_decay' else [2000, 2500, 3000, 3500])
    seeds = ({2000: 101, 2500: 202} if args.profile == 'local' else
             {3000: 303} if args.profile == 'actor_decay' else {3000: 303, 3500: 404})
    jobs = [
        ('movement_1500', trainer + ['--stage', 'movement', '--variant', 'A', '--content', 'fifo',
          '--seed', str(args.seed), '--train-maps', '1', '--coverage-reward',
          'local_mean' if args.profile == 'local' else 'union', '--success-imitation-weight', '1',
          '--horizon-target', 'terminal', '--pointer-mode', 'centered',
          '--alpha-learning-rate', '.0001', '--frontier-shaping-scale', '0',
          '--max-updates', '1500', '--episodes', '100', '--max-hours', '.4',
          '--eval-every', '10', '--eval-maps', '2', '--eval-train', '--final-eval',
          '--device', 'cuda', '--run-dir', str(movement)], movement, 1500, 0),
        ('content_300', trainer + ['--stage', 'content', '--initialize-from', str(movement / 'final.pth'),
          '--max-content-updates', '300', '--episodes', '30', '--max-hours', '.3',
          '--eval-every', '100', '--device', 'cuda', '--run-dir', str(content)], content, 1500, 300),
    ]
    for i, target in enumerate(targets):
        start = (['--variant', 'AB', '--content', 'learned', '--initialize-from', str(content / 'final.pth'),
                  '--run-dir', str(joint)] if i == 0 else ['--resume', str(joint / 'latest.pth')])
        jobs.append((f'finetune_{target}', trainer + ['--stage', 'finetune'] + start +
                     ['--max-updates', str(target), '--episodes', '100', '--max-hours', '.3'] + validation,
                     joint, target, 300))
        if target in seeds:
            jobs.append((f'verify_{target}', [sys.executable, str(ROOT / 'finite_contact_verify.py'),
                         '--checkpoint', str(joint / f'at_{target}.pth'), '--output', str(output / f'verify_{target}'),
                         '--sample-seed', str(seeds[target]), '--samples', '10', '--device', 'cuda'], None, target, 300))
    if args.profile == 'actor_decay':
        small_step = output / 'actor_decay_AB'
        jobs.append(('fork_actor_rate', [sys.executable, str(ROOT / 'finite_contact_fork.py'),
                     '--checkpoint', str(joint / 'at_3000_latest.pth'), '--output', str(small_step),
                     '--policy-learning-rate', '.00001'], small_step, 3000, 300))
        for target, seed in [(3500, 505), (4000, 606)]:
            source = small_step / ('fork_source.pth' if target == 3500 else 'latest.pth')
            jobs.append((f'finetune_{target}', trainer + ['--stage', 'finetune', '--resume', str(source),
                         '--max-updates', str(target), '--episodes', '100', '--max-hours', '.3'] + validation,
                         small_step, target, 300))
            jobs.append((f'verify_{target}', [sys.executable, str(ROOT / 'finite_contact_verify.py'),
                         '--checkpoint', str(small_step / f'at_{target}.pth'), '--output', str(output / f'verify_{target}'),
                         '--sample-seed', str(seed), '--samples', '10', '--device', 'cuda'], None, target, 300))
    if args.dry_run:
        for name, command, *_ in jobs:
            print(name + ': ' + shlex.join(command))
        return
    output.mkdir(parents=True, exist_ok=True)
    records = []
    def save(status):
        (output / 'pipeline.json').write_text(json.dumps({'profile': args.profile, 'seed': args.seed,
            'status': status, 'scope': 'Two robots, one training map; test split unused.',
            'steps': records}, ensure_ascii=False, indent=2))
    save('running')
    try:
        for name, command, directory, updates, head in jobs:
            print(f'[{name}] {shlex.join(command)}', flush=True)
            record = {'step': name, 'command': command, 'status': 'running'}
            records.append(record)
            save('running')
            with (output / (name + '.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        brief = {key: row[key] for key in ('episode', 'learner_updates', 'success', 'time',
                                 'phase', 'successes', 'completed', 'accepted_for_single_map_training') if key in row}
                        if brief:
                            print(json.dumps({'step': name, **brief}), flush=True)
                    code = child.wait()
                except KeyboardInterrupt:
                    child.send_signal(signal.SIGTERM)
                    child.wait()
                    raise
            if code:
                raise RuntimeError(f'{name} failed (exit {code}); see {output / (name + ".log")}')
            if directory is not None:
                filename = 'fork_source.pth' if name == 'fork_actor_rate' else 'final.pth'
                saved = torch.load(directory / filename, map_location='cpu')
                assert saved['learner']['updates'] == updates, f'{name}: update budget not reached'
                assert saved['learner']['content_updates'] == head, f'{name}: content budget not reached'
                if name.startswith('finetune'):
                    for source, target in [('final.pth', f'at_{updates}.pth'), ('latest.pth', f'at_{updates}_latest.pth')]:
                        shutil.copy2(directory / source, directory / target)
            else:
                summary = json.loads((output / name / 'acceptance.json').read_text())
                record['accepted_for_single_map_training'] = summary['accepted_for_single_map_training']
                assert summary['complete'], f'{name}: evaluation wall budget exhausted'
            record['status'] = 'complete'
            save('running')
        accepted = all(record['accepted_for_single_map_training'] for record in records if record['step'].startswith('verify'))
        save('accepted_for_single_map_training' if accepted else 'training_complete_acceptance_failed')
        print(json.dumps({'output': str(output), 'accepted_for_single_map_training': accepted}), flush=True)
        if not accepted:
            raise SystemExit(2)
    except (Exception, KeyboardInterrupt):
        save('interrupted_or_failed')
        raise


if __name__ == '__main__':
    main()
