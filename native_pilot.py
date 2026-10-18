"""Freeze maintained native_training sources, then execute that snapshot.

No Torch/Ray import or training occurs in this bootstrap. Resumes execute the
run's own source, so repository edits cannot hot-patch a running experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parent / 'native_training'


def main(default_num_meta_agent=24):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--output', type=Path)
    probe.add_argument('--resume', action='store_true')
    args, _ = probe.parse_known_args()
    if '--help' in sys.argv or '-h' in sys.argv:
        os.execv(sys.executable, [sys.executable, str(SOURCE/'driver.py'), *sys.argv[1:]])
    if args.output is None:
        probe.error('--output is required')
    root = args.output.resolve()
    forwarded = sys.argv[1:]
    if args.resume and not (root/'bootstrap_identity.json').exists():
        raise ValueError('This entry resumes repair pilots only; historical runs retain their own --finite-contact entry')
    if not args.resume:
        if root.exists() and any(root.iterdir()):
            raise ValueError('A fresh output directory is required; use --resume for existing runs')
        root.mkdir(parents=True, exist_ok=True)
        snapshot = root/'source'
        checksums = {}
        for p in sorted(SOURCE.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.py', '.json', '.md'):
                continue
            name = str(p.relative_to(SOURCE)); dest = snapshot/name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
            checksums[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        for name in ('driver.py', 'native_pilot.py'):
            dest = root/'entry_provenance'/name
            dest.parent.mkdir(exist_ok=True); shutil.copy2(SOURCE.parent/name, dest)
        (root/'bootstrap_identity.json').write_text(json.dumps(checksums, indent=2)+'\n')
        forwarded += ['--bootstrap-prepared']
        if not any(a in ('--workers', '--num-meta-agent') or a.startswith(('--workers=', '--num-meta-agent=')) for a in forwarded):
            forwarded += ['--num-meta-agent', str(default_num_meta_agent)]
    target = root/'source/driver.py'
    if not target.exists():
        raise ValueError('Frozen native training source is missing')
    os.execv(sys.executable, [sys.executable, str(target), *forwarded])
