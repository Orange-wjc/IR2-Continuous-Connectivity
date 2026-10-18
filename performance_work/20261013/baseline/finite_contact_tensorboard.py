#!/usr/bin/env python3
"""Mirror existing training JSONL into TensorBoard without touching the learner."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import time

from torch.utils.tensorboard import SummaryWriter

GROUPS = ('AB_learned', 'A_rule', 'A_fifo')
LOSS_TAGS = {'policy_loss':'loss/policy', 'q1_loss':'loss/q1', 'q2_loss':'loss/q2',
             'sac_policy_loss':'loss/sac_policy', 'imitation_loss':'loss/imitation',
             'entropy':'policy/entropy', 'alpha':'policy/alpha',
             'policy_grad_norm':'policy/grad_norm', 'wait_probability':'policy/wait_probability',
             'pointer_saturation_fraction':'policy/pointer_saturation_fraction'}


class JsonlMirror:
    def __init__(self, source, output, kind):
        self.source, self.output, self.kind = Path(source), Path(output), kind
        self.offset = 0
        self.identity = None
        self.writer = None
        self.last_step = None
        self.window = deque(maxlen=50)

    def reset(self):
        if self.writer:
            self.writer.close()
        self.writer = None
        self.offset = 0
        self.last_step = None
        self.window.clear()

    def poll(self):
        if not self.source.exists():
            return 0
        stat = self.source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.reset()
            self.identity = identity
        count = 0
        with self.source.open('rb') as handle:
            handle.seek(self.offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b'\n'):
                    break  # Retry incomplete appended records on the next poll.
                if not line.strip():
                    self.offset = handle.tell()
                    continue
                row = json.loads(line)
                step = int(row['episode'] if self.kind == 'train' else row['updates'])
                if self.writer is None or (self.last_step is not None and step < self.last_step):
                    if self.writer:
                        self.writer.close()
                    # Rebuilding after a source rewind removes superseded event steps.
                    self.writer = SummaryWriter(str(self.output), purge_step=step, flush_secs=5)
                    self.window.clear()
                self.write(row, step)
                self.last_step = step
                self.offset = handle.tell()
                count += 1
        if self.writer:
            self.writer.flush()
        return count

    def scalar(self, tag, value, step):
        value = float(value)
        if math.isfinite(value):
            self.writer.add_scalar(tag, value, step)

    def write(self, row, step):
        if self.kind == 'sac':
            for key, tag in LOSS_TAGS.items():
                if key in row:
                    self.scalar(tag, row[key], step)
            return
        self.window.append(row)
        for tag, value in {
            'train/success':float(row['success']),
            'train/success_rate_last_50':sum(r['success'] for r in self.window)/len(self.window),
            'train/time_seconds':row['time'],
            'train/time_mean_last_50':sum(r['time'] for r in self.window)/len(self.window),
            'train/min_robot_coverage':min(row['coverage']),
            'train/union_coverage':row['union_coverage'],
            'train/coverage_deficit':row['coverage_deficit'],
            'train/return_mean':sum(row['return'])/len(row['return']),
            'train/sent_bytes':row['sent_bytes'],
            'train/learner_updates':row['learner_updates'],
        }.items():
            self.scalar(tag, value, step)

    def close(self):
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--interval', type=float, default=5.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('Positive polling interval required')
    root = args.experiment.resolve()
    if not (root/'pipeline.json').exists():
        parser.error('Specify the existing experiment directory')
    output = args.output.resolve() if args.output else root/'tensorboard_live'
    mirrors = [JsonlMirror(root/group/file, output/group/kind, kind)
               for group in GROUPS for file,kind in (('training.jsonl','train'),('updates.jsonl','sac'))]
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps({'scope':'Read-only JSONL mirror; training episodes and cumulative SAC updates use separate axes.',
                      'experiment':str(root), 'tensorboard_logdir':str(output)}), flush=True)
    try:
        while not stopped[0]:
            counts = [mirror.poll() for mirror in mirrors]
            if sum(counts):
                print(json.dumps({'new_mirrored_records':sum(counts)}), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        for mirror in mirrors:
            mirror.close()


if __name__ == '__main__':
    main()
