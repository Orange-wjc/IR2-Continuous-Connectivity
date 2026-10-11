"""Bounded full-resume memory smoke test; preserves checkpoints and run logs."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import torch
from ray._private.utils import get_system_memory, get_used_memory


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    model = root / 'model/wall_aware_hybrid3_recovery_v3_5_a'
    checkpoint = model / 'checkpoint.pth'
    initial = torch.load(checkpoint, map_location='cpu', weights_only=True)
    start_episode = initial['episode']
    del initial
    backup = Path(tempfile.mkdtemp(prefix='backup_before_bool_replay_', dir=model))
    shutil.copy2(checkpoint, backup / 'checkpoint.pth')
    run_dir = Path(tempfile.mkdtemp(
        prefix='bool_replay_smoke_', dir=root / 'train/wall_aware_hybrid3_recovery_v3_5_a'))
    target = ((start_episode + 192 + 31) // 32) * 32
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(MPLBACKEND='Agg', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONNOUSERSITE='1')
    code = ('import runpy; import parameter; '
            'parameter.LOAD_MODEL=True; parameter.LOAD_POLICY_ONLY=False; '
            'parameter.CONTINUE_LOG_ALPHA=True; parameter.NUM_META_AGENT=8; '
            'runpy.run_path("driver.py",run_name="__main__")')
    start_time = time.monotonic()
    peak = 0.0
    last_episode = start_episode
    last_mtime = checkpoint.stat().st_mtime_ns
    high_samples = 0
    reason = 'unknown'
    with (run_dir / 'console.log').open('xb') as output:
        process = subprocess.Popen([sys.executable, '-s', '-B', '-u', '-c', code],
                                   env=env, stdout=output, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps({'run_dir': str(run_dir), 'training_pid': process.pid,
                      'start_episode': start_episode, 'target_episode': target,
                      'backup': str(backup)}), flush=True)
    (run_dir / 'run.json').write_text(json.dumps({
        'training_pid': process.pid, 'start_episode': start_episode,
        'target_episode': target, 'backup': str(backup)}, indent=2))
    try:
        with (run_dir / 'memory.jsonl').open('x') as metrics:
            while True:
                mem = dict((line.split(':')[0], int(line.split()[1]))
                           for line in Path('/proc/meminfo').read_text().splitlines())
                vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
                fraction = get_used_memory() / get_system_memory()
                peak = max(peak, fraction)
                try:
                    mtime = checkpoint.stat().st_mtime_ns
                    if mtime != last_mtime:
                        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                        last_episode = saved['episode']
                        del saved
                        last_mtime = mtime
                except (OSError, RuntimeError, EOFError):
                    pass  # A checkpoint may be in the middle of being written.
                log = (run_dir / 'console.log').read_text(errors='replace')
                full = 'Replay at capacity: 10000 transitions' in log
                row = {'time': time.strftime('%H:%M:%S'),
                       'elapsed_s': round(time.monotonic() - start_time),
                       'ray_used_pct': round(fraction * 100, 2),
                       'available_GiB': round(mem['MemAvailable'] / 1048576, 2),
                       'swap_out_pages': int(vm['pswpout']),
                       'swap_in_pages': int(vm['pswpin']),
                       'checkpoint_episode': last_episode, 'replay_full': full,
                       'training_updates_logged': log.count('\ntraining\n'),
                       'oom': 'OutOfMemoryError' in log or 'CUDA out of memory' in log}
                metrics.write(json.dumps(row) + '\n')
                metrics.flush()
                print(json.dumps(row), flush=True)
                if process.poll() is not None:
                    reason = 'training_exited_{}'.format(process.returncode)
                    break
                high_samples = high_samples + 1 if fraction >= 0.92 else 0
                if high_samples >= 2 or fraction >= 0.94:
                    reason = 'memory_pressure_guard'
                    break
                if full and last_episode >= target:
                    reason = 'completed_target_with_full_replay'
                    break
                if time.monotonic() - start_time > 40 * 60:
                    reason = 'time_limit'
                    break
                time.sleep(15)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                reason += '_cleanup_timeout'
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        result = {'reason': reason, 'start_episode': start_episode,
                  'last_saved_episode': last_episode, 'peak_ray_used_pct': round(peak * 100, 2),
                  'elapsed_s': round(time.monotonic() - start_time),
                  'exit_code': process.returncode}
        (run_dir / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
