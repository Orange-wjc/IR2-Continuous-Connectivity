"""Fixed-policy, full training-map 12/16 worker comparison; zero SAC updates."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/ir2-native-mpl'
import gc
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import threading
import time
import torch

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
sys.path.insert(0, str(WORK / 'source'))
from finite_contact.config import Config
from finite_contact.memory import load_checkpoint, semantic_digest
from native_worker import initialize_worker, run_episode


def memory():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, rest = line.split(':', 1)
        info[key] = int(rest.split()[0]) / 1024
    return info


def swaps():
    info = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {key: int(info[key]) for key in ('pswpin', 'pswpout')}


def rss(pid):
    try:
        for line in Path('/proc/%d/status' % pid).read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0


def main():
    plan = json.loads((ROOT / 'pipeline.json').read_text())
    assert plan['status'] == 'paused_for_easy_review'
    saved = load_checkpoint(ROOT / 'AB_learned/latest.pth')
    cfg = Config.from_checkpoint(saved['config'])
    checkpoint_episode = saved['episode']
    weights = {key: saved['learner'][key] for key in ('policy', 'context_encoder', 'content')}
    torch.save(weights, WORK / 'fixed_ab_weights.pth')
    del saved, weights
    gc.collect()
    first = max(plan['completed_episodes'].values()) + 1
    timings = []
    reference = None
    for workers in (12, 16):
        jobs = [dict(config=cfg.to_dict(), weights=str(WORK / 'fixed_ab_weights.pth'),
                     episode=first + i, map_index=(first + i - 1) % cfg.train_maps,
                     output=str(WORK / 'probe' / ('workers_%d' % workers) / 'episodes')) for i in range(48)]
        before_swap = swaps()
        samples = []
        stopping = threading.Event()
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker) as pool:
            futures = [pool.submit(run_episode, job) for job in jobs]
            def monitor():
                while not stopping.is_set():
                    available = memory()['MemAvailable']
                    pids = [p.pid for p in list(pool._processes.values())]
                    samples.append(dict(available_mib=available, workers_rss_mib=sum(rss(p) for p in pids),
                                        parent_rss_mib=rss(os.getpid())))
                    stopping.wait(.5)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                episodes = [future.result() for future in futures]
            finally:
                stopping.set()
                thread.join()
        wall = time.monotonic() - start
        hashes = [semantic_digest({'transitions': item['transitions']}) for item in episodes]
        if reference is None:
            reference = hashes
        assert hashes == reference, 'Parallelism changed fixed-policy experiences'
        after_swap = swaps()
        row = dict(workers=workers, full_training_rollouts=48, wall_seconds=wall,
                   rollouts_per_minute=48 * 60 / wall,
                   min_available_memory_mib=min(x['available_mib'] for x in samples),
                   peak_aggregate_worker_rss_mib=max(x['workers_rss_mib'] for x in samples),
                   peak_parent_rss_mib=max(x['parent_rss_mib'] for x in samples),
                   peak_single_worker_rss_mib=max(x['worker_rss_mib'] for x in episodes),
                   swap_in_pages=after_swap['pswpin'] - before_swap['pswpin'],
                   swap_out_pages=after_swap['pswpout'] - before_swap['pswpout'],
                   hashes=hashes, failures_are_real_outcomes=True,
                   geometry_wall_crossings=sum(x['audit']['wall_crossings'] for x in episodes),
                   radio_capacity_audits_passed=all(x['audit']['radio_totals_and_capacity_passed'] for x in episodes))
        timings.append(row)
        (WORK / 'probe_timings.json').write_text(json.dumps(timings, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'hashes'}), flush=True)
        del episodes, futures
        gc.collect()
    speedup = timings[0]['wall_seconds'] / timings[1]['wall_seconds']
    safe = (timings[1]['min_available_memory_mib'] > 8192
            and timings[1]['swap_out_pages'] == 0
            and all(x['geometry_wall_crossings'] == 0 and x['radio_capacity_audits_passed'] for x in timings))
    review = dict(status='passed', speedup_16_over_12=speedup,
                  wall_time_reduction_percent=100 * (1 - 1 / speedup),
                  memory_safe=safe, faster=speedup > 1.03, recommended_workers=16 if safe and speedup > 1.03 else 12,
                  source_checkpoint_episode=checkpoint_episode, first_episode=first, timings=timings,
                  replay_hashes_identical=True, engineering_train_rollouts=96,
                  engineering_sac_updates=0, development_policy_executions=0, test_policy_executions=0,
                  scope='48 identical native training-map jobs per pool; includes worker startup, no GPU learning. Not a complete training throughput promise.')
    (WORK / 'capacity_review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({k: v for k, v in review.items() if k != 'timings'}), flush=True)


if __name__ == '__main__':
    main()
