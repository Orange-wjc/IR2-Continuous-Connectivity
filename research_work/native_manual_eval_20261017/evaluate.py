"""Explicit frozen-model development evaluation; zero learning/test execution."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/ir2-native-mpl'
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import threading
import time

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
WATCH = Path('/home/wyz/wjc_ws/training_supervisor_native_easy')
GROUPS = {'IR2_FIFO': ('B0', 'fifo'), 'AB_learned': ('AB', 'learned')}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def memory_sample():
    info = {line.split(':', 1)[0]: int(line.split(':', 1)[1].split()[0])
            for line in Path('/proc/meminfo').read_text().splitlines()}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return dict(time=now(), available_memory_gib=info['MemAvailable'] / 1048576,
                used_memory_gib=(info['MemTotal'] - info['MemAvailable']) / 1048576,
                swap_in_pages=int(vm['pswpin']), swap_out_pages=int(vm['pswpout']))


def main():
    # The old coordinator may still be waiting for prefetched CPU-only work.
    pid = read(ROOT / 'coordinator.json')['pid']
    deadline = time.monotonic() + 900
    while (Path('/proc') / str(pid)).exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Old coordinator still alive; evaluation not started')
        time.sleep(2)
    assert (WATCH / 'PAUSE').exists() and (ROOT / 'STOP').exists()
    with (ROOT / '.coordinator.lock').open('a') as local, (ROOT.parent / '.native_training.lock').open('a') as global_lock:
        for handle in (local, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(ROOT / 'pipeline.json')
        assert plan['status'] == 'paused_for_easy_review'
        assert len(set(plan['completed_episodes'].values())) == 1
        assert len(set(plan['sac_updates'].values())) == 1
        assert plan['automatic_development_evaluation'] is False
        assert plan['test_policy_evaluations'] == 0 and not plan['stage2_started']
        episode = min(plan['completed_episodes'].values())
        out = ROOT / 'manual_evaluations' / ('20261017_ep%08d' % episode)
        out.mkdir(parents=True)
        (out / 'checkpoints').mkdir()
        write(out / 'before_pipeline.json', plan)
        manifest = ROOT / 'dataset/native_dataset.json'
        assert sha(manifest) == plan['dataset_sha256']
        data = read(manifest)
        validation = data['families']['easy']['splits']['validation']
        assert len(validation) == 100
        train_geometry = {x['geometry_group'] for x in data['families']['easy']['splits']['train']}
        assert not train_geometry.intersection(x['geometry_group'] for x in validation)
        source = WORK / 'source'
        source.mkdir()
        for name, checksum in plan['source_identity'].items():
            assert sha(ROOT / 'source' / name) == checksum
            dest = source / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'source' / name, dest)
            assert sha(dest) == checksum
        sys.path.insert(0, str(source))
        import torch
        import numpy as np
        from finite_contact.config import Config
        from finite_contact.memory import load_checkpoint, semantic_digest
        from native_worker import initialize_worker, run_episode

        checksums = {str(manifest.relative_to(ROOT)): sha(manifest)}
        checkpoint_review = {}
        jobs = {}
        config_records = {}
        for group, method in GROUPS.items():
            original = ROOT / group / 'latest.pth'
            frozen = out / 'checkpoints' / (group + '.pth')
            shutil.copy2(original, frozen)
            checksums[str(original.relative_to(ROOT))] = sha(original)
            assert sha(frozen) == checksums[str(original.relative_to(ROOT))]
            saved = load_checkpoint(frozen)
            assert saved['episode'] == episode and saved['role'] == group
            assert saved['learner']['updates'] == plan['sac_updates'][group]
            cfg = Config.from_checkpoint(saved['config'])
            assert (cfg.variant, cfg.content) == method
            assert cfg.validation_maps == 100 and cfg.train_maps == 9899
            assert cfg.robots == 2 and cfg.max_decisions == 196 and cfg.horizon_seconds == 3000
            assert cfg.dataset_manifest == str(manifest.resolve())
            directory = ROOT / group / ('validation_%08d' % episode)
            directory.mkdir()
            weights = {key: saved['learner'][key] for key in ('policy', 'context_encoder', 'content')}
            weight_file = directory / 'worker_weights.pth'
            torch.save(weights, weight_file)
            checkpoint_review[group] = dict(episode=episode, sac_updates=saved['learner']['updates'],
                checkpoint_sha256=sha(frozen), rollout_weights_sha256=sha(weight_file),
                model_semantic_digest=semantic_digest(weights), checkpoint_time=saved['time'])
            jobs[group] = [dict(config=cfg.to_dict(), weights=str(weight_file), episode=i + 1,
                map_index=i, split='validation', greedy=True, evaluation=True,
                output=str(directory / 'episodes')) for i in range(100)]
            config_records[group] = cfg.to_dict()
            del saved, weights
        metadata = dict(status='running', started=now(), pid=os.getpid(), episode=episode,
            groups=list(GROUPS), maps_per_group=100, workers=plan['workers'],
            checkpoint_review=checkpoint_review, source_identity=plan['source_identity'],
            dataset_sha256=plan['dataset_sha256'], configs=config_records,
            action_seeding='Same seed_episode(config.seed, map_index+1) as original evaluate_pair',
            greedy=True, development_policy_budget=200, training_policy_budget=0,
            sac_update_budget=0, test_policy_budget=0,
            task='Both robots local free-cell coverage >=99%; native original maps, unchanged time/decision/radio constraints')
        write(out / 'protocol.json', metadata)
        write(WORK / 'evaluation_location.json', dict(output=str(out), episode=episode, pid=os.getpid()))
        progress = dict(status='running', started=now(), episode=episode, pid=os.getpid(),
                        completed={g: 0 for g in GROUPS}, total=200, updated=now())
        write(out / 'status.json', progress)
        stopping = threading.Event()
        samples = []

        def monitor():
            while not stopping.is_set():
                row = memory_sample()
                samples.append(row)
                with (out / 'runtime_memory.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                stopping.wait(2)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        started = time.monotonic()
        results = {g: {} for g in GROUPS}
        try:
            with ProcessPoolExecutor(max_workers=plan['workers'],
                    mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker) as pool:
                # Interleave methods, keep the same logical seeds and fixed weights.
                pending = {pool.submit(run_episode, jobs[g][i]): (g, i)
                           for i in range(100) for g in GROUPS}
                for future in as_completed(pending):
                    group, index = pending[future]
                    item = future.result()
                    metric, audit = item['metric'], item['audit']
                    assert not item['transitions'] and not item['labels']
                    assert audit['wall_crossings'] == 0 and audit['radio_totals_and_capacity_passed']
                    assert metric['map']['split'] == 'validation' and metric['map']['index'] == index
                    assert metric['map']['file_sha256'] == validation[index]['file_sha256']
                    assert metric['success'] == bool(min(metric['coverage']) >= .99)
                    assert index not in results[group]
                    results[group][index] = dict(metric=metric, audit=audit)
                    with (ROOT / group / ('validation_%08d' % episode) / 'metrics.jsonl').open('a') as log:
                        log.write(json.dumps(dict(map_index=index, **item['metric']), allow_nan=False) + '\n')
                    progress['completed'][group] += 1
                    progress['updated'] = now()
                    write(out / 'status.json', progress)
                    print(json.dumps(dict(group=group, map_index=index, success=metric['success'],
                        completed=progress['completed'], time=progress['updated'])), flush=True)
        except BaseException as exc:
            progress.update(status='failed', error=repr(exc), updated=now())
            write(out / 'status.json', progress)
            raise
        finally:
            stopping.set()
            thread.join()
        for group in GROUPS:
            assert set(results[group]) == set(range(100))
            for index in range(100):
                assert results[group][index]['metric']['map']['sha256'] == results['IR2_FIFO'][index]['metric']['map']['sha256']
        for path, checksum in checksums.items():
            assert sha(ROOT / path) == checksum, 'Evaluation changed training state'
        current = read(ROOT / 'pipeline.json')
        assert current == plan
        for name, checksum in plan['source_identity'].items():
            assert sha(ROOT / 'source' / name) == checksum
        assert (WATCH / 'PAUSE').exists() and (ROOT / 'STOP').exists()
        both = [i for i in range(100) if all(results[g][i]['metric']['success'] for g in GROUPS)]
        summary = {}
        for group in GROUPS:
            rows = [results[group][i]['metric'] for i in range(100)]
            value = dict(successes=sum(x['success'] for x in rows), maps=100,
                mean_capped_time=float(np.mean([x['capped_time'] for x in rows])),
                mean_actual_time=float(np.mean([x['time'] for x in rows])),
                mean_union_coverage=float(np.mean([x['union_coverage'] for x in rows])),
                mean_min_local_coverage=float(np.mean([min(x['coverage']) for x in rows])),
                both_success_mean_time=float(np.mean([rows[i]['time'] for i in both])) if both else None,
                both_success_mean_distance=float(np.mean([rows[i]['total_distance'] for i in both])) if both else None,
                both_success_mean_sent_bytes=float(np.mean([rows[i]['sent_bytes'] for i in both])) if both else None,
                failed_decision_limit=sum(not x['success'] and x['stop_reason']=='decision_limit' for x in rows),
                failed_union_already_99=sum(not x['success'] and x['union_coverage']>=.99 for x in rows),
                sac_updates=plan['sac_updates'][group])
            summary[group] = value
            write(ROOT / group / ('validation_%08d' % episode) / 'summary.json', value)
        baseline, candidate = [summary[g] for g in GROUPS]
        report = dict(time=now(), easy_episodes_per_method=episode, groups=summary,
            both_success_maps=len(both), paired_map_hashes_verified=100,
            AB_only_success=sum(results['AB_learned'][i]['metric']['success'] and not results['IR2_FIFO'][i]['metric']['success'] for i in range(100)),
            FIFO_only_success=sum(results['IR2_FIFO'][i]['metric']['success'] and not results['AB_learned'][i]['metric']['success'] for i in range(100)),
            capped_time_reduction_percent=100*(1-candidate['mean_capped_time']/baseline['mean_capped_time']),
            both_success_time_increase_percent=100*(candidate['both_success_mean_time']/baseline['both_success_mean_time']-1) if both else None,
            total_development_policy_executions=200, new_sac_updates=0, training_policy_executions=0,
            test_policy_evaluations=0, wall_seconds=time.monotonic()-started,
            min_available_memory_gib=min(x['available_memory_gib'] for x in samples),
            max_used_memory_gib=max(x['used_memory_gib'] for x in samples),
            source_files_verified=len(plan['source_identity']),
            unchanged_training_checksums=checksums, checkpoint_review=checkpoint_review,
            scope='Fixed geometry-group-held-out native easy development maps; greedy frozen models, one seed',
            geometry_wall_crossings=sum(x['audit']['wall_crossings'] for v in results.values() for x in v.values()),
            radio_capacity_audits_passed=True,
            training_remains_paused=True)
        write(out / 'results.json', report)
        write(ROOT / 'latest_comparison.json', report)
        current.update(last_review_episode=episode, last_manual_review=report['time'], updated=now())
        current.setdefault('control_changes', []).append(dict(time=now(), action='User requested pause and explicit development evaluation',
            episode=episode, development_policy_executions=200, training_policy_executions=0,
            sac_updates=0, test_policy_executions=0, output=str(out), training_remains_paused=True))
        write(ROOT / 'pipeline.json', current)
        progress.update(status='complete', updated=now(), wall_seconds=report['wall_seconds'])
        write(out / 'status.json', progress)
        print('COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
