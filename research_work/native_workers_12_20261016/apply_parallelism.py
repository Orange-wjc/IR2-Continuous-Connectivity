"""Install the tested parameter interface and selected 12-worker configuration."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone, timedelta

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
WATCH = Path('/home/wyz/wjc_ws/training_supervisor_native_easy')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    capacity = json.loads((WORK / 'capacity_review.json').read_text())
    assert capacity['recommended_workers'] == 12 and capacity['memory_safe'] and capacity['faster']
    assert capacity['replay_hashes_identical']
    for log in ('parameter_tests.log', 'supervisor_tests.log'):
        assert '\nOK\n' in (WORK / log).read_text(), log
    assert (WATCH / 'PAUSE').exists() and (ROOT / 'STOP').exists()
    assert not (WATCH / 'active_child.json').exists()
    pid = json.loads((ROOT / 'coordinator.json').read_text())['pid']
    assert not (Path('/proc') / str(pid)).exists(), 'Old coordinator still alive'
    with (ROOT / '.coordinator.lock').open('a') as local, (ROOT.parent / '.native_training.lock').open('a') as global_lock:
        for handle in (local, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((ROOT / 'pipeline.json').read_text())
        assert plan['status'] == 'paused_for_easy_review' and plan['workers'] == 8
        assert plan['automatic_development_evaluation'] is False
        before = plan['source_identity']
        after = json.loads((WORK / 'staged_identity.json').read_text())
        assert set(before) == set(after)
        changed = [key for key in before if before[key] != after[key]]
        assert changed == ['driver.py', 'native_driver.py']
        for key, checksum in before.items():
            assert sha(ROOT / 'source' / key) == checksum, key
        for key, checksum in after.items():
            assert sha(WORK / 'source' / key) == checksum, key
        evidence = ROOT / 'checks/parallelism_12_20261016'
        evidence.mkdir()
        archive = ROOT / 'source_versions/pre_meta_agent_12_20261016'
        shutil.copytree(ROOT / 'source', archive, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('pipeline.json', 'source_identity.json'):
            shutil.copy2(ROOT / name, evidence / ('before_' + name))
        for name in ('capacity_review.json', 'probe_timings.json', 'capacity_probe.log',
                     'parameter_tests.log', 'supervisor_tests.log', 'parameters.patch',
                     'historical_eight_worker_throughput.json'):
            shutil.copy2(WORK / name, evidence / name)
        paths = [ROOT / 'common_start.pth']
        for group, episode in plan['completed_episodes'].items():
            paths.extend([ROOT / group / 'initial.pth', ROOT / group / 'latest.pth'])
            snapshot = ROOT / group / 'checkpoints' / ('episode_%08d.pth' % episode)
            assert not snapshot.exists()
            shutil.copy2(ROOT / group / 'latest.pth', snapshot)
        checksums = {str(path.relative_to(ROOT)): sha(path) for path in paths}
        # The monitor is intentionally paused. Stop it before updating its
        # registered parallelism policy; no training process currently exists.
        subprocess.run(['systemctl', '--user', 'stop', 'ir2-native-easy-supervisor.service'], check=True)
        watch_archive = WATCH / 'versions/pre_meta_agent_12_20261016'
        watch_archive.mkdir(parents=True)
        for name in ('supervisor.py', 'test_supervisor.py', 'REPAIR_INSTRUCTIONS.md', 'deployment.json', 'README.md'):
            shutil.copy2(WATCH / name, watch_archive / name)
        shutil.copy2(WORK / 'supervisor_new.py', WATCH / 'supervisor.py')
        tests = (WORK / 'test_supervisor_new.py').read_text().replace('import supervisor_new as s', 'import supervisor as s')
        (WATCH / 'test_supervisor.py').write_text(tests)
        prompt = (WATCH / 'REPAIR_INSTRUCTIONS.md').read_text()
        assert '8个CPU进程' in prompt
        prompt = prompt.replace('8个CPU进程', '{{WORKERS}}个CPU进程（以本次context.json为准）')
        (WATCH / 'REPAIR_INSTRUCTIONS.md').write_text(prompt)
        for name in changed:
            path = ROOT / 'source' / name
            temporary = path.with_suffix('.py.tmp')
            shutil.copy2(WORK / 'source' / name, temporary)
            temporary.replace(path)
        stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
        event = dict(time=stamp, reason='User requested NUM_META_AGENT parameter and testing 12 parallel workers',
                     previous_workers=8, requested_workers=12, changed_files=changed,
                     old_source_archive=str(archive), saved_episodes=plan['completed_episodes'],
                     checkpoint_hashes_unchanged=checksums, source_before=before, source_after=after,
                     parameter_tests=12, supervisor_tests=19,
                     fixed_policy_cpu_probe_speedup=capacity['speedup_12_over_8'],
                     engineering_rollouts=48, engineering_sac_updates=0,
                     effective_batch=128, microbatch=64, updates_per_episode=8,
                     note='Pool/wave width changes weight lag in both methods; fixed-weight rollout hashes match, entire training trajectories are not claimed identical.')
        plan['source_identity'] = after
        plan.setdefault('control_changes', []).append(event)
        plan['updated'] = stamp
        write(ROOT / 'source_identity.json', after)
        write(ROOT / 'pipeline.json', plan)
        for relative, checksum in checksums.items():
            assert sha(ROOT / relative) == checksum
        write(evidence / 'switch_review.json', event)
        (ROOT / 'STOP').unlink()
        command = ['/home/wyz/anaconda3/envs/ir2_train/bin/python', str(ROOT / 'source/driver.py'),
                   '--finite-contact', '--output', str(ROOT), '--resume',
                   '--num-meta-agent', '12', '--no-auto-evaluation']
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MPLCONFIGDIR='/tmp/ir2-native-mpl', PYTHONUNBUFFERED='1')
        with (ROOT / 'training.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        launch = dict(time=stamp, pid=process.pid, command=command,
                      mode='12 map-sampling workers; unchanged effective batch/SAC budget; manual evaluation only')
        write(ROOT / 'checks/current_launch.json', launch)
        write(evidence / 'launch.json', launch)
        subprocess.run(['systemctl', '--user', 'start', 'ir2-native-easy-supervisor.service'], check=True)
        # Keep supervision PAUSE until the caller verifies initial model restore.
        print(json.dumps(launch, ensure_ascii=False))


if __name__ == '__main__':
    main()
