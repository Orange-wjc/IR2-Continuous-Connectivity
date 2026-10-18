"""Apply a measured 12/16 selection only after the old run has saved/exited."""
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
WATCH = Path('/home/wyz/wjc_ws/training_supervisor_native_easy')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    capacity = json.loads((WORK / 'capacity_review.json').read_text())
    selected = capacity['recommended_workers']
    assert selected in (12, 16) and capacity['replay_hashes_identical']
    assert '\nOK\n' in (WORK / 'parameter_tests.log').read_text()
    assert (WATCH / 'PAUSE').exists() and (ROOT / 'STOP').exists()
    assert not (WATCH / 'active_child.json').exists()
    old_pid = json.loads((ROOT / 'coordinator.json').read_text())['pid']
    assert not (Path('/proc') / str(old_pid)).exists()
    with (ROOT / '.coordinator.lock').open('a') as local, (ROOT.parent / '.native_training.lock').open('a') as global_lock:
        for handle in (local, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((ROOT / 'pipeline.json').read_text())
        assert plan['status'] == 'paused_for_easy_review' and plan['workers'] == plan['NUM_META_AGENT'] == 12
        assert plan['automatic_development_evaluation'] is False
        before = plan['source_identity']
        for name, checksum in before.items():
            assert sha(ROOT / 'source' / name) == checksum
        evidence = ROOT / 'checks/parallelism_16_20261016'
        evidence.mkdir()
        for name in ('pipeline.json', 'source_identity.json'):
            shutil.copy2(ROOT / name, evidence / ('before_' + name))
        for name in ('capacity_review.json', 'probe_timings.json', 'parameter_tests.log'):
            shutil.copy2(WORK / name, evidence / name)
        paths = [ROOT / 'common_start.pth']
        for group, episode in plan['completed_episodes'].items():
            paths += [ROOT / group / 'initial.pth', ROOT / group / 'latest.pth']
            snapshot = ROOT / group / 'checkpoints' / ('parallelism16_%08d.pth' % episode)
            assert not snapshot.exists()
            shutil.copy2(ROOT / group / 'latest.pth', snapshot)
        checksums = {str(path.relative_to(ROOT)): sha(path) for path in paths}
        after = dict(before)
        archive = None
        if selected == 16:
            assert capacity['memory_safe'] and capacity['faster']
            staged = {str(p.relative_to(WORK / 'source')): sha(p)
                      for p in sorted((WORK / 'source').rglob('*'))
                      if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.md', '.json')}
            assert set(staged) == set(before)
            assert [name for name in before if before[name] != staged[name]] == ['driver.py']
            archive = ROOT / 'source_versions/pre_meta_agent_16_20261016'
            shutil.copytree(ROOT / 'source', archive, ignore=shutil.ignore_patterns('__pycache__'))
            destination = ROOT / 'source/driver.py'
            temporary = destination.with_suffix('.py.tmp')
            shutil.copy2(WORK / 'source/driver.py', temporary)
            temporary.replace(destination)
            after = staged
        stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
        event = dict(time=stamp, reason='User requested a 16-worker trial; retain only if measured faster and safe',
                     previous_workers=12, selected_workers=selected,
                     source_before=before, source_after=after,
                     old_source_archive=str(archive) if archive else None,
                     saved_episodes=plan['completed_episodes'], unchanged_checkpoint_hashes=checksums,
                     fixed_policy_cpu_speedup=capacity['speedup_16_over_12'],
                     engineering_train_rollouts=96, engineering_sac_updates=0,
                     development_policy_executions=0, test_policy_executions=0,
                     note='Fixed-policy rollouts equal; changing retained wave width changes long-run policy lag equally for both groups.')
        plan['source_identity'] = after
        plan.setdefault('control_changes', []).append(event)
        plan['updated'] = stamp
        write(ROOT / 'source_identity.json', after)
        write(ROOT / 'pipeline.json', plan)
        for name, checksum in checksums.items():
            assert sha(ROOT / name) == checksum
        write(evidence / 'switch_review.json', event)
        (ROOT / 'STOP').unlink()
        command = ['/home/wyz/anaconda3/envs/ir2_train/bin/python', str(ROOT / 'source/driver.py'),
                   '--finite-contact', '--output', str(ROOT), '--resume',
                   '--num-meta-agent', str(selected), '--no-auto-evaluation']
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MPLCONFIGDIR='/tmp/ir2-native-mpl', PYTHONUNBUFFERED='1')
        with (ROOT / 'training.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        launch = dict(time=stamp, pid=process.pid, command=command, selected_workers=selected)
        write(ROOT / 'checks/current_launch.json', launch)
        write(evidence / 'launch.json', launch)
        print(json.dumps(launch, ensure_ascii=False))


if __name__ == '__main__':
    main()
