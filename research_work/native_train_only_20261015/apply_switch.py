"""Apply the reviewed control-only change after the old coordinator has exited."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone, timedelta

REPO = Path('/home/wyz/wjc_ws/IR2-Continuous-Connectivity')
ROOT = REPO / 'experiments/native_easy_20261015_seed101'
WORK = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    old_pid = json.loads((ROOT / 'coordinator.json').read_text())['pid']
    assert not (Path('/proc') / str(old_pid)).exists(), 'Old coordinator must exit first'
    with (ROOT / '.coordinator.lock').open('a') as local_lock, \
         (ROOT.parent / '.native_training.lock').open('a') as global_lock:
        for handle in (local_lock, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((ROOT / 'pipeline.json').read_text())
        assert plan['status'] == 'paused_for_easy_review', plan['status']
        assert plan['completed_episodes'] == dict(IR2_FIFO=512, AB_learned=512)
        before = plan['source_identity']
        after = json.loads((WORK / 'staged_identity.json').read_text())
        assert set(before) == set(after)
        assert [name for name in before if before[name] != after[name]] == ['native_driver.py']
        for name, checksum in before.items():
            assert digest(ROOT / 'source' / name) == checksum, name
        for name, checksum in after.items():
            assert digest(WORK / 'source' / name) == checksum, name

        evidence = ROOT / 'checks/manual_evaluation_switch_20261015'
        evidence.mkdir()
        archive = ROOT / 'source_versions/automatic_eval_20261015'
        archive.parent.mkdir(exist_ok=True)
        shutil.copytree(ROOT / 'source', archive, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('pipeline.json', 'source_identity.json', 'TRAINING_PLAN.md'):
            shutil.copy2(ROOT / name, evidence / ('before_' + name))
        for name in ('native_driver.patch', 'training_controls_tests.log', 'test_training_controls.py'):
            shutil.copy2(WORK / name, evidence / name)

        checkpoints = [ROOT / 'common_start.pth']
        for group in plan['groups']:
            checkpoints.extend([ROOT / group / 'initial.pth', ROOT / group / 'latest.pth'])
        checkpoint_hashes = {str(p.relative_to(ROOT)): digest(p) for p in checkpoints}
        for group, episode in plan['completed_episodes'].items():
            snapshot = ROOT / group / 'checkpoints' / ('episode_%08d.pth' % episode)
            snapshot.parent.mkdir(exist_ok=True)
            assert not snapshot.exists()
            shutil.copy2(ROOT / group / 'latest.pth', snapshot)
            assert digest(snapshot) == checkpoint_hashes[group + '/latest.pth']
        # Atomic replacement occurs only while no training process holds either lock.
        destination = ROOT / 'source/native_driver.py'
        temporary = destination.with_suffix('.py.tmp')
        shutil.copy2(WORK / 'source/native_driver.py', temporary)
        temporary.replace(destination)
        plan['source_identity'] = after
        plan['automatic_development_evaluation'] = False
        plan['evaluation_policy'] = 'manual_only'
        stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
        event = dict(time=stamp, reason='User requested continuous training with evaluation only on request',
                     automatic_development_evaluation=False, episodes=plan['completed_episodes'],
                     old_pid=old_pid, changed_source_files=['native_driver.py'],
                     source_before=before['native_driver.py'], source_after=after['native_driver.py'],
                     original_source_archive=str(archive), checkpoints_unchanged=checkpoint_hashes,
                     workers=8, effective_batch=128, updates_per_episode=8,
                     periodic_save_unchanged=True, wall_guard_hours=plan['wall_guard_hours'],
                     simulated_coordinator_tests=4, new_engineering_sac_updates=0)
        plan.setdefault('control_changes', []).append(event)
        plan['updated'] = stamp
        write(ROOT / 'source_identity.json', after)
        write(ROOT / 'pipeline.json', plan)
        for relative, checksum in checkpoint_hashes.items():
            assert digest(ROOT / relative) == checksum
        write(evidence / 'switch_review.json', event)
        (ROOT / 'STOP').unlink(missing_ok=True)

        command = ['/home/wyz/anaconda3/envs/ir2_train/bin/python',
                   str(ROOT / 'source/driver.py'), '--finite-contact', '--output', str(ROOT),
                   '--resume', '--no-auto-evaluation']
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MPLCONFIGDIR='/tmp/ir2-native-mpl', PYTHONUNBUFFERED='1')
        with (ROOT / 'training.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        launch = dict(pid=process.pid, time=stamp, command=command,
                      mode='continuous paired training; automatic development evaluation disabled',
                      tensorboard_url='http://127.0.0.1:6007')
        write(evidence / 'launch.json', launch)
        write(ROOT / 'checks/current_launch.json', launch)
        print(json.dumps(launch, ensure_ascii=False))


if __name__ == '__main__':
    main()
