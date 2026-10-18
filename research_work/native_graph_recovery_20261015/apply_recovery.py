"""Install the regression-checked graph fix while stopped, then resume."""
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    old_pid = json.loads((ROOT / 'coordinator.json').read_text())['pid']
    assert not (Path('/proc') / str(old_pid)).exists(), 'Old coordinator still exists'
    reproduced = json.loads((WORK / 'before_failure.json').read_text())
    completed = json.loads((WORK / 'after_success.json').read_text())
    assert reproduced['pose_idx'] == -1 and reproduced['candidate_node_coords'] == []
    assert completed['metric']['episode'] == 818 and completed['metric']['decisions'] == 196
    assert completed['audit']['wall_crossings'] == 0
    assert completed['audit']['radio_totals_and_capacity_passed']
    assert '\nOK\n' in (WORK / 'regression_tests.log').read_text()
    with (ROOT / '.coordinator.lock').open('a') as local, (ROOT.parent / '.native_training.lock').open('a') as global_lock:
        for handle in (local, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((ROOT / 'pipeline.json').read_text())
        assert plan['status'] == 'interrupted_or_failed'
        assert plan['automatic_development_evaluation'] is False
        assert plan['completed_episodes'] == dict(IR2_FIFO=824, AB_learned=816)
        before = plan['source_identity']
        after = json.loads((WORK / 'staged_identity.json').read_text())
        assert set(before) == set(after)
        assert [k for k in before if before[k] != after[k]] == ['graph_generator.py']
        for name, checksum in before.items():
            assert sha(ROOT / 'source' / name) == checksum
        for name, checksum in after.items():
            assert sha(WORK / 'source' / name) == checksum
        assert sha(ROOT / 'dataset/native_dataset.json') == plan['dataset_sha256']
        evidence = ROOT / 'checks/graph_recovery_20261015'
        evidence.mkdir()
        archive = ROOT / 'source_versions/pre_graph_fix_20261015'
        shutil.copytree(ROOT / 'source', archive, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('pipeline.json', 'source_identity.json'):
            shutil.copy2(ROOT / name, evidence / ('before_' + name))
        shutil.copy2(ROOT / 'training.log', evidence / 'failure_training.log')
        for name in ('before_failure.json', 'after_success.json', 'regression_tests.log',
                     'test_graph_recovery.py', 'graph_generator.patch', 'before.log', 'after.log'):
            shutil.copy2(WORK / name, evidence / name)
        paths = [ROOT / 'common_start.pth']
        for group, episode in plan['completed_episodes'].items():
            paths += [ROOT / group / 'initial.pth', ROOT / group / 'latest.pth']
            snapshot = ROOT / group / 'checkpoints' / ('episode_%08d.pth' % episode)
            assert not snapshot.exists()
            shutil.copy2(ROOT / group / 'latest.pth', snapshot)
        checkpoint_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        destination = ROOT / 'source/graph_generator.py'
        temporary = destination.with_suffix('.py.tmp')
        shutil.copy2(WORK / 'source/graph_generator.py', temporary)
        temporary.replace(destination)
        stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
        event = dict(time=stamp, reason='Fix empty local candidates and off-grid received peer poses',
                     previous_error=plan.pop('error'), episodes=plan['completed_episodes'],
                     source_before=before['graph_generator.py'], source_after=after['graph_generator.py'],
                     changed_source_files=['graph_generator.py'], old_source_archive=str(archive),
                     unchanged_checkpoint_hashes=checkpoint_hashes,
                     regression_tests_passed=4, original_failure_reproduced=True,
                     failing_episode_replayed_completely=True, engineering_completed_rollouts=1,
                     engineering_sac_updates=0, test_policy_executions=0,
                     note='Graph fix begins from respective saved boundaries (824/816); no claim of cross-version trajectory identity. Uncommitted CPU artifacts are archived by normal resume.')
        plan['source_identity'] = after
        plan.setdefault('control_changes', []).append(event)
        plan.update(status='resume_prepared', updated=stamp)
        write(ROOT / 'source_identity.json', after)
        write(ROOT / 'pipeline.json', plan)
        for name, checksum in checkpoint_hashes.items():
            assert sha(ROOT / name) == checksum
        write(evidence / 'recovery_review.json', event)
        (ROOT / 'STOP').unlink(missing_ok=True)
        command = ['/home/wyz/anaconda3/envs/ir2_train/bin/python', str(ROOT / 'source/driver.py'),
                   '--finite-contact', '--output', str(ROOT), '--resume', '--no-auto-evaluation']
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MPLCONFIGDIR='/tmp/ir2-native-mpl', PYTHONUNBUFFERED= '1')
        with (ROOT / 'training.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        launch = dict(pid=process.pid, time=stamp, command=command,
                      mode='graph edge-case recovery; continuous paired training, manual evaluation only',
                      tensorboard_url='http://127.0.0.1:6007')
        write(ROOT / 'checks/current_launch.json', launch)
        write(evidence / 'launch.json', launch)
        print(json.dumps(launch, ensure_ascii=False))


if __name__ == '__main__':
    main()
