#!/usr/bin/env python3
"""Host-side, single-run monitor. Repair proposals are isolated from live code."""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
DEFAULT_RUN = Path('/home/wyz/wjc_ws/IR2-Continuous-Connectivity/experiments/native_easy_20261015_seed101')
PYTHON = '/home/wyz/anaconda3/envs/ir2_train/bin/python'
CODEX = '/home/wyz/.vscode/extensions/openai.chatgpt-26.1007.21434-linux-x64/bin/linux-x86_64/codex'
ALLOWED = {'graph_generator.py', 'native_environment.py', 'native_worker.py',
           'node.py', 'graph.py', 'sensor.py', 'finite_contact/memory.py'}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_identity(source):
    return {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.md', '.json')}


def tail(path, limit=60000):
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read().decode('utf-8', 'replace')
    except FileNotFoundError:
        return ''


def current_log(root):
    text = tail(root / 'training.log')
    marker = '{"stage": "training_controls"'
    return text[text.rfind(marker):] if marker in text else text


def matching_processes(root, proc=Path('/proc')):
    result = []
    expected = str((root / 'source/driver.py').resolve())
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = directory.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if expected not in args or '--finite-contact' not in args:
                continue
            output = args[args.index('--output') + 1]
            state = directory.joinpath('stat').read_text().split(') ', 1)[1].split()[0]
            if Path(output).resolve() == root.resolve() and state != 'Z':
                result.append(int(directory.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def orphan_workers(process_group, proc=Path('/proc')):
    result = []
    if not process_group:
        return result
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = directory.joinpath('stat').read_text().split(') ', 1)[1].split()
            command = directory.joinpath('cmdline').read_bytes().decode()
            if (stat[0] != 'Z' and int(stat[2]) == process_group
                    and ('multiprocessing.spawn' in command or 'multiprocessing.resource_tracker' in command)):
                result.append(int(directory.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


@contextmanager
def training_locks(root):
    with (root / '.coordinator.lock').open('a') as local, (root.parent / '.native_training.lock').open('a') as global_lock:
        for handle in (local, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def fixed_policy(plan):
    assert plan['automatic_development_evaluation'] is False, 'Automatic evaluation must remain off'
    assert plan['groups'] == ['IR2_FIFO', 'AB_learned']
    assert type(plan['workers']) is int and plan['workers'] > 0
    assert plan.get('NUM_META_AGENT',plan['workers']) == plan['workers'], 'Parallelism metadata mismatch'
    assert plan['effective_batch'] == 128
    assert plan['updates_per_completed_episode'] == 8 and not plan.get('preflight')
    assert not plan.get('stage2_started', False)
    assert plan['stage'].startswith('easy'), 'Only this easy stage is authorized'
    assert plan['test_policy_evaluations'] == 0


def guard_expired(plan, coordinator):
    if plan['status'] != 'paused_for_easy_review':
        return False
    try:
        elapsed = (datetime.fromisoformat(plan['updated']) - datetime.fromisoformat(coordinator['time'])).total_seconds()
        return elapsed >= plan['wall_guard_hours'] * 3600
    except (KeyError, ValueError):
        return False


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class Supervisor:
    def __init__(self, root=DEFAULT_RUN, base=BASE):
        self.root = Path(root).resolve()
        self.repo = self.root.parent.parent
        self.base = Path(base).resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.state_file = self.base / 'status.json'
        self.state = read(self.state_file, {})
        self.stopping = False

    def event(self, name, **fields):
        row = dict(time=now(), event=name)
        row.update(fields)
        with (self.base / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def status(self, name, **fields):
        self.state.update(status=name, heartbeat=now(), supervisor_pid=os.getpid(), **fields)
        write(self.state_file, self.state)

    def cancelled(self):
        return self.stopping or (self.base / 'PAUSE').exists() or (self.root / 'STOP').exists()

    def resources_ok(self):
        free = shutil.disk_usage(self.root).free
        available = None
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024
        workers=read(self.root / 'pipeline.json',{}).get('workers',8)
        needed=max(8 * 1024**3,2 * 1024**3+workers * 700 * 1024**2)
        if free < 10 * 1024**3 or available is None or available < needed:
            return False
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, text=True)
            return gpu.returncode == 0 and int(gpu.stdout.splitlines()[0]) >= 3072
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            return False

    def verify(self, plan):
        fixed_policy(plan)
        assert source_identity(self.root / 'source') == plan['source_identity'], 'Unrecorded source change'
        assert sha(self.root / 'dataset/native_dataset.json') == plan['dataset_sha256']
        for group in plan['groups']:
            assert (self.root / group / 'latest.pth').stat().st_size > 1000000

    def restart(self, plan, reason):
        self.verify(plan)
        assert not matching_processes(self.root), 'A coordinator already exists'
        if self.cancelled():
            return
        assert self.resources_ok(), 'Insufficient free RAM or disk for restart'
        checkpoints = {g: sha(self.root / g / 'latest.pth') for g in plan['groups']}
        event = dict(time=now(), reason=reason, episodes=plan.get('completed_episodes'),
                     unchanged_checkpoint_hashes=checkpoints, automatic_development_evaluation=False)
        plan.setdefault('supervisor_restarts', []).append(event)
        plan.pop('error', None)
        plan.update(status='resume_prepared', updated=now())
        write(self.root / 'pipeline.json', plan)
        command = [PYTHON, str(self.root / 'source/driver.py'), '--finite-contact',
                   '--output', str(self.root), '--resume', '--no-auto-evaluation']
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MPLCONFIGDIR='/tmp/ir2-native-mpl', PYTHONUNBUFFERED='1')
        with (self.root / 'training.log').open('ab', buffering=0) as log:
            # KillMode=process on the monitor service preserves this independent
            # training process if the monitor is restarted or stopped.
            process = subprocess.Popen(command, cwd=self.repo, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        launch = dict(time=now(), pid=process.pid, command=command, reason=reason)
        write(self.root / 'checks/current_launch.json', launch)
        self.event('training_restarted', **launch)
        self.status('restarting', training_pid=process.pid, last_restart=launch,
                    expected_progress=plan.get('completed_episodes'), restart_started=time.time())

    def incident(self, plan, fingerprint):
        attempts = self.state.setdefault('attempts', {})
        attempt = attempts.get(fingerprint, 0) + 1
        attempts[fingerprint] = attempt
        directory = self.base / 'incidents' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + fingerprint + '_%d' % attempt)
        directory.mkdir(parents=True)
        for name in ('pipeline.json', 'coordinator.json', 'source_identity.json'):
            shutil.copy2(self.root / name, directory / name)
        (directory / 'training_tail.log').write_text(current_log(self.root))
        shutil.copytree(self.root / 'source', directory / 'source_before', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(directory / 'source_before', directory / 'source', ignore=shutil.ignore_patterns('__pycache__'))
        for group in plan['groups']:
            d = directory / group
            d.mkdir()
            for name in ('status.json', 'worker_weights.pth'):
                path = self.root / group / name
                if path.exists():
                    shutil.copy2(path, d / name)
        context = dict(time=now(), root=str(self.root), repo=str(self.repo),
                       dataset=str(self.root / 'dataset/native_dataset.json'),
                       plan=plan, fingerprint=fingerprint, attempt=attempt,
                       prior_repairs=[str(p) for p in sorted((self.base / 'incidents').glob('*' + fingerprint + '*')) if p != directory])
        write(directory / 'context.json', context)
        self.status('repairing', incident=str(directory), fingerprint=fingerprint)
        return directory

    def repair_prompt(self, directory):
        template = (self.base / 'REPAIR_INSTRUCTIONS.md').read_text()
        workers=read(directory / 'context.json')['plan']['workers']
        return template.replace('{{INCIDENT}}', str(directory)).replace('{{ROOT}}', str(self.root)).replace('{{WORKERS}}',str(workers))

    def run_child(self, command, directory, log_name, timeout, prompt=None):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', MPLCONFIGDIR='/tmp/ir2-native-mpl')
        with (directory / log_name).open('wb') as log:
            process = subprocess.Popen(command, cwd=directory, env=env,
                                       stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
            write(self.base / 'active_child.json', dict(pid=process.pid, command=command, started=time.time(), timeout=timeout))
            if prompt is not None:
                process.stdin.write(prompt.encode())
                process.stdin.close()
            began = time.monotonic()
            try:
                while process.poll() is None:
                    if self.cancelled():
                        raise InterruptedError('User paused supervision or training')
                    if time.monotonic() - began > timeout:
                        raise TimeoutError('Supervision child exceeded timeout')
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError('Child failed (%d): %s' % (process.returncode, log_name))
            finally:
                stop_process(process)
                (self.base / 'active_child.json').unlink(missing_ok=True)

    def propose(self, directory):
        command = [CODEX, '-a', 'never', '-s', 'workspace-write', 'exec',
                   '--skip-git-repo-check', '--json', '-C', str(directory),
                   '--output-schema', str(self.base / 'repair_schema.json'),
                   '-o', str(directory / 'proposal.json'), '-']
        self.run_child(command, directory, 'codex_events.jsonl', 1800, self.repair_prompt(directory))
        proposal = read(directory / 'proposal.json')
        assert proposal['status'] == 'candidate_ready', proposal.get('summary')
        assert proposal['engineering_only'] and proposal['failed_case_reproduced']
        assert not proposal['changed_training_objective'] and not proposal['skipped_maps']
        assert proposal['test_policy_executions'] == 0 and proposal['sac_updates'] == 0
        return proposal

    def validate(self, directory, plan, proposal):
        before = source_identity(directory / 'source_before')
        after = source_identity(directory / 'source')
        assert before == plan['source_identity']
        assert set(before) == set(after), 'Source files added or removed'
        changed = [name for name in before if before[name] != after[name]]
        assert changed and set(changed) <= ALLOWED, 'Repair exceeds engineering allowlist'
        assert sorted(changed) == sorted(proposal['changed_files'])
        for name in changed:
            p = directory / 'source' / name
            assert not p.is_symlink()
            ast.parse(p.read_text(), filename=name)
        self.status('validating', incident=str(directory), changed_files=changed)
        validation = (directory / proposal['validation_script']).resolve()
        assert validation.is_relative_to(directory) if hasattr(validation, 'is_relative_to') else str(validation).startswith(str(directory) + '/')
        assert validation.suffix == '.py' and validation.is_file() and not validation.is_symlink()
        # Run the proposal's regression/reproduction harness again ourselves.
        self.run_child([PYTHON, str(validation)], directory, 'independent_validation.log', 1200)
        result = read(directory / 'validation_result.json')
        assert result['passed'] and result['regression_tests'] >= 1
        assert result['failed_case_reproduced_before'] and result['failed_case_completed_after']
        assert result['test_policy_executions'] == 0 and result['sac_updates'] == 0
        # A test harness must not mutate the candidate or frozen original copy.
        assert source_identity(directory / 'source') == after
        assert source_identity(directory / 'source_before') == before
        return changed, after, result

    def activate(self, directory, plan, proposal, changed, after, validation):
        assert not matching_processes(self.root) and not self.cancelled()
        self.verify(plan)
        checkpoint_hashes = {g: sha(self.root / g / 'latest.pth') for g in plan['groups']}
        archive = self.root / 'source_versions' / ('supervisor_' + directory.name)
        archive.parent.mkdir(exist_ok=True)
        shutil.copytree(self.root / 'source', archive, ignore=shutil.ignore_patterns('__pycache__'))
        write(directory / 'checkpoint_hashes.json', checkpoint_hashes)
        for group, checksum in checkpoint_hashes.items():
            snapshots = self.root / group / 'checkpoints'
            snapshots.mkdir(exist_ok=True)
            snapshot = snapshots / ('supervisor_' + directory.name + '.pth')
            shutil.copy2(self.root / group / 'latest.pth', snapshot)
            assert sha(snapshot) == checksum
        patch = []
        for name in changed:
            old = (directory / 'source_before' / name).read_text()
            new = (directory / 'source' / name).read_text()
            patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='before/' + name, tofile='after/' + name))
        (directory / 'repair.patch').write_text(''.join(patch))
        # A durable transaction marker lets a later monitor roll back a partial
        # file replacement after an unexpected machine/process interruption.
        transaction = dict(archive=str(archive), before=plan['source_identity'], after=after)
        write(self.base / 'activation_transaction.json', transaction)
        for name in changed:
            destination = self.root / 'source' / name
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            shutil.copy2(directory / 'source' / name, temporary)
            temporary.replace(destination)
        plan['source_identity'] = after
        event = dict(time=now(), reason=proposal['summary'], incident=str(directory),
                     changed_files=changed, old_source_archive=str(archive),
                     unchanged_checkpoints=checkpoint_hashes, validation=validation,
                     episodes=plan.get('completed_episodes'))
        plan.setdefault('supervisor_repairs', []).append(event)
        write(self.root / 'source_identity.json', after)
        write(self.root / 'pipeline.json', plan)
        for group, checksum in checkpoint_hashes.items():
            assert sha(self.root / group / 'latest.pth') == checksum
        (self.base / 'activation_transaction.json').unlink()
        write(directory / 'activation.json', event)
        self.event('repair_activated', **event)

    def recover_transaction(self, plan):
        marker = self.base / 'activation_transaction.json'
        transaction = read(marker)
        if not transaction:
            return plan
        if plan['source_identity'] == transaction['after'] and source_identity(self.root / 'source') == transaction['after']:
            marker.unlink()
            return plan
        for name in transaction['before']:
            shutil.copy2(Path(transaction['archive']) / name, self.root / 'source' / name)
        plan['source_identity'] = transaction['before']
        write(self.root / 'source_identity.json', transaction['before'])
        write(self.root / 'pipeline.json', plan)
        marker.unlink()
        self.event('partial_activation_rolled_back')
        return plan

    def tick(self):
        plan = read(self.root / 'pipeline.json')
        processes = matching_processes(self.root)
        progress = {g: read(self.root / g / 'status.json', {}).get('episodes', 0) for g in plan['groups']}
        self.state.update(progress=progress, training_pids=processes, experiment_status=plan['status'])
        retry_marker = self.base / 'RETRY'
        if retry_marker.exists():
            for key in ('blocked_fingerprint', 'retry_after', 'transient_retry'):
                self.state.pop(key, None)
            self.state['attempts'] = {}
            retry_marker.unlink()
            self.event('user_reenabled_recovery')
        if self.cancelled():
            self.status('user_paused')
            return
        child_path = self.base / 'active_child.json'
        child = read(child_path)
        if child:
            process_path = Path('/proc') / str(child['pid'])
            try:
                args = process_path.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
                state = process_path.joinpath('stat').read_text().split(') ', 1)[1].split()[0]
                same_child = args == child['command'] and state != 'Z'
            except OSError:
                same_child = False
            if same_child:
                if time.time() - child['started'] > child['timeout'] + 60:
                    os.killpg(child['pid'], signal.SIGTERM)
                    self.status('waiting_for_old_repair_exit')
                else:
                    self.status('waiting_for_existing_repair', repair_pid=child['pid'])
                return
            child_path.unlink(missing_ok=True)
        if len(processes) > 1:
            self.status('needs_attention', reason='Multiple matching coordinators; no automatic mutation')
            return
        if processes:
            try:
                self.state['training_process_group'] = os.getpgid(processes[0])
            except ProcessLookupError:
                pass
            if progress != self.state.get('last_progress'):
                self.state.update(last_progress=progress, last_progress_time=time.time())
                self.state.pop('blocked_fingerprint', None)
                self.state.pop('retry_after', None)
            age = time.time() - self.state.get('last_progress_time', time.time())
            self.status('healthy' if age < 2700 else 'progress_stalled', progress_age_seconds=round(age))
            return
        # A coordinator may exit before its workers. Both native locks must be
        # free, and no matching process may exist, before diagnosis/mutation.
        try:
            with training_locks(self.root):
                plan = self.recover_transaction(read(self.root / 'pipeline.json'))
                if matching_processes(self.root):
                    self.status('waiting_for_exit')
                    return
                group = self.state.get('training_process_group', read(self.root / 'coordinator.json', {}).get('pid'))
                orphans = orphan_workers(group)
                if orphans:
                    started = self.state.setdefault('orphan_cleanup_started', time.time())
                    signum = signal.SIGTERM if time.time() - started < 60 else signal.SIGKILL
                    for pid in orphans:
                        try:
                            os.kill(pid, signum)
                        except ProcessLookupError:
                            pass
                    self.event('orphan_sampling_workers_stopped', pids=orphans, signal=signum)
                    self.status('waiting_for_orphan_exit')
                    return
                self.state.pop('orphan_cleanup_started', None)
                if plan['status'] in ('complete', 'preflight_complete'):
                    self.status('finished')
                    return
                fixed_policy(plan)
                if plan['status'] == 'paused_for_easy_review' and not guard_expired(plan, read(self.root / 'coordinator.json', {})):
                    self.status('user_paused', reason='Respect manual SIGINT/SIGTERM pause')
                    return
                if not self.resources_ok():
                    self.status('waiting_resources', reason='Need 10 GiB free disk, RAM for saved worker count (at least 8 GiB) and 3 GiB free GPU memory')
                    return
                self.verify(plan)
                if guard_expired(plan, read(self.root / 'coordinator.json', {})):
                    self.restart(plan, '48-hour checkpoint/resource review passed; continuous supervision authorized')
                    return
                error = plan.get('error', '')
                log = current_log(self.root)
                if 'CUDA out of memory' in error + log and plan['microbatch'] > 1:
                    before = plan['microbatch']
                    plan['microbatch'] = max(1, before // 2)
                    plan.setdefault('resource_adjustments', []).append(dict(time=now(), before_microbatch=before,
                        after_microbatch=plan['microbatch'], effective_batch_unchanged=128, reason='CUDA OOM'))
                    self.restart(plan, 'Reduce microbatch after CUDA OOM, preserve effective batch 128')
                    return
                fingerprint = hashlib.sha256(json.dumps(dict(error=error or log[-3000:], progress=progress), sort_keys=True).encode()).hexdigest()[:16]
                if fingerprint == self.state.get('blocked_fingerprint'):
                    self.status('needs_attention', reason='Unresolved incident; preserving checkpoints, no restart loop')
                    return
                if time.time() < self.state.get('retry_after', 0):
                    self.status('repair_backoff')
                    return
                if not error and 'Traceback' not in log and self.state.get('transient_retry') != fingerprint:
                    self.state['transient_retry'] = fingerprint
                    self.restart(plan, 'One checkpoint-preserving retry after external/process interruption')
                    return
                directory = self.incident(plan, fingerprint)
                try:
                    proposal = self.propose(directory)
                    changed, after, validation = self.validate(directory, plan, proposal)
                    self.activate(directory, plan, proposal, changed, after, validation)
                    self.restart(plan, 'Validated isolated repair: ' + proposal['summary'])
                    self.state.pop('retry_after', None)
                except InterruptedError:
                    self.status('user_paused')
                except Exception as error:
                    self.event('repair_attempt_failed', incident=str(directory), error=repr(error))
                    if self.state['attempts'][fingerprint] >= 3:
                        self.state['blocked_fingerprint'] = fingerprint
                        (self.base / 'NEEDS_ATTENTION.md').write_text('# Training supervision needs attention\n\n' + str(directory) + '\n\n' + repr(error) + '\n')
                        self.status('needs_attention', reason=repr(error))
                    else:
                        self.state['retry_after'] = time.time() + 900
                        self.status('repair_backoff', reason=repr(error))
        except BlockingIOError:
            self.status('waiting_for_training_lock')

    def run(self):
        with (self.base / '.supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.event('supervisor_started', run=str(self.root), check_interval_seconds=60)
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, lambda *_: setattr(self, 'stopping', True))
            while not self.stopping:
                try:
                    self.tick()
                except Exception as error:
                    self.event('supervisor_check_failed', error=repr(error))
                    self.status('needs_attention', reason=repr(error))
                for _ in range(60):
                    if self.stopping:
                        break
                    time.sleep(1)
            self.status('supervisor_stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'once', 'status', 'pause', 'resume'))
    parser.add_argument('--root', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    supervisor = Supervisor(args.root)
    if args.command == 'run':
        supervisor.run()
    elif args.command == 'once':
        supervisor.tick()
    elif args.command == 'status':
        print(json.dumps(read(BASE / 'status.json', {}), indent=2, ensure_ascii=False))
    elif args.command == 'pause':
        (BASE / 'PAUSE').touch()
        print('Recovery paused. Existing training continues; touch its STOP file to stop training.')
    elif args.command == 'resume':
        (BASE / 'PAUSE').unlink(missing_ok=True)
        (BASE / 'RETRY').touch()
        print('Supervision enabled; a training STOP file is still respected if present.')


if __name__ == '__main__':
    main()
