"""Dataset selection and progress reporting for the original IR2 driver.

No policy, reward, replay sampling or optimizer logic lives in this module.
"""
from collections import deque
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


class _WorkerOutput:
    """Write every worker message to both its file and the console."""
    def __init__(self, detail, console):
        self.detail = detail
        self.console = console
        self.pending = ''

    def _emit(self, line):
        self.console.write(line)
        self.console.flush()

    def write(self, value):
        self.detail.write(value)
        self.pending += value
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            self._emit(line + '\n')
        return len(value)

    def flush(self):
        self.detail.flush()
        self.console.flush()

    def finish(self):
        if self.pending:
            self._emit(self.pending)
            self.pending = ''
        self.flush()


@contextmanager
def worker_output(directory, worker_id):
    directory = Path(directory) / 'worker_logs'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('worker_%d.log' % worker_id)).open('a', buffering=1) as detail:
        output = _WorkerOutput(detail, sys.stdout)
        with redirect_stdout(output):
            try:
                yield
            finally:
                output.finish()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_maps(config):
    """Expose the first ten original PNGs without changing Env or its modulo."""
    source = Path(config.TRAIN_SET_SOURCE_DIR).resolve()
    files = sorted(source.glob('*.png'), key=lambda p: p.name, reverse=True)
    selected = files[:config.TRAIN_MAP_LIMIT]
    if len(selected) != config.TRAIN_MAP_LIMIT or config.TRAIN_MAP_LIMIT < 1:
        raise ValueError('Not enough original PNG maps for TRAIN_MAP_LIMIT')
    directory = Path(config.TRAIN_SET_DIR).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    expected = {p.name for p in selected}
    unexpected = {p.name for p in directory.iterdir()} - expected
    if unexpected:
        raise ValueError('Unexpected files in the fixed-map directory: ' + str(unexpected))
    entries = []
    for src in selected:
        dest = directory / src.name
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src:
                raise ValueError('Map link differs from the selected original: ' + str(dest))
        else:
            dest.symlink_to(src)
        entries.append(dict(name=src.name, source=str(src), sha256=sha256(src)))
    manifest = dict(source_directory=str(source), available_maps=len(files),
                    selection='First PNGs in original reverse filename ordering; no score-based selection',
                    map_count=len(entries), maps=entries,
                    episode_index_rule='episode_number % map_count (original Env rule)',
                    generated_maps=0, resized_maps=0)
    path = Path(config.BASELINE_DIR) / 'map_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Existing map manifest differs; use a new IR2_RUN_NAME')
    else:
        write_json(path, manifest)
    return manifest


def record_setup(config, manifest, check_only=False):
    core = ['driver.py', 'parameter.py', 'runner.py', 'env.py', 'model.py',
            'multi_robot_worker.py', 'graph_generator.py', 'robot.py', 'sensor.py',
            'ss_realistic_model.py', 'baseline_support.py', 'baseline_replay.py']
    core += ['training_review.py']
    core += [str(p) for p in sorted(Path('contact_module').glob('*.py'))]
    provenance = dict(created=datetime.now().astimezone().isoformat(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p:sha256(p) for p in core}, map_manifest=manifest,
        num_meta_agent=config.NUM_META_AGENT, num_gpu=config.NUM_GPU,
        batch_size=config.BATCH_SIZE, replay_size=config.REPLAY_SIZE,
        minimum_buffer_size=config.MINIMUM_BUFFER_SIZE, learning_rate=config.LR,
        robots=[config.NUM_ROBOTS_MIN, config.NUM_ROBOTS_MAX], input_dim=config.INPUT_DIM,
        embedding_dim=config.EMBEDDING_DIM, max_episode_steps=config.MAX_EPS_STEPS,
        updates_per_training_iteration=8, resume=config.LOAD_MODEL,
        original_signal_strength_model=config.USE_SIGNAL_STRENGTH_NOT_PROXIMITY,
        replay_mask_storage='lossless-bitpack-zlib-v1',
        graph_boundary_repair='Do not append a local candidate when all K slots are global neighbors',
        automatic_evaluation=False, training_started=False,
        training_review_interval=300 if config.CONTACT_CONFIG.enabled else None,
        automatic_review_control=os.environ.get('IR2_AUTO_REVIEW','0')=='1',
        acceptance=dict(window_attempts=300,minimum_successes=297,all_maps_minimum_visits=20,
                        invalid_attempts_allowed=0),
        contact_config=config.CONTACT_CONFIG.to_dict())
    path = Path(config.BASELINE_DIR) / ('config_check.json' if check_only else 'setup.json')
    if not check_only and path.exists():
        previous = json.loads(path.read_text())
        history = path.parent / 'setup_history'
        write_json(history / (datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'), previous)
    write_json(path, provenance)
    return provenance


def memory_available_gib():
    values = {line.split(':')[0]:int(line.split()[1])
              for line in Path('/proc/meminfo').read_text().splitlines()}
    return values['MemAvailable'] / 1048576


class Progress:
    def __init__(self, directory, manifest, writer, resume=None, review_enabled=False):
        self.directory = Path(directory)
        self.writer = writer
        self.maps = [m['name'] for m in manifest['maps']]
        self.recent = deque(maxlen=100)
        self.per_map = {name:deque(maxlen=10) for name in self.maps}
        self.visits = {name:0 for name in self.maps}
        self.completed = (resume or {}).get('completed_episodes', 0)
        self.invalid = (resume or {}).get('invalid_episodes', 0)
        self.updates = (resume or {}).get('training_updates', 0)
        self.review = None
        self.last_review = None
        self.acceptance = None
        self.acceptance_recent = deque(maxlen=300)
        if review_enabled:
            from training_review import TrainingReview
            self.review = TrainingReview(directory, self.maps, writer)
        saved = (resume or {}).get('baseline_progress')
        if saved:
            self.acceptance_recent.extend(saved.get('acceptance_recent',[]))
            self.recent.extend(saved['recent'])
            self.visits.update(saved['visits'])
            for name, values in saved['per_map'].items():
                self.per_map[name].extend(values)

    def snapshot(self):
        return dict(recent=list(self.recent), visits=self.visits.copy(),
                    per_map={name:list(a) for name,a in self.per_map.items()},
                    acceptance_recent=list(self.acceptance_recent))

    def record(self, info, valid, metrics, replay_size, dispatched):
        episode = info['episode_number']
        name = self.maps[episode % len(self.maps)]
        if valid:
            self.completed += 1
            success = bool(metrics['success_rate'])
            self.recent.append(success)
            self.per_map[name].append(success)
            self.visits[name] += 1
        else:
            self.invalid += 1
            success = None
        self.acceptance_recent.append(dict(episode=episode,map=name,valid=bool(valid),
                                           success=bool(success),
                                           reward_anomaly=metrics.get('contact_positive_empty_waits',0)>0))
        row = dict(episode=episode, map=name, valid_rollout=bool(valid),
                   task_success=success, completed_episodes=self.completed,
                   invalid_episodes=self.invalid, training_updates=self.updates,
                   replay_size=replay_size, dispatched_episodes=dispatched,
                   available_memory_gib=memory_available_gib(),
                   metrics={k:float(v) for k,v in metrics.items()},
                   diagnostics=info.get('contact_diagnostics',[]),
                   time=datetime.now().astimezone().isoformat())
        with (self.directory / 'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        rate = 100 * sum(self.recent) / len(self.recent) if self.recent else None
        state = dict(**row, success_window_size=len(self.recent), success_rate_100_percent=rate,
                     per_map={m:dict(visits=self.visits[m], window_visits=len(a),
                                    success_rate_last10_percent=100*sum(a)/len(a) if a else None)
                              for m,a in self.per_map.items()})
        write_json(self.directory / 'status.json', state)
        if valid:
            step = self.completed
            for metric_name, value in metrics.items():
                if metric_name.startswith('contact_'):
                    self.writer.add_scalar('Contact/' + metric_name[8:], value, step)
            self.writer.add_scalar('Progress/Success Rate 100 Percent', rate, step)
            self.writer.add_scalar('Progress/Window Size', len(self.recent), step)
            self.writer.add_scalar('Progress/SAC Updates', self.updates, step)
            self.writer.add_scalar('Progress/Replay Size', replay_size, step)
            self.writer.add_scalar('Progress/Invalid Rollouts', self.invalid, step)
            self.writer.add_scalar('Maps/'+name+'/Success Last10 Percent',
                                   state['per_map'][name]['success_rate_last10_percent'], self.visits[name])
        print(json.dumps(dict(stage='progress' if valid else 'warning', episode=episode, map=name,
            completed=self.completed, invalid=self.invalid, task_success=success,
            recent_success_percent=rate, window_size=len(self.recent),
            replay=replay_size, sac_updates=self.updates,
            min_map_visits=min(self.visits.values())), ensure_ascii=False), flush=True)
        self.last_review = None
        if valid and self.review is not None:
            try:
                self.last_review = self.review.maybe_record(self.completed)
            except Exception as exc:
                # A plotting/reporting failure must not discard model updates.
                error = dict(stage='training_review_error',completed=self.completed,error=repr(exc))
                write_json(self.directory/'reviews'/('error_{:06d}.json'.format(self.completed)),error)
                self.last_review = dict(flags=['review_failed'],trend='no_clear_rise',error=repr(exc))
                print(json.dumps(error,ensure_ascii=False),flush=True)
            attempts = list(self.acceptance_recent)
            count = sum(r['success'] for r in attempts)
            visits = {m:sum(r['map']==m for r in attempts) for m in self.maps}
            if (len(attempts)==300 and count>=297 and min(visits.values())>=20 and
                all(r['valid'] and not r.get('reward_anomaly',False) for r in attempts)):
                self.acceptance = dict(completed_episodes=self.completed,window_attempts=300,
                    successes=count,success_percent=100*count/300,per_map_visits=visits,
                    invalid_attempts=0,reward_anomalies=0,training_updates=self.updates,attempts=attempts,
                    scope='Training criterion only; no held-out performance claim')
                write_json(self.directory/'acceptance.json',self.acceptance)
