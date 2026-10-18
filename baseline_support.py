"""Dataset selection and progress reporting for the original IR2 driver.

No policy, reward, replay sampling or optimizer logic lives in this module.
"""
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


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


def record_setup(config, manifest):
    core = ['driver.py', 'parameter.py', 'runner.py', 'env.py', 'model.py',
            'multi_robot_worker.py', 'graph_generator.py', 'robot.py', 'sensor.py',
            'ss_realistic_model.py', 'baseline_support.py']
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
        automatic_evaluation=False, training_started=False)
    write_json(Path(config.BASELINE_DIR) / 'setup.json', provenance)
    return provenance


def memory_available_gib():
    values = {line.split(':')[0]:int(line.split()[1])
              for line in Path('/proc/meminfo').read_text().splitlines()}
    return values['MemAvailable'] / 1048576


class Progress:
    def __init__(self, directory, manifest, writer, resume=None):
        self.directory = Path(directory)
        self.writer = writer
        self.maps = [m['name'] for m in manifest['maps']]
        self.recent = deque(maxlen=100)
        self.per_map = {name:deque(maxlen=10) for name in self.maps}
        self.visits = {name:0 for name in self.maps}
        self.completed = (resume or {}).get('completed_episodes', 0)
        self.invalid = (resume or {}).get('invalid_episodes', 0)
        self.updates = (resume or {}).get('training_updates', 0)
        saved = (resume or {}).get('baseline_progress')
        if saved:
            self.recent.extend(saved['recent'])
            self.visits.update(saved['visits'])
            for name, values in saved['per_map'].items():
                self.per_map[name].extend(values)

    def snapshot(self):
        return dict(recent=list(self.recent), visits=self.visits.copy(),
                    per_map={name:list(a) for name,a in self.per_map.items()})

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
        row = dict(episode=episode, map=name, valid_rollout=bool(valid),
                   task_success=success, completed_episodes=self.completed,
                   invalid_episodes=self.invalid, training_updates=self.updates,
                   replay_size=replay_size, dispatched_episodes=dispatched,
                   metrics={k:float(v) for k,v in metrics.items()},
                   time=datetime.now().astimezone().isoformat())
        with (self.directory / 'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        rate = 100 * sum(self.recent) / len(self.recent) if self.recent else None
        state = dict(**row, success_window_size=len(self.recent), success_rate_100_percent=rate,
                     available_memory_gib=memory_available_gib(),
                     per_map={m:dict(visits=self.visits[m], window_visits=len(a),
                                    success_rate_last10_percent=100*sum(a)/len(a) if a else None)
                              for m,a in self.per_map.items()})
        write_json(self.directory / 'status.json', state)
        if valid:
            step = self.completed
            self.writer.add_scalar('Progress/Success Rate 100 Percent', rate, step)
            self.writer.add_scalar('Progress/Window Size', len(self.recent), step)
            self.writer.add_scalar('Progress/SAC Updates', self.updates, step)
            self.writer.add_scalar('Progress/Replay Size', replay_size, step)
            self.writer.add_scalar('Progress/Invalid Rollouts', self.invalid, step)
            self.writer.add_scalar('Maps/'+name+'/Success Last10 Percent',
                                   state['per_map'][name]['success_rate_last10_percent'], self.visits[name])
        print(json.dumps(dict(stage='progress', episode=episode, map=name,
            completed=self.completed, invalid=self.invalid, task_success=success,
            recent_success_percent=rate, window_size=len(self.recent),
            replay=replay_size, sac_updates=self.updates,
            min_map_visits=min(self.visits.values())), ensure_ascii=False), flush=True)
