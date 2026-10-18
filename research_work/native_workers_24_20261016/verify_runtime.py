"""Observe resumed training and verify persisted source/checkpoint boundaries."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1] / 'experiments/native_easy_20261015_seed101'
EVIDENCE = ROOT / 'checks/parallelism_24_20261016'
WATCH = Path('/home/wyz/wjc_ws/training_supervisor_native_easy')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def sample():
    info = {line.split(':', 1)[0]: int(line.split(':', 1)[1].split()[0])
            for line in Path('/proc/meminfo').read_text().splitlines()}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'],
                         check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    used, utilization = [int(x.strip()) for x in gpu.splitlines()[0].split(',')]
    plan = read(ROOT / 'pipeline.json')
    return dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                available_memory_gib=info['MemAvailable'] / 1048576,
                used_memory_gib=(info['MemTotal'] - info['MemAvailable']) / 1048576,
                swap_in_pages=int(vm['pswpin']), swap_out_pages=int(vm['pswpout']),
                gpu_memory_mib=used, gpu_utilization=utilization,
                episodes=plan['completed_episodes'], status=plan['status'])


def main():
    capacity = read(WORK / 'capacity_review.json')
    selected = capacity['recommended_workers']
    boundary = read(EVIDENCE / 'switch_review.json')
    launch = read(EVIDENCE / 'launch.json')
    samples = []
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        assert (Path('/proc') / str(launch['pid'])).exists(), 'Resumed coordinator exited'
        row = sample()
        samples.append(row)
        with (EVIDENCE / 'runtime_memory_samples.jsonl').open('a') as output:
            output.write(json.dumps(row) + '\n')
        progressed = all(row['episodes'][name] >= episode + selected
                         for name, episode in boundary['saved_episodes'].items())
        if len(samples) % 6 == 1 or progressed:
            print(json.dumps(row), flush=True)
        if progressed and len(samples) >= 6:
            break
        time.sleep(5)
    else:
        raise RuntimeError('No complete paired wave saved within the runtime observation window')
    plan = read(ROOT / 'pipeline.json')
    assert plan['workers'] == plan['NUM_META_AGENT'] == selected
    assert plan['automatic_development_evaluation'] is False
    assert plan['test_policy_evaluations'] == 0 and not plan['stage2_started']
    assert not (ROOT / 'STOP').exists() and not (WATCH / 'PAUSE').exists()
    assert plan['effective_batch'] == 128 and plan['updates_per_completed_episode'] == 8
    for name, checksum in plan['source_identity'].items():
        assert sha(ROOT / 'source' / name) == checksum
    assert read(ROOT / 'source_identity.json') == plan['source_identity']
    assert sha(ROOT / 'dataset/native_dataset.json') == plan['dataset_sha256']
    assert plan['dataset_sha256'] == read(EVIDENCE / 'before_pipeline.json')['dataset_sha256']
    for name, checksum in boundary['unchanged_checkpoint_hashes'].items():
        path = ROOT / name
        if path.name == 'latest.pth':
            episode = boundary['saved_episodes'][path.parent.name]
            path = path.parent / 'checkpoints' / ('parallelism24_%08d.pth' % episode)
        assert sha(path) == checksum, str(path)
    progress = {name: read(ROOT / name / 'status.json') for name in plan['groups']}
    for name, status in progress.items():
        assert status['episodes'] >= boundary['saved_episodes'][name] + selected
        assert status['sac_updates'] == 216 + 8 * (status['episodes'] - 8)
    supervisor = read(WATCH / 'status.json')
    assert supervisor['status'] == 'healthy'
    assert supervisor['training_pids'] == [launch['pid']]
    result = dict(status='passed', verified_at=samples[-1]['time'], workers=selected,
                  NUM_META_AGENT=selected, coordinator_pid=launch['pid'],
                  real_training_progress=progress, both_methods_completed_new_full_waves=True,
                  source_files_verified=len(plan['source_identity']),
                  source_files_changed=sum(boundary['source_before'][name] != value
                                           for name, value in plan['source_identity'].items()),
                  dataset_identity_unchanged=True,
                  initial_and_resume_checkpoints_verified=True,
                  min_runtime_available_memory_gib=min(x['available_memory_gib'] for x in samples),
                  max_runtime_used_memory_gib=max(x['used_memory_gib'] for x in samples),
                  max_gpu_memory_mib=max(x['gpu_memory_mib'] for x in samples),
                  max_gpu_utilization=max(x['gpu_utilization'] for x in samples),
                  new_system_swap_out_pages=samples[-1]['swap_out_pages'] - samples[0]['swap_out_pages'],
                  new_system_swap_in_pages=samples[-1]['swap_in_pages'] - samples[0]['swap_in_pages'],
                  swap_scope='Global host counters include unrelated processes; old swap preexists',
                  effective_batch=plan['effective_batch'], microbatch=plan['microbatch'],
                  updates_per_completed_episode=8, automatic_evaluation=False,
                  engineering_cpu_rollouts=96, engineering_sac_updates=0,
                  test_policy_executions=0, supervision='healthy',
                  speed_measurement_scope=capacity['scope'])
    (EVIDENCE / 'post_switch_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
