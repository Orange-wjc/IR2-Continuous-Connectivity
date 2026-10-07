"""Evaluate the released IR2 policy using the unchanged TestWorker."""
import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
FIELDS = ['map_type', 'map_index', 'map_file', 'seed', 'num_robots', 'step_limit',
          'execution_ok', 'success', 'steps', 'max_distance_coordinate_units',
          'mean_explored', 'min_explored', 'individual_explored', 'final_connected',
          'wall_seconds', 'error']
PAPER = {'corridor': {'success_percent': 94, 'steps': 91, 'distance': 1212},
         'hybrid': {'success_percent': 100, 'steps': 48.8, 'distance': 581},
         'complex': {'success_percent': 87, 'steps': 185.8, 'distance': 2892}}


def evaluate(task):
    map_type, index, base_seed = task
    import numpy as np
    import torch
    import test_parameter as config
    import env
    import graph_generator
    import node
    import graph
    import test_multi_robot_worker as worker_module
    from model import PolicyNet
    seed = base_seed + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    difficult = map_type == 'complex'
    overrides = {'TEST_SET_NAME': map_type, 'TEST_SET_DIR': 'DungeonMaps/test/' + map_type,
                 'MAX_EPS_STEPS': 384 if difficult else 196,
                 'NODE_COORDS_SCALING_FACTOR': 1/1000 if difficult else 1/640,
                 'GLOBAL_GRAPH_NODE_COORDS_THRESH': 340 if difficult else 200,
                 'SAVE_GIFS': False, 'VIZ_GRAPH_EDGES': False}
    for module in (config, env, graph_generator, node, graph, worker_module):
        for name, value in overrides.items():
            setattr(module, name, value)
    if not hasattr(evaluate, 'policy'):
        evaluate.policy = PolicyNet(config.INPUT_DIM, config.EMBEDDING_DIM).to('cuda')
        checkpoint = torch.load(config.MODEL_PATH, map_location='cpu', weights_only=True)
        evaluate.policy.load_state_dict(checkpoint['policy_model'], strict=True)
        evaluate.policy.eval()
    row = dict.fromkeys(FIELDS, '')
    row.update(map_type=map_type, map_index=index, seed=seed,
               num_robots=5 if difficult else 4, step_limit=overrides['MAX_EPS_STEPS'],
               execution_ok=False, success=False)
    capture = io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(capture), torch.no_grad():
            worker = worker_module.TestWorker(0, row['num_robots'], evaluate.policy,
                                               index, device='cuda', greedy=True, save_image=False)
            row['map_file'] = worker.env.file_path
            ok = worker.work(index)
        rates = [worker.env.evaluate_exploration_rate(i) for i in range(row['num_robots'])]
        metrics = worker.perf_metrics
        row.update(execution_ok=bool(ok), success=bool(ok and worker.env.check_done()),
                   steps=metrics.get('travel_steps', ''),
                   max_distance_coordinate_units=max(r.travel_dist for r in worker.robot_list),
                   mean_explored=float(np.mean(rates)), min_explored=float(min(rates)),
                   individual_explored=json.dumps(rates),
                   final_connected=bool(worker.env.connectivity_rate))
        if not ok:
            row['error'] = 'Original worker returned False (graph/path/observation failure)'
    except Exception:
        row['error'] = traceback.format_exc()
    row['wall_seconds'] = round(time.monotonic() - started, 3)
    return row, capture.getvalue()


def summarize(rows):
    import numpy as np
    result = {}
    for kind in PAPER:
        group = [r for r in rows if r['map_type'] == kind]
        valid = [r for r in group if r['execution_ok']]
        successful = [r for r in valid if r['success']]
        def stats(items, key):
            values = [float(r[key]) for r in items if r[key] != '']
            return {'mean': float(np.mean(values)), 'std_population': float(np.std(values))} if values else None
        result[kind] = {'attempted': len(group), 'execution_failures': len(group)-len(valid),
                        'successful': len(successful),
                        'success_percent_all_attempts': 100*len(successful)/len(group) if group else None,
                        'steps_execution_ok': stats(valid, 'steps'),
                        'steps_success_only': stats(successful, 'steps'),
                        'distance_execution_ok_coordinate_units': stats(valid, 'max_distance_coordinate_units'),
                        'distance_success_only_coordinate_units': stats(successful, 'max_distance_coordinate_units'),
                        'mean_explored_all': stats(group, 'mean_explored'),
                        'paper_stage2_reference': PAPER[kind]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--output', default='reproduction_results/pretrained_stage2_20261005')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    jsonl = output / 'episodes.jsonl'
    if jsonl.exists():
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    completed = {(r['map_type'], int(r['map_index'])) for r in rows}
    tasks = [(kind, i, args.seed) for i in range(args.count) for kind in PAPER
             if (kind, i) not in completed]
    import torch
    import test_parameter
    manifest = {'command': vars(args), 'checkpoint_sha256': hashlib.sha256(Path(test_parameter.MODEL_PATH).read_bytes()).hexdigest(),
                'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'python': __import__('sys').version, 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(0), 'checkpoint_episode':
                torch.load(test_parameter.MODEL_PATH, map_location='cpu', weights_only=True).get('episode'),
                'communication': 'original signal-strength default',
                'map_order': 'original os.listdir sorted reverse=True; first count maps',
                'maps': {k: sorted(os.listdir('DungeonMaps/test/'+k), reverse=True)[:args.count] for k in PAPER},
                'test_config': {k:v for k,v in vars(test_parameter).items() if k.isupper() and isinstance(v,(str,int,float,bool))},
                'distance_units': 'original map-coordinate units; physical meters not independently established',
                'failure_policy': 'keep all attempts, no substitution or retry',
                'scheduler': 'multiprocessing spawn; original TestWorker and policy unchanged'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(evaluate, task): task for task in tasks}
        for future in concurrent.futures.as_completed(pending):
            row, log = future.result()
            rows.append(row)
            with jsonl.open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
            if log or row['error']:
                (output/(row['map_type']+'_'+str(row['map_index'])+'.log')).write_text(log+row['error'])
            with (output/'episodes.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(sorted(rows,key=lambda r:(r['map_type'],int(r['map_index']))))
            (output/'summary.json').write_text(json.dumps(summarize(rows),indent=2,ensure_ascii=False))
            print(f"Completed {len(rows)}/{3*args.count}: {row['map_type']} {row['map_index']} ok={row['execution_ok']} success={row['success']} steps={row['steps']} seconds={row['wall_seconds']} elapsed={time.monotonic()-started:.0f}s", flush=True)
    print(json.dumps(summarize(rows),indent=2,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
