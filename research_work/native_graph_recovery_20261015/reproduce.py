import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import traceback
import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path.insert(0, str(WORK / 'source'))
from native_driver import configuration
from native_worker import run_episode, initialize_worker

if __name__ == '__main__':
    initialize_worker()
    cfg = configuration(REPO / 'experiments/native_easy_20261015_seed101/dataset/native_dataset.json')
    cfg.variant = 'AB'
    cfg.content = 'learned'
    mode = sys.argv[1] if len(sys.argv) > 1 else 'before'
    try:
        result = run_episode(dict(config=cfg.to_dict(), weights=str(WORK / 'ab_episode_818_weights.pth'),
                                  episode=818, map_index=817, output=str(WORK / mode / 'episodes')))
    except Exception as error:
        frame = error.__traceback__
        while frame.tb_next is not None:
            frame = frame.tb_next
        local = frame.tb_frame.f_locals
        report = dict(error=repr(error), file=frame.tb_frame.f_code.co_filename,
                      function=frame.tb_frame.f_code.co_name, line=frame.tb_lineno)
        for key in ('id', 'position', 'robot_location', 'pose_idx', 'x0', 'x1', 'y0', 'y1',
                    'candidate_node_coords', 'local_occupancy_map'):
            value = local.get(key)
            report[key] = value.tolist() if hasattr(value, 'tolist') else value
        (WORK / (mode + '_failure.json')).write_text(json.dumps(report, indent=2) + '\n')
        traceback.print_exc()
        print(json.dumps(report), flush=True)
        raise SystemExit(1)
    (WORK / (mode + '_success.json')).write_text(json.dumps(dict(metric=result['metric'], audit=result['audit']), indent=2) + '\n')
    print(json.dumps(dict(status='passed', metric=result['metric'], audit=result['audit'])), flush=True)
