"""Compare 4/8 CPU workers on eight complete, identical training-map tasks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']='/tmp/ir2-native-mpl'
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time
import torch
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'source'))
from native_driver import configuration,write
from native_worker import run_episode,initialize_worker
from finite_contact.memory import semantic_digest

def main():
    cfg=configuration(WORK/'checks/native_dataset.json')
    cfg.variant='AB';cfg.content='learned'
    out=WORK/'checks/parallel_capacity_probe';out.mkdir(exist_ok=True)
    weights=WORK/'checks/performance_probe_v2/weights.pth'
    reference=None;results=[]
    for workers in (4,8):
        jobs=[dict(config=cfg.to_dict(),weights=str(weights),episode=1001+i,map_index=i,
                   output=str(out/('workers_%d'%workers)/'episodes')) for i in range(8)]
        began=time.monotonic()
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
            episodes=list(pool.map(run_episode,jobs))
        elapsed=time.monotonic()-began
        hashes=[semantic_digest({'transitions':r['transitions']}) for r in episodes]
        if reference is None:reference=hashes
        assert hashes==reference
        row=dict(workers=workers,wall_seconds=elapsed,episodes=8,worker_peak_rss_mib=max(r['worker_rss_mib'] for r in episodes),
                 logical_hashes=hashes,max_graph_nodes=max(r['metric']['peak_graph_nodes'] for r in episodes))
        results.append(row);print(json.dumps(row),flush=True)
        write(out/'timings.json',results)
    write(WORK/'checks/parallel_capacity_review.json',dict(status='passed',timings=results,
        speedup_eight_over_four=results[0]['wall_seconds']/results[1]['wall_seconds'],
        method='AB_learned',new_engineering_train_rollouts=16,new_engineering_sac_updates=0,validation_policy_executions=0,test_policy_executions=0,
        note='Includes worker startup; eight native training maps, exact replay preserved, not a full-run throughput promise'))
    print('PARALLEL CAPACITY CHECK PASSED',flush=True)

if __name__=='__main__':main()
