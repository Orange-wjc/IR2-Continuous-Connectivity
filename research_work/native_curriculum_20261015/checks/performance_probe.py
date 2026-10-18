"""Engineering-only native training-map timings; no validation/test policies."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/tmp/ir2-native-mpl'
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

WORK=Path(__file__).resolve().parents[1];REPO=WORK.parents[1]
sys.path.insert(0,str(WORK/'source'))
from native_driver import configuration,atomic_weights,save_state,restore_state,write
from native_worker import run_episode,initialize_worker
from native_learning import NativeLearner
from finite_contact.memory import load_checkpoint,semantic_digest


def main():
    torch.set_num_threads(1)
    out=WORK/'checks/performance_probe_v2';out.mkdir(exist_ok=True)
    cfg=configuration(WORK/'checks/native_dataset.json')
    initial=load_checkpoint(REPO/'experiments/native_preflight_20261015/common_start.pth')['learner']
    source=NativeLearner(cfg,'cpu',32);source.load(copy.deepcopy(initial),optimizers=True)
    atomic_weights(out/'weights.pth',source)
    reference=None;reference_results=None;timings=[]
    for workers in (1,2,4):
        jobs=[dict(config=cfg.to_dict(),weights=str(out/'weights.pth'),episode=1001+i,map_index=i,
                   output=str(out/('workers_%d'%workers)/'episodes')) for i in range(4)]
        cache=out/('results_workers_%d.pth'%workers)
        if cache.exists():
            cached=torch.load(cache,map_location='cpu');results=cached['results'];elapsed=cached['elapsed']
        else:
            began=time.monotonic()
            if workers==1:results=[run_episode(j) for j in jobs]
            else:
                with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
                    results=list(pool.map(run_episode,jobs))
            elapsed=time.monotonic()-began
            torch.save(dict(results=results,elapsed=elapsed),cache,pickle_protocol=4)
        hashes=[semantic_digest({'transitions':r['transitions']}) for r in results]
        if reference is None:reference=hashes;reference_results=results
        assert hashes==reference,'Changing worker count changed retained experience'
        timing=dict(workers=workers,elapsed_seconds=elapsed,episodes=4,logical_replay_hashes=hashes,
                    worker_peak_rss_mib=max(r['worker_rss_mib'] for r in results),
                    max_nodes=max(r['metric']['peak_graph_nodes'] for r in results))
        timings.append(timing);print(json.dumps(timing),flush=True)
        write(out/'cpu_timings.json',timings)
    del source
    torch.set_num_threads(2)
    transitions=[t for r in reference_results for t in r['transitions']]
    gpu=[]
    for micro in (16,32,64,128):
        x=NativeLearner(cfg,'cuda',micro);x.load(copy.deepcopy(initial),optimizers=True)
        x.replay.extend((transitions*max(1,(2000+len(transitions)-1)//len(transitions)))[:10000])
        random.seed(159);x.update();torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats();began=time.monotonic()
        for _ in range(3):r=x.update()
        torch.cuda.synchronize();elapsed=time.monotonic()-began
        entry=dict(microbatch_limit=micro,actual_microbatch=r['microbatch'],max_nodes=r['max_nodes'],
                   three_updates_seconds=elapsed,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2)
        gpu.append(entry);print(json.dumps(entry),flush=True)
        del x;torch.cuda.empty_cache()
    # Actual CUDA full-state continuation: same next batch, Adam and RNG.
    x=NativeLearner(cfg,'cuda',32);x.load(copy.deepcopy(initial),optimizers=True)
    x.replay.extend((transitions*max(1,(2000+len(transitions)-1)//len(transitions)))[:10000]);x.update()
    save_state(out/'cuda_resume.pth',x,cfg,1,'engineering_probe')
    x.update();expected=semantic_digest(x.state());del x;torch.cuda.empty_cache()
    y,_,ep,_=restore_state(out/'cuda_resume.pth','cuda',32);y.update()
    assert semantic_digest(y.state())==expected,'CUDA continuation is not exactly reproducible'
    result=dict(status='passed',cpu=timings,cuda=gpu,cuda_resume_exact=True,
        parallel_replay_exact=True,train_rollouts=12,new_engineering_sac_updates=19,
        prior_failed_hash_probe_train_rollouts=4,prior_failed_hash_probe_sac_updates=0,
        validation_policy_executions=0,test_policy_executions=0,
        note='Four full native training maps, process startup included; not a complete long-run speed guarantee')
    write(WORK/'checks/performance_review.json',result)
    print('NATIVE PERFORMANCE/RESUME CHECK PASSED',flush=True)

if __name__=='__main__':main()
