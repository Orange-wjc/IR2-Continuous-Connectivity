"""Four original196-step Ray rollouts, compact full replay, saved weights read-only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['IR2_RUN_NAME']='ir2_baseline_repair_validation_20261018'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-repair-mpl');os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import sys,json,time,copy,resource
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch,ray
import parameter as config
from runner import Runner
from baseline_replay import PackedMask,stack_masks
from baseline_support import prepare_maps,memory_available_gib,write_json,sha256

@ray.remote(num_cpus=1,num_gpus=.25)
class CheckedRunner(Runner):
 def probe(self,weights,episode):
  np.random.seed(101+episode*1009);torch.manual_seed(101+episode*1009)
  result=self.job(weights,episode)
  return result,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def main():
 torch.set_num_threads(2);prepare_maps(config)
 checkpoint=ROOT/'model/ir2_baseline_easy10_20261018/checkpoint.pth';before=sha256(checkpoint)
 state=torch.load(checkpoint,map_location='cpu')
 weights=[state['policy_model'],state['q_net1_model']];del state
 ray.init(address='local',num_cpus=6,num_gpus=1,include_dashboard=False,
  object_store_memory=1024**3,_temp_dir='/tmp/ir2-repair-ray')
 actors=[];observations=[];memory_min=memory_available_gib();started=time.monotonic();last=0
 try:
  actors=[CheckedRunner.remote(i) for i in range(4)]
  pending=[a.probe.remote(weights,ep) for a,ep in zip(actors,[21,22,23,24])]
  while pending:
   ready,pending=ray.wait(pending,timeout=1)
   memory_min=min(memory_min,memory_available_gib())
   assert memory_available_gib()>4,'Unexpected host-memory pressure in isolated runtime check'
   for value in ray.get(ready):observations.append(value)
   if time.monotonic()-last>20:
    print(json.dumps(dict(stage='full_runtime_check',finished=len(observations),total=4,
     elapsed_seconds=round(time.monotonic()-started),available_memory_gib=memory_available_gib())),flush=True);last=time.monotonic()
   if time.monotonic()-started>600:raise TimeoutError('Full rollout engineering check exceeded600s; retain worker logs')
  records=[];rows=[[] for _ in range(15)]
  for (valid,columns,metrics,info),rss in observations:
   assert valid,(info,metrics)
   assert len(columns[0])==196*3 or bool(metrics['success_rate'])
   assert all(isinstance(v,PackedMask) for i in [5,14] for v in columns[i])
   for a,b in zip(rows,columns):a.extend(b)
   records.append(dict(info=info,transitions=len(columns[0]),rss_mib=rss,
    metrics={k:float(v) for k,v in metrics.items()}))
  n=len(rows[0]);assert n>0
  # Full10000 slots with independent tensor/payload allocations drawn from
  # complete real trajectories. This is storage stress, not learning evidence.
  replay=[[] for _ in range(15)]
  for index in range(10000):
   for column,source in zip(replay,rows):
    x=source[index%n]
    column.append(PackedMask(x.shape,x.dtype,x.binary,memoryview(x.payload).tobytes()) if isinstance(x,PackedMask) else x.clone())
  memory_min=min(memory_min,memory_available_gib())
  for i in [5,14]:assert stack_masks(replay[i][:128]).shape==(128,360,360)
  assert sha256(checkpoint)==before
  report=dict(passed=True,full_rollouts=records,compact_replay_capacity=10000,
   min_available_memory_gib=memory_min,parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
   production_checkpoint_unchanged=True,elapsed_seconds=time.monotonic()-started,
   scope='Four3-robot196-step real original-map jobs and full-capacity compact replay; no learner updates or production resume')
  write_json(Path(__file__).parent/'full_runtime_verified.json',report)
  print(json.dumps(report,indent=2),flush=True)
 finally:
  for actor in actors:ray.kill(actor)
  ray.shutdown()
if __name__=='__main__':main()
