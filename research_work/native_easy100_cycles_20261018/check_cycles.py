"""Fixed-pool endless controls, STOP/resume and TensorBoard scalars smoke test."""
from pathlib import Path
import json,subprocess,time,os,hashlib,sys,tempfile,copy
import numpy as np
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
PYTHON='/home/wyz/anaconda3/envs/ir2_train/bin/python'
ROOT=REPO/'experiments/native_easy100_cycles_preflight_20261018'
PARENT=REPO/'experiments/native_repair_100_preflight_v2_20261017'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stop_after(cmd,env,log,target):
 proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+600;requested=False
 while proc.poll() is None:
  if time.monotonic()>deadline:raise RuntimeError('Fixed-pool preflight timed out')
  p=ROOT/'pipeline.json'
  if not requested and p.exists() and min(read(p)['completed_episodes'].values())>=target:
   (ROOT/'STOP').touch();requested=True
  time.sleep(.05)
 assert requested and proc.returncode==0
 assert read(ROOT/'pipeline.json')['status']=='stopped_by_request'
 return read(ROOT/'pipeline.json')['completed_episodes']
def main():
 assert not ROOT.exists()
 protected={str(p):sha(p) for p in [PARENT/'pipeline.json',PARENT/'IR2_FIFO/latest.pth',PARENT/'AB_learned/latest.pth']}
 env=os.environ.copy()
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
 cmd=[PYTHON,str(REPO/'driver.py'),'--finite-contact','--output',str(ROOT)]
 log=(WORK/'preflight.log').open('w')
 first=stop_after(cmd+['--dataset',str(PARENT/'dataset/native_dataset.json'),'--from-run',str(PARENT),
  '--continuous','--train-maps','100','--preflight','--num-meta-agent','2'],env,log,6)
 plan=read(ROOT/'pipeline.json')
 assert plan['pilot_episodes'] is None and plan['wall_guard_hours'] is None
 assert plan['continuous_training'] and not plan['endpoint_development_evaluation'] and not plan['automatic_development_evaluation']
 assert not plan['stop_on_success_goal'] and plan['success_rate_goal']==.99
 assert not (ROOT/'latest_comparison.json').exists() and not (ROOT/'RESULTS.md').exists()
 (ROOT/'STOP').unlink()
 final=stop_after(cmd+['--resume'],env,log,min(first.values())+1)
 assert min(final.values())>min(first.values())
 for group in ('IR2_FIFO','AB_learned'):
  rows=[json.loads(s) for s in (ROOT/group/'training.jsonl').read_text().splitlines()]
  assert [r['episode'] for r in rows]==list(range(1,final[group]+1))
  assert all(row['map']['index']==(row['episode']-1)%100 for row in rows)
  assert not list((ROOT/group).glob('validation_*'))
 assert not (ROOT/'shared_warmup').exists() and not (ROOT/'head_collection').exists()
 assert all(sha(Path(p))==h for p,h in protected.items())
 sys.path.insert(0,str(ROOT/'source'))
 from native_driver import training_endpoint_reached,planned_wave_size,automatic_review_due,restore_state,jobs_for
 from native_training_progress import TrainingProgress
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 for g in ('IR2_FIFO','AB_learned'):
  e=EventAccumulator(str(ROOT/'tensorboard'/g));e.Reload()
  assert 'train/success_rate_100_percent' in e.Tags()['scalars']
  assert e.Scalars('train/success_rate_100_percent')[-1].step==final[g]
 with tempfile.TemporaryDirectory(prefix='ir2-cycles-controls-') as temp:
  learner,cfg,_,_=restore_state(ROOT/'IR2_FIFO/latest.pth','cpu',16)
  jobs=jobs_for(cfg,learner,Path(temp)/'jobs',99,4)
  assert [j['map_index'] for j in jobs]==[98,99,0,1]
  class Writer:
   def add_scalar(self,*args):pass
   def flush(self):pass
  d=Path(temp)/'metrics';d.mkdir();p=TrainingProgress(d,Writer(),cfg,0)
  example=json.loads((ROOT/'IR2_FIFO/training.jsonl').read_text().splitlines()[0])
  for ep in range(1,101):
   row=copy.deepcopy(example);row.update(episode=ep,success=ep!=100);p.record(row)
  assert p.summary()['success_rate_100_percent']==99.
  control=dict(continuous_training=True,pilot_episodes=None,rollout_wave=1,automatic_development_evaluation=True,review_interval=1)
  counts={'IR2_FIFO':10000,'AB_learned':10000}
  assert not training_endpoint_reached(control,counts)
  assert planned_wave_size(control,counts,'IR2_FIFO')==1
  assert not automatic_review_due(control,counts)
 report=dict(full_state_inheritance=True,fixed_pool_maps=100,wrap_indices_verified=[98,99,0,1],
  no_episode_or_walltime_limit=True,no_success_threshold_stop_at_99=True,no_automatic_evaluation=True,
  inherited_and_live_tensorboard_scalars=True,stop_and_resume=True,parent_preserved=True,
  no_prefix_or_head_retraining=True,engineering_only=True,final_episodes=final)
 (WORK/'preflight_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
