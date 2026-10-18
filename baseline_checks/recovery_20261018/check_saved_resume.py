"""Load the actual120-update checkpoint into a temporary two-step control check."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['IR2_RUN_NAME']='ir2_baseline_resume_validation_20261018'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-repair-mpl');os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import sys,json,signal,tempfile,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch,ray,driver
from runner import Runner
from baseline_support import Progress,prepare_maps,sha256,write_json

@ray.remote(num_cpus=1,num_gpus=.25)
class ShortRunner(Runner):
 def do_job(self,episode_number):
  import multi_robot_worker
  multi_robot_worker.MAX_EPS_STEPS=2
  return super().do_job(episode_number)
class StopProgress(Progress):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.stop_at=self.completed+2
 def record(self,*a,**kw):
  super().record(*a,**kw)
  if self.completed>=self.stop_at:os.kill(os.getpid(),signal.SIGINT)

def equal(a,b):
 if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 sys.argv=['driver.py'];prepare_maps(driver.config)
 source=ROOT/'model/ir2_baseline_easy10_20261018/checkpoint.pth';before=sha256(source)
 old=torch.load(source,map_location='cpu')
 driver.RLRunner=ShortRunner;driver.Progress=StopProgress
 with tempfile.TemporaryDirectory(prefix='ir2-resume-repair-check-') as temp:
  root=Path(temp)
  paths=dict(BASELINE_DIR=str(root/'experiment'),TRAIN_SET_DIR=str(root/'maps'),
   MODEL_DIR=str(root/'model'),TRAIN_DIR=str(root/'train'),GIFS_DIR=str(root/'gifs'),MODEL_PATH=str(root/'model/checkpoint.pth'))
  Path(paths['MODEL_DIR']).mkdir();shutil.copy2(source,paths['MODEL_PATH'])
  for k,v in paths.items():setattr(driver,k,v);setattr(driver.config,k,v)
  driver.LOAD_MODEL=True;driver.CONTINUE_LOG_ALPHA=True;driver.config.LOAD_MODEL=True
  driver.main()
  result=torch.load(paths['MODEL_PATH'],map_location='cpu')
  for key in ['policy_model','q_net1_model','q_net2_model','log_alpha','policy_optimizer','q_net1_optimizer','q_net2_optimizer','log_alpha_optimizer']:assert equal(old[key],result[key]),key
  assert result['completed_episodes']==old['completed_episodes']+2
  assert result['training_updates']==old['training_updates']==120
  state=json.loads((Path(paths['BASELINE_DIR'])/'run_state.json').read_text())
  assert state['status']=='stopped_by_request'
  assert not ray.is_initialized()
  assert sha256(source)==before
  report=dict(passed=True,actual_checkpoint_loaded=True,initial_completed=old['completed_episodes'],
   resumed_completed=result['completed_episodes'],training_updates=120,new_sac_updates=0,
   all_networks_optimizers_temperature_preserved=True,production_checkpoint_unchanged=True,
   sigint_save_and_ray_cleanup=True,run_state_recorded=True,
   scope='Actual saved checkpoint copied to temporary output; only2-step samples, no new gradient; temporary model removed')
 write_json(Path(__file__).parent/'saved_resume_verified.json',report)
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 finally:
  if ray.is_initialized():ray.shutdown()
