"""Exercise original driver control flow in a temporary, non-training run.

Short two-step jobs cannot fill the unchanged replay minimum2000. Verify the
actual SIGINT save/cleanup and original checkpoint resume without learning.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-original-check-mpl')
from pathlib import Path
import sys,tempfile,json,signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import ray,torch
import driver
from runner import Runner
from baseline_support import Progress,write_json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

def main():
    sys.argv=['driver.py']
    checks={}
    driver.RLRunner=ShortRunner;driver.Progress=StopProgress
    with tempfile.TemporaryDirectory(prefix='ir2-foreground-check-') as temp:
        temp=Path(temp)
        paths=dict(BASELINE_DIR=str(temp),TRAIN_SET_DIR=str(temp/'maps'),
                   MODEL_DIR=str(temp/'model'),TRAIN_DIR=str(temp/'train'),
                   GIFS_DIR=str(temp/'gifs'),MODEL_PATH=str(temp/'model/checkpoint.pth'))
        for k,v in paths.items():setattr(driver,k,v);setattr(driver.config,k,v)
        for resume in [False,True]:
            driver.LOAD_MODEL=resume;driver.CONTINUE_LOG_ALPHA=resume;driver.config.LOAD_MODEL=resume
            driver.main()
            cp=torch.load(paths['MODEL_PATH'],map_location='cpu')
            expected=4 if resume else 2
            assert cp['completed_episodes']==expected
            assert cp['training_updates']==0
            assert not ray.is_initialized()
            assert len(cp['baseline_progress']['recent'])==expected
            assert sum(cp['baseline_progress']['visits'].values())==expected
            assert all(k in cp for k in ['policy_model','q_net1_model','q_net2_model','log_alpha','policy_optimizer','q_net1_optimizer','q_net2_optimizer','log_alpha_optimizer','episode'])
            checks['resume' if resume else 'fresh']=dict(completed=cp['completed_episodes'],sac_updates=cp['training_updates'],
                saved_on_sigint=True,ray_shutdown=True,original_checkpoint_keys=True,stats_preserved=True)
            del cp
        events=list((temp/'train').iterdir())
        assert events
        last=EventAccumulator(str(events[-1]));last.Reload()
        assert 'Progress/Success Rate 100 Percent' in last.Tags()['scalars']
        checks['real_tensorboard_events']=True
        checks['no_production_checkpoint_or_progress_created']=not (ROOT/'model/ir2_baseline_easy10_20261018/checkpoint.pth').exists() and not (ROOT/'experiments/ir2_baseline_easy10_20261018/progress.jsonl').exists()
        assert checks['no_production_checkpoint_or_progress_created']
    result=dict(passed=True,scope='Original driver orchestration with original replay minimum2000; all sampling shortened to two steps, zero SAC updates, temporary checkpoints removed',checks=checks)
    write_json(ROOT/'baseline_checks/foreground_verified.json',result)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    finally:
        if ray.is_initialized():ray.shutdown()
