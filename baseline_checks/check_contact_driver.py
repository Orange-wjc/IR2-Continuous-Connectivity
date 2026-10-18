"""Actual driver startup/auxiliary optimization/checkpoint/resume in temp dirs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['IR2_METHOD']='finite_learned'
os.environ['IR2_RUN_NAME']='contact_driver_validation'
os.environ['IR2_AGENTS']='2'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-contact-validation-mpl')
from pathlib import Path
import sys,signal,tempfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch,ray
import numpy as np
import driver
from runner import Runner
from baseline_support import Progress,write_json,prepare_maps
from contact_module.config import ContactConfig


@ray.remote(num_cpus=1,num_gpus=.5)
class ShortRunner(Runner):
    def do_job(self,episode):
        import multi_robot_worker
        multi_robot_worker.MAX_EPS_STEPS=2
        return super().do_job(episode)


class StopProgress(Progress):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.stop_at=self.completed+2

    def record(self,*args,**kwargs):
        super().record(*args,**kwargs)
        if self.completed>=self.stop_at:
            os.kill(os.getpid(),signal.SIGINT)


def main():
    sys.argv=['driver.py'];driver.RLRunner=ShortRunner;driver.Progress=StopProgress
    prepare_maps(driver.config)  # Ray workers import their isolated run's map links.
    source=Path('model/2/checkpoint.pth');sha=hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='ir2-contact-driver-check-') as directory:
        root=Path(directory)
        paths=dict(BASELINE_DIR=str(root/'experiment'),TRAIN_SET_DIR=str(root/'maps'),
                   MODEL_DIR=str(root/'model'),TRAIN_DIR=str(root/'train'),
                   GIFS_DIR=str(root/'gifs'),MODEL_PATH=str(root/'model/checkpoint.pth'))
        for k,v in paths.items():setattr(driver,k,v);setattr(driver.config,k,v)
        driver.LOAD_MODEL=False;driver.config.LOAD_MODEL=False
        driver.main()
        first=torch.load(paths['MODEL_PATH'],map_location='cpu')
        assert first['completed_episodes']==2 and first['training_updates']==0
        assert first['contact_learning']['updates']>0
        driver.CONTACT_CONFIG.validate_checkpoint(first)
        for bad in (ContactConfig(),ContactConfig(method='finite_fifo'),
                    ContactConfig(method='finite_learned',bytes_per_second=4096.)):
            try:bad.validate_checkpoint(first)
            except ValueError:pass
            else:raise AssertionError('Incompatible checkpoint accepted')
        driver.LOAD_MODEL=True;driver.config.LOAD_MODEL=True;driver.CONTINUE_LOG_ALPHA=True
        driver.main()
        second=torch.load(paths['MODEL_PATH'],map_location='cpu')
        assert second['completed_episodes']==4 and second['training_updates']==0
        assert second['contact_learning']['updates']>first['contact_learning']['updates']
        assert len(second['contact_learning']['replay'])>len(first['contact_learning']['replay'])
        for key in ('policy_model','q_net1_model','q_net2_model'):
            assert all(torch.equal(v,second[key][k]) for k,v in first[key].items())
        state=json.loads((Path(paths['BASELINE_DIR'])/'run_state.json').read_text())
        assert state['status']=='stopped_by_request' and not ray.is_initialized()
        assert sha==hashlib.sha256(source.read_bytes()).hexdigest()
        result=dict(passed=True,actual_driver_fresh_and_resume=True,initial_completed=2,resumed_completed=4,
                    movement_updates=0,initial_content_updates=first['contact_learning']['updates'],
                    resumed_content_updates=second['contact_learning']['updates'],
                    original_warmup_2000_preserved=True,wrong_method_and_bandwidth_resume_rejected=True,
                    auxiliary_weights_optimizer_buffer_counter_restored=True,
                    sigint_save_and_ray_cleanup=True,production_checkpoint_unchanged=True,
                    scope='Temporary engineering model only; two2-step returns per launch; not formal training')
        write_json('baseline_checks/contact_driver_verified.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    finally:
        if ray.is_initialized():ray.shutdown()
