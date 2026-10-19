"""Eight real Ray samplers + original GPU SAC on module-generated observations.

Uses an isolated run name; does not start a persistent training run.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['IR2_METHOD']='finite_learned'
os.environ['IR2_RUN_NAME']='contact_runtime_validation'
os.environ['IR2_AGENTS']='8'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-contact-validation-mpl')
from pathlib import Path
import sys,time,json,importlib.util,subprocess,textwrap,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch,ray
import parameter as config
from runner import Runner
from model import PolicyNet,QNet
from baseline_support import prepare_maps,write_json,memory_available_gib
from baseline_replay import stack_masks
from contact_module.learning import ContentTrainer
from contact_module.state import zero_projection_import
from contact_module.preparation import load_initialization,load_content_initialization
CHECK_STEPS=int(os.environ.get('IR2_CHECK_STEPS','8'))
CHECK_EPISODES=int(os.environ.get('IR2_CHECK_EPISODES','8'))


@ray.remote(num_cpus=1,num_gpus=1/8)
class ShortRunner(Runner):
    def do_job(self,episode):
        import multi_robot_worker
        multi_robot_worker.MAX_EPS_STEPS=CHECK_STEPS
        if not getattr(multi_robot_worker.Worker, '_decision_check', False):
            original = multi_robot_worker.Worker.run_episode
            def audited(worker, number):
                valid = original(worker, number)
                if valid:
                    for robot in worker.robot_list:
                        rows = robot.episode_buffer
                        assert len({len(c) for c in rows}) == 1
                        for index in range(len(rows[0])-1):
                            for column in range(6):
                                a, b = rows[9+column][index], rows[column][index+1]
                                if hasattr(a, 'unpack'): a, b = a.unpack(), b.unpack()
                                assert torch.equal(a, b), (number, robot.robot_id, index, column)
                return valid
            multi_robot_worker.Worker.run_episode = audited
            multi_robot_worker.Worker._decision_check = True
        np.random.seed(101+episode);torch.manual_seed(101+episode)
        return super().do_job(episode)


def main():
    torch.set_num_threads(2);manifest=prepare_maps(config)
    protected=[Path('model/2/checkpoint.pth'),Path('experiments/2/progress.jsonl')]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    torch.manual_seed(101)
    p=PolicyNet(6,128,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim);q=QNet(6,128,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim)
    provenance=saved=None
    if os.environ.get('IR2_CHECK_RESUME_CHECKPOINT'):
        if os.environ.get('IR2_CHECK_INITIALIZATION'):
            raise ValueError('Choose a prepared resume or a fresh initialization')
        source=torch.load(os.environ['IR2_CHECK_RESUME_CHECKPOINT'],map_location='cpu')
        config.CONTACT_CONFIG.validate_checkpoint(source)
        p.load_state_dict(source['policy_model']);q.load_state_dict(source['q_net1_model'])
        saved=source['contact_learning'];provenance=saved.get('provenance')
    if os.environ.get('IR2_CHECK_INITIALIZATION'):
        source,provenance=load_initialization(os.environ['IR2_CHECK_INITIALIZATION'],manifest)
        p.load_state_dict(zero_projection_import(p,source['policy_model']));q.load_state_dict(zero_projection_import(q,source['q_net1_model']))
        if os.environ.get('IR2_CHECK_CONTENT_INIT'):
            saved,_=load_content_initialization(os.environ['IR2_CHECK_CONTENT_INIT'],
                config.CONTACT_CONFIG,provenance,source['policy_model'])
    auxiliary=ContentTrainer(p,config.CONTACT_CONFIG,saved,provenance=provenance)
    starting_auxiliary_updates=auxiliary.updates
    weights=[p.state_dict(),q.state_dict(),auxiliary.weights()]
    started=time.monotonic();minimum=memory_available_gib()
    ray.init(address='local',num_cpus=10,num_gpus=1,include_dashboard=False,
             object_store_memory=1024**3,_temp_dir='/tmp/ir2-contact-check-ray')
    actors=[];results=[]
    try:
        actors=[ShortRunner.remote(i) for i in range(8)]
        pending=[actors[i%8].job.remote(weights,i+1) for i in range(CHECK_EPISODES)]
        last=0
        while pending:
            ready,pending=ray.wait(pending,timeout=1)
            minimum=min(minimum,memory_available_gib());assert minimum>4
            results.extend(ray.get(ready))
            elapsed=time.monotonic()-started
            if elapsed-last>20:
                print(json.dumps(dict(stage='module_runtime_check',finished=len(results),total=CHECK_EPISODES,
                     elapsed_seconds=round(elapsed),available_memory_gib=memory_available_gib())),flush=True)
                last=elapsed
            if elapsed>600:raise TimeoutError('Isolated Ray validation exceeded 600 seconds')
    finally:
        for a in actors:ray.kill(a)
        ray.shutdown()
    columns=[[] for _ in range(15)];records=[]
    for valid,rows,metrics,info in results:
        assert valid,(info['episode_number'],metrics)
        for a,b in zip(columns,rows):a.extend(b)
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics['contact_sent_bytes']>0
        assert metrics['contact_positive_empty_waits']==0
        assert metrics['contact_delivery_ledger_residual']==0
        assert abs(metrics['contact_reward_delivery']*1000 + metrics['contact_initial_delivery_cells']
                   - metrics['contact_unique_cells']) < 1e-6
        loss=auxiliary.update(info['contact_labels']);assert loss is not None
        records.append(dict(episode=info['episode_number'],transitions=len(rows[0]),
                            labels=len(info['contact_labels']),metrics={k:float(v) for k,v in metrics.items()}))
    assert len(columns[0])>=128
    batch=[stack_masks(c[:128]) if i in (5,14) else torch.stack(c[:128]) for i,c in enumerate(columns)]
    # Reuse the validated engineering harness, preserving the exact original
    # batch128 x8 SAC code while enabling WAIT in both actor and critics.
    path=ROOT/'baseline_checks/recovery_20261018/check_cuda_equivalence.py'
    spec=importlib.util.spec_from_file_location('cuda_check',path)
    check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
    check.PolicyNet=lambda i,e:PolicyNet(i,e,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim)
    check.QNet=lambda i,e:QNet(i,e,allow_wait=True,contact_state_dim=config.CONTACT_CONFIG.state_dim)
    source=subprocess.check_output(['git','show','main:driver.py'],text=True)
    lo=source.index('                    ### Obtain losses via SAC (Soft Actor-Critic) ###')
    hi=source.index('                perf_data = []',lo)
    code=compile(textwrap.dedent(source[lo:hi]),'original_sac','exec')
    state,peak=check.run(batch,code)
    assert all(torch.isfinite(v).all() for v in state['global_policy_net'].values())
    state_policy_norm=float(state['global_policy_net']['communication_embedding.weight'].norm())
    state_q1_norm=float(state['global_q_net1']['communication_embedding.weight'].norm())
    assert state_policy_norm>0 and state_q1_norm>0
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report=dict(passed=True,workers=8,robots=3,rollout_steps=CHECK_STEPS,
                actual_distinct_transitions=len(columns[0]),effective_gpu_batch=128,
                original_sac_gpu_iterations=8,auxiliary_optimizer_updates=auxiliary.updates,
                auxiliary_new_updates=auxiliary.updates-starting_auxiliary_updates,
                peak_cuda_mib=peak,min_host_available_gib=minimum,
                elapsed_seconds=time.monotonic()-started,rollouts=records,
                production_files_unchanged=True,
                all_own_decision_next_states_exact=True,all_actual_delivery_rewards_conserved=True,
                communication_policy_weight_norm=state_policy_norm,communication_q1_weight_norm=state_q1_norm,
                scope='Isolated engineering samples and copy-only gradients; no formal training/effectiveness evidence')
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in list(Path('contact_module').glob('*.py'))+
                            [Path(n) for n in ('driver.py','runner.py','env.py','model.py','multi_robot_worker.py')]}
    name='contact_full_runtime_verified.json' if CHECK_STEPS==196 else 'contact_runtime_verified.json'
    write_json('baseline_checks/'+name,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
