"""Bounded engineering check; never launches the production training loop.

Uses two original-Runner rollouts shortened to two steps, then one original SAC
optimizer block (eight iterations) at batch128. No model or training log is written to the run.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-original-check-mpl')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import json,random,time,subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ray
import parameter as config
from model import PolicyNet,QNet
from baseline_support import prepare_maps,write_json
from runner import Runner

def main():
    torch.set_num_threads(2);torch.manual_seed(101);np.random.seed(101);random.seed(101)
    manifest=prepare_maps(config)
    from env import Env
    checks={}
    names=[e['name'] for e in manifest['maps']]
    # Original Env remains untouched; verify selection, actual decoding and wrap.
    visited=[]
    for ep in [1,9,10,11,20]:
        env=Env(ep,config.NUM_ROBOTS_MIN,k_size=config.K_SIZE,plot=False)
        visited.append((ep,env.file_path))
        assert env.file_path==names[ep%10]
        assert env.ground_truth.shape==(480,640)
        assert len(np.unique(env.ground_truth))==2
        del env
    checks['original_env_map_wrap_and_decoding']=visited
    ray.init(address='local',num_cpus=4,num_gpus=1,include_dashboard=False,
             object_store_memory=512*1024**2,_temp_dir='/tmp/ir2-check-ray')
    @ray.remote(num_cpus=1,num_gpus=.25)
    class ShortRunner(Runner):
        def do_job(self,episode_number):
            import multi_robot_worker
            multi_robot_worker.MAX_EPS_STEPS=2
            return super().do_job(episode_number)
    actors=[]
    try:
        cpu_policy=PolicyNet(config.INPUT_DIM,config.EMBEDDING_DIM)
        cpu_q=QNet(config.INPUT_DIM,config.EMBEDDING_DIM)
        weights=[cpu_policy.state_dict(),cpu_q.state_dict()]
        actors=[ShortRunner.remote(i) for i in range(2)]
        items=ray.get([a.job.remote(weights,i+1) for i,a in enumerate(actors)],timeout=120)
        experience_buffer=[[] for _ in range(15)]
        checks['two_original_runner_jobs']=[]
        for valid,rows,metrics,info in items:
            assert valid,(info,metrics)
            assert len(rows)==15 and len(rows[0])>=6
            for i in range(15):experience_buffer[i].extend(rows[i])
            checks['two_original_runner_jobs'].append(dict(info=info,transitions=len(rows[0]),metrics={k:float(v) for k,v in metrics.items()}))
        for a in actors:ray.kill(a)
        actors=[]
        # Replay is repeated solely to exercise batch128 device memory, not to
        # claim real training or alter the production minimum replay size2000.
        indices=[i%len(experience_buffer[0]) for i in range(config.BATCH_SIZE)]
        rollouts=[[rows[i] for i in indices] for rows in experience_buffer]
        names=['node_inputs_batch','edge_inputs_batch','current_inputs_batch','node_padding_mask_batch',
               'edge_padding_mask_batch','edge_mask_batch','action_batch','reward_batch','done_batch',
               'next_node_inputs_batch','next_edge_inputs_batch','next_current_inputs_batch',
               'next_node_padding_mask_batch','next_edge_padding_mask_batch','next_edge_mask_batch']
        values={name:torch.stack(rows).cuda() for name,rows in zip(names,rollouts)}
        nets={name:cls(config.INPUT_DIM,config.EMBEDDING_DIM).cuda() for name,cls in [
            ('global_policy_net',PolicyNet),('global_q_net1',QNet),('global_q_net2',QNet),
            ('global_target_q_net1',QNet),('global_target_q_net2',QNet)]}
        nets['global_target_q_net1'].load_state_dict(nets['global_q_net1'].state_dict())
        nets['global_target_q_net2'].load_state_dict(nets['global_q_net2'].state_dict())
        for key in ('global_target_q_net1','global_target_q_net2'):nets[key].eval()
        values.update(nets)
        for alias,name in [('dp_policy','global_policy_net'),('dp_q_net1','global_q_net1'),('dp_q_net2','global_q_net2'),('dp_target_q_net1','global_target_q_net1'),('dp_target_q_net2','global_target_q_net2')]:
            values[alias]=nn.DataParallel(nets[name])
        for name in ['global_policy','global_q_net1','global_q_net2']:
            net=nets['global_policy_net' if name=='global_policy' else name]
            values[name+'_optimizer']=optim.Adam(net.parameters(),lr=config.LR)
        values['log_alpha']=torch.tensor([float(config.INITIAL_LOG_ALPHA)],device='cuda',requires_grad=True)
        values['log_alpha_optimizer']=optim.Adam([values['log_alpha']],lr=1e-4)
        values.update(torch=torch,nn=nn,GAMMA=config.GAMMA,
                      entropy_target=.05*(-np.log(1/config.K_SIZE)),target_q_update_counter=1)
        original=subprocess.check_output(['git','show','main:driver.py'],text=True)
        start=original.index('                    ### Obtain losses via SAC (Soft Actor-Critic) ###')
        end=original.index('                perf_data = []',start)
        import textwrap
        code=textwrap.dedent(original[start:end])
        before=[p.detach().clone() for p in nets['global_policy_net'].parameters()]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(8):
            exec(compile(code,'original_main_sac_block','exec'),values)
        torch.cuda.synchronize()
        assert any(not torch.equal(a,b) for a,b in zip(before,nets['global_policy_net'].parameters()))
        for key in ['policy_loss','q1_loss','q2_loss','alpha_loss']:
            assert torch.isfinite(values[key]).all(),key
        for name in ['global_policy_optimizer','global_q_net1_optimizer','global_q_net2_optimizer','log_alpha_optimizer']:
            assert all(int(s['step'])==8 for s in values[name].state.values())
        checks['original_sac_batch128_cuda_block']=dict(effective_batch=config.BATCH_SIZE,optimizer_iterations=8,
            policy_loss=float(values['policy_loss']),q1_loss=float(values['q1_loss']),
            q2_loss=float(values['q2_loss']),alpha_loss=float(values['alpha_loss']),
            policy_parameters_changed=True,peak_cuda_mib=torch.cuda.max_memory_allocated()/1048576)
        result=dict(passed=True,scope='Two original-map 2-step CPU/Ray rollouts and eight unchanged original batch128 CUDA updates; not a learnability experiment',checks=checks)
        write_json(ROOT/'baseline_checks/runtime_verified.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    finally:
        for a in actors:ray.kill(a)
        ray.shutdown()
if __name__=='__main__':main()
