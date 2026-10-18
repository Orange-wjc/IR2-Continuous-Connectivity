"""Dense versus lossless replay: same original SAC batch128 x8 updates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-repair-mpl');os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import sys,json,random,subprocess,textwrap,gc,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import parameter as config
from model import PolicyNet,QNet
from baseline_replay import PackedMask,stack_masks
from baseline_support import prepare_maps,write_json

NAMES=['node_inputs_batch','edge_inputs_batch','current_inputs_batch','node_padding_mask_batch',
 'edge_padding_mask_batch','edge_mask_batch','action_batch','reward_batch','done_batch',
 'next_node_inputs_batch','next_edge_inputs_batch','next_current_inputs_batch',
 'next_node_padding_mask_batch','next_edge_padding_mask_batch','next_edge_mask_batch']
def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
 return copy.deepcopy(value)
def run(batch,code):
 torch.manual_seed(43);torch.cuda.manual_seed_all(43)
 torch.cuda.reset_peak_memory_stats()
 values={name:t.cuda() for name,t in zip(NAMES,batch)}
 nets={name:cls(6,128).cuda() for name,cls in [('global_policy_net',PolicyNet),('global_q_net1',QNet),('global_q_net2',QNet),('global_target_q_net1',QNet),('global_target_q_net2',QNet)]}
 nets['global_target_q_net1'].load_state_dict(nets['global_q_net1'].state_dict())
 nets['global_target_q_net2'].load_state_dict(nets['global_q_net2'].state_dict())
 for name in ['global_target_q_net1','global_target_q_net2']:nets[name].eval()
 values.update(nets)
 for alias,name in [('dp_policy','global_policy_net'),('dp_q_net1','global_q_net1'),('dp_q_net2','global_q_net2'),('dp_target_q_net1','global_target_q_net1'),('dp_target_q_net2','global_target_q_net2')]:values[alias]=nn.DataParallel(nets[name])
 for alias,name in [('global_policy_optimizer','global_policy_net'),('global_q_net1_optimizer','global_q_net1'),('global_q_net2_optimizer','global_q_net2')]:values[alias]=optim.Adam(nets[name].parameters(),lr=config.LR)
 values['log_alpha']=torch.tensor([float(config.INITIAL_LOG_ALPHA)],device='cuda',requires_grad=True)
 values['log_alpha_optimizer']=optim.Adam([values['log_alpha']],lr=1e-4)
 values.update(torch=torch,nn=nn,GAMMA=config.GAMMA,entropy_target=.05*(-np.log(1/config.K_SIZE)),target_q_update_counter=1)
 for _ in range(8):exec(code,values)
 torch.cuda.synchronize()
 for name in ['policy_loss','q1_loss','q2_loss','alpha_loss']:assert torch.isfinite(values[name]).all()
 state={name:cpu(net.state_dict()) for name,net in nets.items()}
 for name in ['global_policy_optimizer','global_q_net1_optimizer','global_q_net2_optimizer','log_alpha_optimizer']:
  assert all(int(s['step'])==8 for s in values[name].state.values())
  state[name]=cpu(values[name].state_dict())
 state['log_alpha']=cpu(values['log_alpha'])
 return state,torch.cuda.max_memory_allocated()/1048576

def main():
 torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17);random.seed(17)
 prepare_maps(config)
 import multi_robot_worker as workers
 steps=workers.MAX_EPS_STEPS;workers.MAX_EPS_STEPS=2
 try:
  w=workers.Worker(0,3,PolicyNet(6,128),QNet(6,128),21,device='cpu',save_image=False)
  assert w.run_episode(21)
 finally:workers.MAX_EPS_STEPS=steps
 rows=w.episode_buffer;assert len(rows[0])==6
 choices=random.sample(list(range(6))*22,128)
 packed=[[column[i] for i in choices] for column in rows]
 dense=[[v.unpack() if isinstance(v,PackedMask) else v for v in column] for column in packed]
 a=[torch.stack(column) for column in dense]
 b=[stack_masks(column) if i in (5,14) else torch.stack(column) for i,column in enumerate(packed)]
 assert all(x.dtype==y.dtype and torch.equal(x,y) for x,y in zip(a,b))
 source=subprocess.check_output(['git','show','main:driver.py'],text=True)
 begin=source.index('                    ### Obtain losses via SAC (Soft Actor-Critic) ###')
 end=source.index('                perf_data = []',begin)
 code=compile(textwrap.dedent(source[begin:end]),'original_sac','exec')
 first,peak_dense=run(a,code);gc.collect();torch.cuda.empty_cache()
 second,peak_packed=run(b,code);gc.collect();torch.cuda.empty_cache()
 max_error=0.;exact=True
 def compare(x,y):
  nonlocal max_error,exact
  if isinstance(x,torch.Tensor):
   assert x.shape==y.shape and x.dtype==y.dtype
   exact=exact and torch.equal(x,y)
   error=float((x-y).abs().max()) if x.numel() else 0.;max_error=max(max_error,error)
   assert torch.allclose(x,y,rtol=1e-5,atol=1e-6),error
  elif isinstance(x,dict):
   assert x.keys()==y.keys()
   for k in x:compare(x[k],y[k])
  elif isinstance(x,(list,tuple)):
   assert len(x)==len(y)
   for u,v in zip(x,y):compare(u,v)
  else:assert x==y
 compare(first,second)
 report=dict(passed=True,all_15_collated_tensors_bitwise_equal=True,effective_batch=128,
  optimizer_iterations_each=8,all_networks_optimizers_and_temperature_equivalent=True,
  all_final_tensors_bitwise_equal=exact,max_absolute_state_error=max_error,
  peak_cuda_dense_mib=peak_dense,peak_cuda_packed_mib=peak_packed,
  scope='Engineering updates on copies with a repeated two-step original-map sample; production checkpoint not changed')
 write_json(Path(__file__).parent/'cuda_equivalence_verified.json',report)
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
