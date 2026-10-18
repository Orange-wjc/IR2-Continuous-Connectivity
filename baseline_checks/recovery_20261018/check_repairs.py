"""Reproduce graph overflow and verify lossless full-capacity replay storage."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-repair-mpl');os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import sys,copy,json,importlib.util,io,contextlib,hashlib,random,resource,gc,zlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import parameter as config
import numpy as np
import torch
from graph_generator import Graph_generator
from graph import Graph
from baseline_replay import PackedMask,pack_mask_rows,stack_masks
from baseline_support import prepare_maps,write_json
from robot import Robot
WORK=Path(__file__).resolve().parent
ARCHIVE=ROOT/'experiments/ir2_baseline_easy10_20261018/before_memory_graph_repair_20261018'

def old_module(name,filename):
 spec=importlib.util.spec_from_file_location(name,ARCHIVE/filename)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def fixture(cls,count,custom=False):
 origin=np.array([200.,200.]);angles=np.arange(30)*2*np.pi/30
 far=origin+np.c_[100*np.cos(angles),100*np.sin(angles)]
 nodes=np.vstack([origin,far])
 global_graph=Graph();global_graph.add_node(tuple(origin))
 for p in far[:count]:global_graph.add_edge(tuple(origin),tuple(p),100.)
 g=cls.__new__(cls);g.k_size=30;g.graph=Graph();g.plot=False;g.check_collision=lambda *a:False
 if custom:indices=g.find_k_neighbor_custom(nodes,origin,None,g.graph,global_graph=global_graph,max_edge_len=200)
 else:indices=g.find_k_neighbor(nodes,origin,None,global_graph=global_graph)
 return nodes,g.graph,indices

def observation_from_graph(nodes,graph):
 from multi_robot_worker import Worker
 from types import SimpleNamespace
 n=len(nodes)
 for p in nodes:graph.edges.setdefault(tuple(p),{})
 env=SimpleNamespace(all_robot_map_belief_area_diff=[None]*3,all_rendezvous_utility_inputs=[None]*3,
  all_node_coords=[nodes]*3,all_graph=[graph.edges]*3,all_node_utility=[np.ones(n)]*3,
  all_guidepost=[np.zeros((n,1))]*3,all_robot_positions_belief=[[nodes[0]]*3]*3,
  all_curr_vertices=[None]*3,
  generate_rendezvous_utility_layer=lambda *a:(0.,np.zeros((n,1)),True),
  find_index_from_coords=lambda p,robot_id:int(np.argmin(np.linalg.norm(nodes-p,axis=1))))
 worker=Worker.__new__(Worker);worker.env=env;worker.device='cpu';worker.k_size=30;worker.node_padding_size=360
 return worker.get_observations(nodes[0],0,21,46)

def main():
 torch.set_num_threads(1);torch.manual_seed(73);np.random.seed(73);random.seed(73)
 before=old_module('ir2_graph_before','graph_generator.py').Graph_generator
 checks={};graph_details=[]
 for custom in [False,True]:
  for count in [0,5,29,30]:
   nodes,old,old_indices=fixture(before,count,custom)
   _,new,new_indices=fixture(Graph_generator,count,custom)
   a=list(old.edges[tuple(nodes[0])]);b=list(new.edges[tuple(nodes[0])])
   if count<30:assert a==b and old_indices==new_indices
   else:
    assert len(a)==31 and len(b)==30 and set(b)==set(a)-{tuple(nodes[0])}
    with contextlib.redirect_stdout(io.StringIO()) as out:old_obs,old_ok=observation_from_graph(nodes,old)
    assert not old_ok and 'Not (edge_inputs.shape' in out.getvalue()
    new_obs,new_ok=observation_from_graph(nodes,new);assert new_ok
    assert new_obs[1].shape==new_obs[4].shape==(1,1,30)
   graph_details.append(dict(custom=custom,global_neighbors=count,old_degree=len(a),new_degree=len(b)))
 checks['graph_zero_budget_reproduced_and_fixed']=True
 checks['legal_graphs_unchanged']=True
 # Exact storage, dtype and signed-zero/nonbinary fallback.
 for tensor in [torch.randint(0,2,(360,360)).float(),torch.ones(360,360),
                torch.tensor([[0.,-0.,.125],[1.,float('nan'),float('inf')]],dtype=torch.float64)]:
  packed=PackedMask.from_tensor(tensor);decoded=packed.unpack()
  assert tensor.dtype==decoded.dtype and tensor.shape==decoded.shape
  assert tensor.numpy().tobytes()==decoded.numpy().tobytes()
 checks['mask_roundtrip_bitwise_all_values_and_dtypes']=True
 # Use actual original-map observations to compare all15 replay columns.
 prepare_maps(config)
 from env import Env
 from model import PolicyNet,QNet
 from multi_robot_worker import Worker
 worker=Worker(0,3,PolicyNet(6,128),QNet(6,128),1,device='cpu',save_image=False)
 obs,valid=worker.get_observations(worker.robot_list[0].robot_position,0,1,0);assert valid
 old_robot=old_module('ir2_robot_before','robot.py').Robot(0,np.zeros(2))
 new_robot=Robot(0,np.zeros(2))
 for robot in [old_robot,new_robot]:
  robot.save_observations(obs);robot.save_action(torch.tensor([0]));robot.save_reward_done(1.25,False);robot.save_next_observations(obs)
 dense=[column[0] for column in old_robot.episode_buffer]
 packed=[column[0] for column in new_robot.episode_buffer]
 for a,b in zip(dense,packed):
  b=b.unpack() if isinstance(b,PackedMask) else b
  assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 assert torch.equal(stack_masks([dense[5],packed[5]]),torch.stack([dense[5],dense[5]]))
 checks['all_15_replay_columns_unchanged_on_original_png']=True
 checks['dense_and_packed_batch_collation_identical']=True
 mask_bytes=2*dense[5].numel()*dense[5].element_size()
 dense_row_bytes=sum(x.numel()*x.element_size() for x in dense)
 packed_mask_bytes=len(packed[5].payload)+len(packed[14].payload)
 # Fill all10000 slots with independently owned tensors/mask payloads. Avoid
 # aliases that would make a small repeated list look artificially efficient.
 replay=[[] for _ in range(15)]
 rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 for i in range(10000):
  for column,x in zip(replay,packed):
   value=PackedMask(x.shape,x.dtype,x.binary,memoryview(x.payload).tobytes()) if isinstance(x,PackedMask) else x.clone()
   column.append(value)
 assert all(len(x)==10000 for x in replay)
 indices=random.sample(range(10000),128)
 for c in [5,14]:assert torch.equal(stack_masks([replay[c][i] for i in indices]),torch.stack([dense[c]]*128))
 rss_after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 assert rss_after-rss_before<2048,(rss_before,rss_after)
 checks['full_10000_replay_with_distinct_storage_under_2gib_growth']=True
 # Worst-case incompressible binary masks still have a strict small payload.
 random_mask=torch.randint(0,2,(360,360)).float();worst=PackedMask.from_tensor(random_mask)
 assert len(worst.payload)<16300
 report=dict(passed=all(checks.values()),checks=checks,graph_details=graph_details,
  memory=dict(legacy_mask_gib_10000=mask_bytes*10000/2**30,
   legacy_total_tensor_gib_10000=dense_row_bytes*10000/2**30,
   real_mask_payload_bytes_per_transition=packed_mask_bytes,
   real_stored_tensor_and_payload_gib_10000=(dense_row_bytes-mask_bytes+packed_mask_bytes)*10000/2**30,
   random_dense_binary_mask_payload_bytes=len(worst.payload),
   full_replay_peak_rss_growth_mib=rss_after-rss_before,peak_rss_mib=rss_after),
  scope='Boundary regression, original-map replay equality and full10000-slot host-memory engineering check; no production gradient or rollout')
 write_json(WORK/'repairs_verified.json',report)
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
