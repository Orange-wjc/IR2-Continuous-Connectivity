"""Controlled tests execute original propagation/erasure and observation functions.
Graph regeneration is stubbed only in isolated communication fixtures; sensor and
link stubs are specified per test. Originals on disk are never modified.
"""
import copy, json, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import test_parameter as config
from control_env import build_env
import sys
MODE=sys.argv[1] if len(sys.argv)>1 else 'both'
Env=build_env(MODE)
from sensor import sensor_work
from ss_realistic_model import SS_realistic_model
from test_multi_robot_worker import TestWorker
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/connectivity_control_20261006/unit_tests'/MODE
random.seed(20261006); np.random.seed(20261006); torch.manual_seed(20261006)
base=Env(0,4,k_size=30,plot=False)

def fixture():
    e=copy.deepcopy(base)
    e.ground_truth=np.full((500,500),255,dtype=np.uint8)
    e.agents_merged_belief=np.full((500,500),127,dtype=np.uint8)
    e.all_robot_belief=[[np.full((500,500),127,dtype=np.uint8) for j in range(4)] for i in range(4)]
    for i in range(4): e.all_robot_belief[i][i][200,200+i]=255
    e.all_robot_positions_belief=[[None]*4 for _ in range(4)]
    for i,p in enumerate([(50,50),(80,50),(110,50),(450,450)]): e.all_robot_positions_belief[i][i]=np.array(p)
    e.all_rendezvous_utility_inputs=[np.zeros((len(c),1)) for c in e.all_node_coords]
    e.update_graph=lambda *a,**kw: True
    e.find_frontier=lambda belief: np.empty((0,2))
    e.calculate_num_observed_frontiers=lambda *a:0
    e.update_robot_belief=lambda position,radius,belief,truth: belief
    return e

def run_step(e,actor,positions,step):
    e.all_robot_positions_belief[actor][actor]=positions[actor]
    return e.single_robot_step(actor,positions,0,step,0)

def observe(e,position):
    w=TestWorker.__new__(TestWorker); w.env=e; w.device='cpu'; w.node_padding_size=config.NODE_PADDING_SIZE; w.k_size=config.K_SIZE
    tensors,ok=w.get_observations(position,0,0,0,plot=False)
    assert ok
    return tensors

def same(a,b):
    return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(a,b))
results=[]
# A-B-C is a chain, D isolated. Original transitive map propagation must work.
e=fixture(); positions=[np.array(p) for p in [(50,50),(80,50),(110,50),(450,450)]]
e.ss_realistic_model=SimpleNamespace(is_within_signal_strength=lambda truth,a,b: np.linalg.norm(a-b)<40)
run_step(e,0,positions,1)
results.append(dict(test='multi_hop_chain',passed=all(e.all_robot_belief[i][i][200,200+j]==255 for i in range(3) for j in range(3)) and e.all_robot_belief[0][0][200,203]==127,components=e.group_ids_list))
# Independently copied snapshots remove the cross-agent write in a control.
control=copy.deepcopy(e)
control.all_robot_belief=[[m.copy() if m is not None else None for m in row] for row in control.all_robot_belief]
control.ss_realistic_model=SimpleNamespace(is_within_signal_strength=lambda *args:False)
def control_stamp(position,radius,belief,truth): belief[300,300]=255; return belief
control.update_robot_belief=control_stamp
control_before=control.all_robot_belief[1][1].copy(); run_step(control,0,positions,2)
results.append(dict(test='independent_snapshot_copy_control',passed=bool(np.array_equal(control_before,control.all_robot_belief[1][1])),scope='in-memory test fixture only; core source unchanged'))
# Cached information about D may legitimately be relayed by C through B to A.
forward=fixture(); forward.ss_realistic_model=SimpleNamespace(is_within_signal_strength=lambda truth,a,b:np.linalg.norm(a-b)<40)
forward.all_robot_belief[2][3][250,250]=255; forward.all_robot_belief_step_updated[2][3]=9
forward.all_robot_positions_belief[2][3]=np.array([170,50]); forward.all_robot_positions_step_updated[2][3]=9
run_step(forward,0,positions,10)
results.append(dict(test='multi_hop_fresher_cached_information',passed=bool(forward.all_robot_belief[0][3][250,250]==255 and forward.all_robot_belief_step_updated[0][3]==9 and np.array_equal(forward.all_robot_positions_belief[0][3],np.array([170,50])) and forward.all_robot_positions_step_updated[0][3]==9),cached_position=[170,50],actual_disconnected_position=positions[3].tolist()))
# Same arrays persist across a loss of the physical communication edge.
shared_before=bool(np.shares_memory(e.all_robot_belief[0][0],e.all_robot_belief[1][1]))
before=e.all_robot_belief[1][1].copy(); cached_before=e.all_robot_belief[1][0].copy()
e.ss_realistic_model=SimpleNamespace(is_within_signal_strength=lambda *args:False)
def stamp(position,radius,belief,truth): belief[300,300]=255; return belief
e.update_robot_belief=stamp
run_step(e,0,positions,2)
changed=int(np.count_nonzero(before!=e.all_robot_belief[1][1]))
results.append(dict(test='disconnect_transition_map_isolation',passed=changed==0,shared_before=shared_before,observer_new_pixels=changed,observer_cached_map_changed=int(np.count_nonzero(cached_before!=e.all_robot_belief[1][0])),components=e.group_ids_list,expected='observer receives zero new cells after all links are broken'))
# With independently allocated starting maps no cross-agent write should occur.
e=fixture(); e.ss_realistic_model=SimpleNamespace(is_within_signal_strength=lambda *args:False); e.update_robot_belief=stamp
before=e.all_robot_belief[1][1].copy(); run_step(e,0,positions,2)
results.append(dict(test='already_disconnected_independent_maps',passed=bool(np.array_equal(before,e.all_robot_belief[1][1]))))
# Current true remote poses are never forwarded across a disconnected component.
e=fixture(); e.ss_realistic_model=SimpleNamespace(is_within_signal_strength=lambda *args:False)
e.all_robot_positions_belief[0][1]=np.array([80,50]); old=e.all_robot_positions_belief[0][1].copy()
positions[1]=np.array([300,300]); run_step(e,0,positions,2)
results.append(dict(test='disconnected_true_pose_not_forwarded',passed=bool(np.array_equal(old,e.all_robot_positions_belief[0][1]))))
# Actual model observations with fixed local/cached knowledge; remote partition,
# remote private maps, true poses and global merged map are varied.
a=copy.deepcopy(base); a.group_ids_list=[[0],[1,2,3]]
b=copy.deepcopy(a); b.group_ids_list=[[0],[1],[2],[3]]
b.all_robot_positions_gt=[np.array([400,400]) for _ in range(4)]
b.agents_merged_belief=np.full_like(b.ground_truth,255)
for i in range(1,4): b.all_robot_belief[i][i]=np.full_like(b.ground_truth,255)
obs_a=observe(a,a.start_position); obs_b=observe(b,b.start_position)
results.append(dict(test='observation_remote_private_state_and_partition_invariance',passed=same(obs_a,obs_b),all_six_observation_tensors_identical=same(obs_a,obs_b)))
# Hidden wall lies outside ego's 80-pixel lidar range. Its presence affects a
# hypothetical link to a stale belief, despite identical current link outcomes.
a=fixture(); b=copy.deepcopy(a)
truth=np.full((500,500),255,dtype=np.uint8); a.ground_truth=truth.copy(); b.ground_truth=truth.copy()
start=base.start_position.copy(); x,y=map(int,start); b.ground_truth[y-1:y+2,x+90:x+100]=1
positions=[start]+[np.array(p) for p in [(470,470),(470,460),(460,470)]]
stale=start+np.array([100,0]); ss=SS_realistic_model(X_g_min=5,X_g_max=5,K_min=5,K_max=5)
for e in (a,b):
    e.ss_realistic_model=copy.deepcopy(ss); e.update_robot_belief=Env.update_robot_belief.__get__(e,Env)
    for i in range(4): e.all_robot_positions_belief[i][i]=positions[i]
    e.all_robot_positions_belief[0][1]=stale.copy()
    e.all_robot_positions_missing_counts[0][1]=config.REMOVE_POSE_BELIEF_MISSING_COUNT-1
sensor_a=sensor_work(positions[0],80,np.full((500,500),127,dtype=np.uint8),a.ground_truth)
sensor_b=sensor_work(positions[0],80,np.full((500,500),127,dtype=np.uint8),b.ground_truth)
actual_links_a=[ss.is_within_signal_strength(a.ground_truth,positions[0],p) for p in positions[1:]]
actual_links_b=[ss.is_within_signal_strength(b.ground_truth,positions[0],p) for p in positions[1:]]
belief_link_a=ss.is_within_signal_strength(a.ground_truth,positions[0],stale); belief_link_b=ss.is_within_signal_strength(b.ground_truth,positions[0],stale)
run_step(a,0,positions,2); run_step(b,0,positions,2)
removed_a=a.all_robot_positions_belief[0][1] is None; removed_b=b.all_robot_positions_belief[0][1] is None
# Observe with identical local graph and zero surplus: tests the p feature alone.
for e in (a,b): e.generate_rendezvous_utility_layer=lambda robot_id,eps:(np.zeros(4),np.zeros((len(base.all_node_coords[0]),1)),True)
oa=observe(a,positions[0]); ob=observe(b,positions[0]); features_changed=int(torch.count_nonzero(oa[0]!=ob[0]).item())
results.append(dict(test='hidden_terrain_stale_pose_erasure_noninterference',passed=removed_a==removed_b and same(oa,ob),noise_X_g=ss.X_g,noise_K=ss.K,lidar_identical=bool(np.array_equal(sensor_a,sensor_b)),actual_ego_links_a=actual_links_a,actual_ego_links_b=actual_links_b,truth_reference_stale_link_a=belief_link_a,truth_reference_stale_link_b=belief_link_b,local_stale_link_a=ss.is_within_signal_strength(a.all_robot_belief[0][0],positions[0],stale),local_stale_link_b=ss.is_within_signal_strength(b.all_robot_belief[0][0],positions[0],stale),pose_removed_a=removed_a,pose_removed_b=removed_b,observation_feature_cells_changed=features_changed,scope='real sensor, signal-strength, propagation and erasure; graph update stubbed, surplus fixed to zero to isolate position feature'))
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'information_tests.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,default=lambda x:x.item()))
print(json.dumps(results,indent=2,ensure_ascii=False,default=lambda x:x.item()))
