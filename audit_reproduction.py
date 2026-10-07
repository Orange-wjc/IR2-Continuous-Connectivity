"""Passive replay capture; independent calculation is in verify_audit.py."""
import os, json, random, contextlib, io, time, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/audit_20261006'
TASKS=[('corridor',0),('corridor',9),('hybrid',0),('hybrid',1),('complex',0),('complex',5)]
def replay(task):
    import numpy as np
    import torch
    import test_parameter as config
    import env, graph_generator, node, graph, test_multi_robot_worker as wm
    from model import PolicyNet
    kind,index=task; seed=20261005+index
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.set_num_threads(1)
    hard=kind=='complex'
    for module in (config,env,graph_generator,node,graph,wm):
        for name,value in dict(TEST_SET_NAME=kind,TEST_SET_DIR='DungeonMaps/test/'+kind,MAX_EPS_STEPS=384 if hard else 196,NODE_COORDS_SCALING_FACTOR=1/1000 if hard else 1/640,GLOBAL_GRAPH_NODE_COORDS_THRESH=340 if hard else 200,SAVE_GIFS=False,VIZ_GRAPH_EDGES=False).items(): setattr(module,name,value)
    policy=PolicyNet(config.INPUT_DIM,config.EMBEDDING_DIM).to('cuda')
    policy.load_state_dict(torch.load(config.MODEL_PATH,map_location='cpu',weights_only=True)['policy_model']); policy.eval()
    w=wm.TestWorker(0,5 if hard else 4,policy,index,device='cuda',greedy=True,save_image=False)
    paths=[[r.robot_position.tolist()] for r in w.robot_list]; events=[]; leak_events=[]
    original_step=w.env.single_robot_step; original_sensor=w.env.update_robot_belief
    state={}
    def sensor(position,radius,belief,truth):
        actor=state['actor']
        shared=[j for j in range(w.n_agent) if j!=actor and np.shares_memory(belief,w.env.all_robot_belief[j][j])]
        snapshots={j:w.env.all_robot_belief[j][j].copy() for j in shared}
        result=original_sensor(position,radius,belief,truth)
        state['changes']=[dict(observer=j,changed_pixels=int(np.count_nonzero(snapshots[j]!=w.env.all_robot_belief[j][j])),new_free=int(np.count_nonzero((snapshots[j]!=255)&(w.env.all_robot_belief[j][j]==255)))) for j in shared]
        return result
    def step(actor,positions,episode,round_id,distance):
        state['actor']=actor; state['changes']=[]
        paths[actor].append(positions[actor].tolist())
        result=original_step(actor,positions,episode,round_id,distance)
        component=next((g for g in w.env.group_ids_list if actor in g),[])
        for change in state['changes']:
            if change['observer'] not in component and change['changed_pixels']:
                leak_events.append(dict(round=round_id,actor=actor,component=component,**change))
        events.append(dict(round=round_id,actor=actor,ok=bool(result[0])))
        return result
    w.env.update_robot_belief=sensor; w.env.single_robot_step=step
    start=time.monotonic(); log=io.StringIO()
    with contextlib.redirect_stdout(log),torch.no_grad(): ok=w.work(index)
    name=kind+'_'+str(index)
    np.savez_compressed(OUT/(name+'_maps.npz'),personal=np.asarray([w.env.all_robot_belief[j][j] for j in range(w.n_agent)],dtype=np.uint8))
    result=dict(map_type=kind,map_index=index,map_file=w.env.file_path,seed=seed,execution_ok=bool(ok),paths=paths,events=events,original_metrics=w.perf_metrics,original_individual_explored=[w.env.evaluate_exploration_rate(j) for j in range(w.n_agent)],map_alias_leak_events=leak_events,wall_seconds=time.monotonic()-start)
    (OUT/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (OUT/(name+'.log')).write_text(log.getvalue())
    return dict(name=name,ok=bool(ok),steps=w.perf_metrics.get('travel_steps'),alias_leak_events=len(leak_events),seconds=result['wall_seconds'])
def main():
    os.chdir(ROOT); OUT.mkdir(parents=True,exist_ok=True)
    tasks=[t for t in TASKS if not (OUT/(t[0]+'_'+str(t[1])+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(replay,t) for t in tasks]
        for f in as_completed(futures): print(json.dumps(f.result()),flush=True)
if __name__=='__main__': main()
