"""Frozen fixed-weight evaluation on geometry-disjoint Complex layouts."""
import argparse,concurrent.futures,contextlib,csv,hashlib,io,json,math,multiprocessing,os,random,time,traceback
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/independent_20261006'
METHODS=('ir2','frontier_rendezvous')
CONDITIONS=[dict(name='low',X_g=0.0,K=0.0),dict(name='medium',X_g=6.5,K=6.5),dict(name='high',X_g=13.0,K=13.0)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_protocol():
    from audit_map_split import fingerprints
    source=ROOT/'reproduction_results/credibility_20261006/geometry_disjoint_candidate_manifest.json'
    groups=json.loads(source.read_text())['complex']
    audit=json.loads((source.parent/'map_split_audit.json').read_text())
    assert len(groups)==30 and len({r['layout_sha256'] for r in groups})==30
    eligible=set(audit['eligible_geometry_disjoint_test_maps'])
    order=sorted(os.listdir(ROOT/'DungeonMaps/test/complex'),reverse=True)
    layouts=[]
    for i,group in enumerate(groups):
        path=ROOT/group['representative'];assert group['representative'] in eligible
        assert fingerprints(path)[1]==group['layout_sha256']
        layouts.append(dict(layout_index=i,layout_sha256=group['layout_sha256'],map_path=group['representative'],map_sha256=sha(path),map_index=order.index(path.name)))
    provenance=json.loads((ROOT/'reproduction_results/audit_20261006/audit_provenance.json').read_text())
    for name,digest in provenance['core_sha256'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'model/stage2/checkpoint.pth')==provenance['checkpoint_sha256']
    protocol=dict(layouts=layouts,conditions=CONDITIONS,methods=METHODS,target_episodes=180,num_robots=5,step_limit=384,success='every personal belief >=99% ground-truth free area',seed_base=20261006,device='cuda',ir2='released Stage2; strict load; eval/no_grad/greedy; both information-isolation corrections',baseline='Dijkstra to nearest reachable positive-frontier-utility node; else nearest reachable known peer indicator; else least-visited legal non-self neighbor (oldest visit, distance, index ties). Uses observation and own route history only; not paper Pursuit.',noise='X_g and K fixed together at 0,6.5,13; sensitivity conditions, not random independent noise draws',initialization='original starting marker and original robot placement; identical within every pair',units='map coordinates; original physical-link model unchanged; meter/radio calibration remains unresolved',failure_policy='all attempts retained; no retries, map substitutions or outcome-based tuning',primary='paired success rates including execution failures; report each condition',secondary='steps including exploration timeouts; distance comparisons primarily on jointly successful pairs; separately show timeout distances',stability_rule='predeclared engineering screening criterion: >=24/30 successes in each condition and max-min <=4/30; only for tested communication conditions, not a universal standard',statistical_plan='paired differences; bootstrap resample layout groups, keeping all three conditions together; 5000 samples fixed RNG; no inference about training variability',limitations='only exact/rotation/reflection geometry exclusion from provided training directory; cannot prove checkpoint training history; no near-duplicate audit; some layouts occurred in prior descriptive tests, so this is a validation set, not a future tuning holdout',checkpoint_sha256=provenance['checkpoint_sha256'],core_sha256=provenance['core_sha256'],candidate_manifest_sha256=sha(source),code_sha256={name:sha(ROOT/name) for name in ['control_env.py','independent_policies.py','evaluate_independent.py']})
    OUT.mkdir(parents=True,exist_ok=True)
    protocol['methods']=list(protocol['methods'])
    target=OUT/'protocol.json'
    if target.exists():assert json.loads(target.read_text())==protocol,'Frozen protocol or inputs changed'
    else:
        target.write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
        (OUT/'protocol.sha256').write_text(sha(target)+'\n')
    return protocol


def evaluate(task):
    import numpy as np
    import torch
    import test_parameter as config
    import env,graph_generator,node,graph
    import test_multi_robot_worker as wm
    from control_env import build_env
    from independent_policies import frontier_rendezvous
    from model import PolicyNet
    from PIL import Image
    layout,condition,method=task
    seed=20261006+layout['layout_index']*10+CONDITIONS.index(condition)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(1)
    overrides=dict(TEST_SET_NAME='complex',TEST_SET_DIR='DungeonMaps/test/complex',MAX_EPS_STEPS=384,NODE_COORDS_SCALING_FACTOR=1/1000,GLOBAL_GRAPH_NODE_COORDS_THRESH=340,SAVE_GIFS=False,VIZ_GRAPH_EDGES=False,SS_XG_MIN=condition['X_g'],SS_XG_MAX=condition['X_g'],SS_K_MIN=condition['K'],SS_K_MAX=condition['K'])
    for module in (config,env,graph_generator,node,graph,wm):
        for name,value in overrides.items():setattr(module,name,value)
    if not hasattr(evaluate,'original_worker'):evaluate.original_worker=wm.TestWorker
    wm.Env=build_env('both')
    policy=None
    if method=='ir2':
        if not hasattr(evaluate,'policy'):
            evaluate.policy=PolicyNet(config.INPUT_DIM,config.EMBEDDING_DIM).to('cuda')
            checkpoint=torch.load(config.MODEL_PATH,map_location='cpu',weights_only=True)
            evaluate.policy.load_state_dict(checkpoint['policy_model'],strict=True);evaluate.policy.eval()
        policy=evaluate.policy
    class AuditedWorker(evaluate.original_worker):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert self.env.file_path==Path(layout['map_path']).name
            self.paths=[[r.robot_position.tolist()] for r in self.robot_list]
            self.initial_positions=[p[0] for p in self.paths]
            self.events=[];self.copy_violations=0;self.action_checks=0;self.reason_counts=Counter()
            self.visits=[{} for _ in range(self.n_agent)];self.last_visit=[{} for _ in range(self.n_agent)];self.action_counts=[0]*self.n_agent
            original_sensor=self.env.update_robot_belief;original_step=self.env.single_robot_step
            def sensor(position,radius,belief,truth):
                if any(m is not None and np.shares_memory(belief,m) for row in self.env.all_robot_belief for m in row):self.copy_violations+=1
                return original_sensor(position,radius,belief,truth)
            def step(actor,positions,episode,round_id,distance):
                self.paths[actor].append(positions[actor].tolist())
                result=original_step(actor,positions,episode,round_id,distance)
                components=[list(g) for g in self.env.group_ids_list]
                self.events.append(dict(round=round_id,actor=actor,ok=bool(result[0]),components=components))
                if actor==self.n_agent-1 and (round_id+1)%50==0:print(f"{method}/{layout['layout_index']}/{condition['name']}: round {round_id+1}",flush=True)
                return result
            self.env.update_robot_belief=sensor;self.env.single_robot_step=step

        def select_node(self,observations,robot_id):
            inputs,edges,current,padding,edge_padding,mask=observations
            assert bool(torch.isfinite(inputs).all()) and bool(torch.isfinite(mask).all()),'Nonfinite observation'
            neighbor_indices=edges[0,0].detach().cpu().numpy()
            valid_slots=np.flatnonzero(edge_padding[0,0].detach().cpu().numpy()==0)
            assert len(valid_slots)>0,'All candidate actions masked'
            current_id=int(current.item());legal=neighbor_indices[valid_slots].astype(int)
            assert np.all((legal>=0)&(legal<inputs.shape[1]))
            assert all(mask[0,current_id,int(i)].item()==0 for i in legal),'Legal slot absent from observed graph'
            if method=='ir2':
                logp=self.local_policy_net(inputs,edges,current,padding,edge_padding,mask)
                assert tuple(logp.shape)==(1,self.k_size) and bool(torch.isfinite(logp).all()),'Nonfinite policy output'
                slot=int(torch.argmax(logp,dim=1).item());reason='policy'
            else:
                features=inputs[0].detach().cpu().numpy();adjacency=mask[0].detach().cpu().numpy()
                here=tuple(float(v) for v in features[current_id,:2])
                if not self.visits[robot_id]:self.visits[robot_id][here]=1;self.last_visit[robot_id][here]=0
                hop,reason=frontier_rendezvous(features,adjacency,current_id,legal,self.visits[robot_id],self.last_visit[robot_id])
                slot=int(next(i for i in valid_slots if int(neighbor_indices[i])==hop))
                key=tuple(float(v) for v in features[hop,:2]);self.action_counts[robot_id]+=1
                self.visits[robot_id][key]=self.visits[robot_id].get(key,0)+1;self.last_visit[robot_id][key]=self.action_counts[robot_id]
            assert slot in valid_slots,'Selected padded action'
            self.action_checks+=1;self.reason_counts[reason]+=1
            index=int(neighbor_indices[slot])
            return self.env.all_node_coords[robot_id][index],torch.tensor([slot],dtype=torch.long,device=self.device)
    wm.TestWorker=AuditedWorker
    identifier=f"{layout['layout_index']:02d}_{condition['name']}_{method}"
    row=dict(case_id=identifier,layout_index=layout['layout_index'],layout_sha256=layout['layout_sha256'],map_path=layout['map_path'],condition=condition['name'],X_g=condition['X_g'],K=condition['K'],method=method,seed=seed,num_robots=5,step_limit=384,execution_ok=False,success=False,error='')
    worker=None;capture=io.StringIO();started=time.monotonic()
    try:
        with contextlib.redirect_stdout(capture),torch.no_grad():
            worker=AuditedWorker(0,5,policy,layout['map_index'],device='cuda',greedy=True,save_image=False)
            row['execution_ok']=bool(worker.work(layout['map_index']))
        if not row['execution_ok']:row['error']='Original worker returned False'
    except Exception:row['error']=traceback.format_exc()
    row['wall_seconds']=round(time.monotonic()-started,3)
    if worker is None:
        row['independent_checks_pass']=False
        return row,capture.getvalue()
    personal=np.asarray([worker.env.all_robot_belief[i][i] for i in range(5)],dtype=np.uint8)
    with Image.open(ROOT/layout['map_path']) as im:free=np.asarray(im.convert('L'))>150
    free_count=int(np.count_nonzero(free));rates=[int(np.count_nonzero((m==255)&free))/free_count for m in personal]
    false_free=[int(np.count_nonzero((m==255)&~free)) for m in personal]
    distances=[math.fsum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(p,p[1:])) for p in worker.paths]
    path_steps=[len(p)-1 for p in worker.paths];steps=worker.perf_metrics.get('travel_steps',min(path_steps))
    original_rates=[worker.env.evaluate_exploration_rate(i) for i in range(5)]
    connected=sum(len(e['components'])==1 for e in worker.events)
    row.update(steps=int(steps),max_distance_coordinate_units=max(distances),min_explored=min(rates),mean_explored=sum(rates)/5,individual_explored=rates,success=bool(row['execution_ok'] and all(v>=.99 for v in rates)),copy_violations=worker.copy_violations,action_checks=worker.action_checks,full_connectivity_fraction_per_action=connected/len(worker.events) if worker.events else None,final_connected=bool(worker.env.connectivity_rate))
    checks=dict(distance=all(math.isclose(a,r.travel_dist,rel_tol=1e-12,abs_tol=1e-8) for a,r in zip(distances,worker.robot_list)),coverage=all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(rates,original_rates)),no_false_free=not any(false_free),copy_isolation=worker.copy_violations==0,noise=worker.env.ss_realistic_model.X_g==condition['X_g'] and worker.env.ss_realistic_model.K==condition['K'])
    if row['execution_ok']:checks.update(steps=all(s==steps for s in path_steps),complete_rounds=len(worker.events)==steps*5,success=bool(worker.env.check_done())==row['success'])
    row['independent_checks_pass']=all(checks.values())
    record=dict(row=row,checks=checks,initial_positions=worker.initial_positions,paths=worker.paths,events=worker.events,independent=dict(distances=distances,coverages=rates,false_free=false_free,path_steps=path_steps,free_count=free_count),baseline_decision_counts=dict(worker.reason_counts))
    np.savez_compressed(OUT/'raw'/(identifier+'_maps.npz'),personal=personal)
    (OUT/'raw'/(identifier+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return row,capture.getvalue()


def read_rows():
    p=OUT/'episodes.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def execute(pool,tasks):
    completed={r['case_id'] for r in read_rows()}
    todo=[t for t in tasks if f"{t[0]['layout_index']:02d}_{t[1]['name']}_{t[2]}" not in completed]
    pending=[pool.submit(evaluate,t) for t in todo]
    for future in concurrent.futures.as_completed(pending):
        row,log=future.result();assert row['case_id'] not in completed;completed.add(row['case_id'])
        (OUT/'logs'/(row['case_id']+'.log')).write_text(log+row['error'])
        with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps({k:row.get(k) for k in ['case_id','execution_ok','success','steps','independent_checks_pass','wall_seconds']}),flush=True)
        if not row['independent_checks_pass']:raise RuntimeError('Independent accounting failed; stop and inspect')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--pilot-only',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);protocol=freeze_protocol()
    (OUT/'raw').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        execute(pool,[(protocol['layouts'][0],condition,method) for condition in CONDITIONS for method in METHODS])
        (OUT/'pilot_complete.json').write_text(json.dumps(dict(episodes=len(read_rows()))));print('PILOT COMPLETE',flush=True)
        if not args.pilot_only:execute(pool,[(layout,condition,method) for layout in protocol['layouts'] for condition in CONDITIONS for method in METHODS])
    if not args.pilot_only:
        rows=read_rows();assert len(rows)==180 and len({r['case_id'] for r in rows})==180
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with (OUT/'episodes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r:r['case_id']))
        (OUT/'complete.json').write_text(json.dumps(dict(episodes=180)));print('FULL 180 COMPLETE',flush=True)


if __name__=='__main__':main()
