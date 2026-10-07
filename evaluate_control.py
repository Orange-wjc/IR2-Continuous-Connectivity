"""Paired unchanged-weight controls. Baseline is the prior verified 300 runs."""
import argparse, concurrent.futures, csv, hashlib, json, math, multiprocessing, os, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/connectivity_control_20261006'
PILOT=[('corridor',0),('corridor',9),('hybrid',0),('hybrid',1),('complex',0),('complex',5)]
BASE=ROOT/'reproduction_results/pretrained_stage2_20261005'


def evaluate_control(task):
    import numpy as np
    import torch
    import test_parameter
    import test_multi_robot_worker as wm
    from control_env import build_env
    from evaluate_pretrained import evaluate
    from PIL import Image
    mode,kind,index=task
    wm.Env=build_env(mode)
    if not hasattr(evaluate_control,'original_worker'):
        evaluate_control.original_worker=wm.TestWorker
    OriginalWorker=evaluate_control.original_worker
    class AuditedWorker(OriginalWorker):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            evaluate_control.worker=self
            self.paths=[[r.robot_position.tolist()] for r in self.robot_list]
            self.events=[]; self.copy_violations=0
            sensor_original=self.env.update_robot_belief
            step_original=self.env.single_robot_step
            def sensor(position,radius,belief,truth):
                if any(m is not None and np.shares_memory(belief,m) for row in self.env.all_robot_belief for m in row):
                    self.copy_violations+=1
                return sensor_original(position,radius,belief,truth)
            def step(actor,positions,episode,round_id,distance):
                self.paths[actor].append(positions[actor].tolist())
                result=step_original(actor,positions,episode,round_id,distance)
                self.events.append(dict(round=round_id,actor=actor,ok=bool(result[0]),component=next((g for g in self.env.group_ids_list if actor in g),[])))
                if actor==self.n_agent-1 and (round_id+1)%50==0:
                    print(f'{mode}/{kind}/{index}: round {round_id+1}',file=__import__('sys').stderr,flush=True)
                return result
            self.env.update_robot_belief=sensor; self.env.single_robot_step=step
    wm.TestWorker=AuditedWorker
    evaluate_control.worker=None
    row,log=evaluate((kind,index,20261005)); row['mode']=mode
    w=evaluate_control.worker
    if w is None:
        row['independent_checks_pass']=False; row['copy_violations']=None
        return row,log
    personal=np.asarray([w.env.all_robot_belief[i][i] for i in range(w.n_agent)],dtype=np.uint8)
    name=kind+'_'+str(index); destination=OUT/mode
    np.savez_compressed(destination/(name+'_maps.npz'),personal=personal)
    with Image.open(ROOT/'DungeonMaps/test'/kind/row['map_file']) as im:
        free=np.asarray(im.convert('L'))>150
    rates=[int(np.count_nonzero((m==255)&free))/int(np.count_nonzero(free)) for m in personal]
    false_free=[int(np.count_nonzero((m==255)&~free)) for m in personal]
    distances=[math.fsum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(p,p[1:])) for p in w.paths]
    original_rates=json.loads(row['individual_explored']) if row['individual_explored'] else []
    independent=dict(distances=distances,max_distance=max(distances),steps_by_robot=[len(p)-1 for p in w.paths],coverages=rates,false_free=false_free)
    checks=dict(distance=math.isclose(max(distances),row['max_distance_coordinate_units'],rel_tol=1e-12,abs_tol=1e-8),coverage=len(rates)==len(original_rates) and all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(rates,original_rates)),zero_false_free=not any(false_free),success=bool(row['success'])==bool(row['execution_ok'] and all(v>=.99 for v in rates)),copy_isolation=w.copy_violations==0)
    if row['execution_ok']:
        checks['steps']=all(c==int(row['steps']) for c in independent['steps_by_robot'])
        checks['complete_rounds']=len(w.events)==int(row['steps'])*w.n_agent
    row['independent_checks_pass']=all(checks.values()); row['copy_violations']=w.copy_violations
    record=dict(row=row,paths=w.paths,events=w.events,independent=independent,checks=checks,signal_noise=dict(X_g=w.env.ss_realistic_model.X_g,K=w.env.ss_realistic_model.K))
    (destination/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return row,log


def rows_for(mode):
    f=OUT/mode/'episodes.jsonl'
    return [json.loads(l) for l in f.read_text().splitlines() if l] if f.exists() else []


def save_result(row,log):
    directory=OUT/row['mode']; rows=rows_for(row['mode'])
    assert not any(r['map_type']==row['map_type'] and r['map_index']==row['map_index'] for r in rows)
    with (directory/'episodes.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    rows.append(row)
    (directory/(row['map_type']+'_'+str(row['map_index'])+'.log')).write_text(log+str(row['error']))
    with (directory/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(sorted(rows,key=lambda r:(r['map_type'],r['map_index'])))
    print(json.dumps({k:row[k] for k in ['mode','map_type','map_index','execution_ok','success','steps','independent_checks_pass','copy_violations','wall_seconds']}),flush=True)
    if not row['independent_checks_pass']:
        raise RuntimeError('Independent/copy-isolation checks failed; inspect raw record before continuing')


def execute(pool,tasks):
    todo=[]
    for mode,kind,index in tasks:
        if not any(r['map_type']==kind and r['map_index']==index for r in rows_for(mode)):
            todo.append((mode,kind,index))
    pending={pool.submit(evaluate_control,t):t for t in todo}
    for future in concurrent.futures.as_completed(pending):
        save_result(*future.result())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=8); parser.add_argument('--pilot-only',action='store_true'); args=parser.parse_args()
    os.chdir(ROOT)
    for mode in ('both','map_only'): (OUT/mode).mkdir(parents=True,exist_ok=True)
    import test_parameter
    from control_env import patch_text
    baseline=[json.loads(l) for l in (BASE/'episodes.jsonl').read_text().splitlines() if l]
    assert len(baseline)==300 and all(r['execution_ok'] for r in baseline)
    provenance=json.loads((ROOT/'reproduction_results/audit_20261006/audit_provenance.json').read_text())
    for name,digest in provenance['core_sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/test_parameter.MODEL_PATH).read_bytes()).hexdigest()==provenance['checkpoint_sha256']
    for mode in ('both','map_only'): (OUT/(mode+'.patch')).write_text(patch_text(mode))
    manifest=dict(checkpoint_sha256=provenance['checkpoint_sha256'],original_core_sha256=provenance['core_sha256'],baseline=str(BASE),baseline_sha256=hashlib.sha256((BASE/'episodes.jsonl').read_bytes()).hexdigest(),seed=20261005,policy='released Stage2, eval, no_grad, greedy',communication='unchanged original signal-strength physical model',mode_both='copy-on-write at sensor; stale-link prediction from personal belief; unknown=obstacle',mode_map_only='copy-on-write at sensor only',initialization='unchanged original begin; same initial cached beliefs',pilot_cases=PILOT,full_both_target=300,workers=args.workers,code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['control_env.py','evaluate_control.py']},failure_policy='retain all attempts including exploration and execution failures; no retry/substitution')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        execute(pool,[(mode,k,i) for k,i in PILOT for mode in ('both','map_only')])
        (OUT/'pilot_complete.json').write_text(json.dumps(dict(both=len(rows_for('both')),map_only=len(rows_for('map_only')))))
        print('PILOT COMPLETE',flush=True)
        if not args.pilot_only:
            execute(pool,[('both',k,i) for i in range(100) for k in ('corridor','hybrid','complex')])
            assert len(rows_for('both'))==300
            (OUT/'full_complete.json').write_text(json.dumps(dict(both=300,map_only=6)))
            print('FULL 300 COMPLETE',flush=True)
if __name__=='__main__': main()
