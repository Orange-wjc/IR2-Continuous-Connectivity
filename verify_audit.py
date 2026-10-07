"""Independent reader: no imports from IR2 model, Env, Worker or evaluators."""
import json, math, statistics, hashlib, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'reproduction_results/pretrained_stage2_20261005'
OUT=ROOT/'reproduction_results/audit_20261006'
def close(a,b): return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9)
def main():
    rows=[json.loads(line) for line in (OLD/'episodes.jsonl').read_text().splitlines() if line]
    summary=json.loads((OLD/'summary.json').read_text()); aggregate={}
    for kind in ('corridor','hybrid','complex'):
        group=[r for r in rows if r['map_type']==kind]; valid=[r for r in group if r['execution_ok']]
        successes=sum(r['success'] for r in group)
        steps=[float(r['steps']) for r in valid]; distances=[float(r['max_distance_coordinate_units']) for r in valid]
        checks=[len(group)==100,len({r['map_index'] for r in group})==100,successes==summary[kind]['successful'],close(statistics.mean(steps),summary[kind]['steps_execution_ok']['mean']),close(statistics.pstdev(steps),summary[kind]['steps_execution_ok']['std_population']),close(statistics.mean(distances),summary[kind]['distance_execution_ok_coordinate_units']['mean']),close(statistics.pstdev(distances),summary[kind]['distance_execution_ok_coordinate_units']['std_population'])]
        checks += [bool(r['success'])==all(float(v)>=.99 for v in json.loads(r['individual_explored'])) for r in valid]
        checks += [int(r['steps'])==int(r['step_limit']) for r in valid if not r['success']]
        aggregate[kind]=dict(passed=all(checks),count=len(group),successes=successes,failures=len(group)-successes,steps_mean=statistics.mean(steps),distance_mean_coordinate_units=statistics.mean(distances),distance_std_population=statistics.pstdev(distances))
    replay_results=[]
    for kind,index in [('corridor',0),('corridor',9),('hybrid',0),('hybrid',1),('complex',0),('complex',5)]:
        name=kind+'_'+str(index)
        if '--partial' in sys.argv and not (OUT/(name+'.json')).exists(): continue
        record=json.loads((OUT/(name+'.json')).read_text())
        paths=record['paths']; individual_distances=[math.fsum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(p,p[1:])) for p in paths]
        action_counts=[len(p)-1 for p in paths]; steps=max(action_counts)
        personal=np.load(OUT/(name+'_maps.npz'))['personal']
        map_path=ROOT/'DungeonMaps/test'/kind/record['map_file']
        with Image.open(map_path) as image: free=np.asarray(image.convert('L'))>150
        denominator=int(np.count_nonzero(free))
        coverages=[int(np.count_nonzero((m==255)&free))/denominator for m in personal]
        false_free=[int(np.count_nonzero((m==255)&~free)) for m in personal]
        success=all(v>=.99 for v in coverages); metrics=record['original_metrics']
        old=next(r for r in rows if r['map_type']==kind and int(r['map_index'])==index)
        checks=dict(execution_ok=record['execution_ok'],distance_matches_original=close(max(individual_distances),metrics['travel_dist']),steps_match_original=steps==metrics['travel_steps'],every_robot_actions_equal_rounds=all(c==steps for c in action_counts),coverage_matches_original=all(close(a,b) for a,b in zip(coverages,record['original_individual_explored'])),zero_false_free_cells=not any(false_free),success_matches_original=success==metrics['success_rate'],replay_matches_old_run=success==old['success'] and steps==old['steps'] and close(max(individual_distances),old['max_distance_coordinate_units']),round_event_count_consistent=len(record['events'])==steps*len(paths),all_events_ok=all(e['ok'] for e in record['events']))
        far_events=[]
        for event in record['map_alias_leak_events']:
            round_id=event['round']; actor=event['actor']; observer=event['observer']
            actor_pos=paths[actor][round_id+1]; observer_pos=paths[observer][round_id+int(observer<actor)]
            separation=math.hypot(actor_pos[0]-observer_pos[0],actor_pos[1]-observer_pos[1])
            if separation>162 and event['new_free']>0:
                far_events.append(dict(**event,actor_observer_distance=separation,actor_position=actor_pos,observer_position=observer_pos))
        replay_results.append(dict(name=name,passed=all(checks.values()),disjoint_sensor_footprint_leak_events=len(far_events),first_disjoint_footprint_leak=far_events[:1],checks=checks,independent_steps=steps,independent_max_distance=max(individual_distances),independent_robot_distances=individual_distances,independent_coverages=coverages,false_free_cells=false_free,success=success,alias_leak_events=len(record['map_alias_leak_events']),raw_map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest()))
    demo=json.loads((OLD/'hybrid_example.json').read_text()); demo_distance=max(math.fsum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(p,p[1:])) for p in demo['paths'])
    result=dict(aggregate300=aggregate,raw_replays6=replay_results,old_demo_distance=demo_distance,old_demo_distance_matches=close(demo_distance,demo['metrics']['travel_dist']),scope='300 summary rows reaggregated; six raw trajectory/final personal-map replays independently calculated. Not a raw-level audit of all 300.',physical_unit='coordinate units audited; 0.25 m/pixel inferred from paper sizes and image dimensions, not metadata or signal-model calibration')
    (OUT/('metric_verification_partial.json' if '--partial' in sys.argv else 'metric_verification.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False)); print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
