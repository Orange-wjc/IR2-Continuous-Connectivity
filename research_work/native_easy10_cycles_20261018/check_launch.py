"""Verify the actual unrestricted diagnostic, without mutating its checkpoints."""
import json,sys,time,urllib.request,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
ROOT=REPO/'experiments/native_easy10_cycles_20261018_seed101'
sys.path.insert(0,str(ROOT/'source'))
from native_data import digest
from finite_contact.memory import load_checkpoint,semantic_digest
from native_continuation import preserved_state

def read(p):return json.loads(p.read_text())
started=time.monotonic();tags={}
while True:
 plan=read(ROOT/'pipeline.json')
 assert plan['status'] not in ('interrupted_or_failed',),plan.get('error')
 try:
  with urllib.request.urlopen('http://127.0.0.1:6009/data/plugin/scalars/tags',timeout=5) as response:tags=json.load(response)
 except OSError:tags={}
 if min(plan['completed_episodes'].values())>=1 and all('train/success_rate_10_percent' in tags.get(name,{}) for name in plan['groups']):break
 if time.monotonic()-started>180:raise RuntimeError('Actual-launch verification timed out; training left untouched, inspect logs')
 time.sleep(2)
checks={}
checks['running_and_not_preflight']=plan['status']=='running' and not plan['preflight']
checks['fixed_ten_maps_and_manual_evaluation']=plan['train_maps']==10 and not plan['automatic_development_evaluation'] and not plan['endpoint_development_evaluation']
checks['no_scheduled_or_success_stop']=plan['pilot_episodes'] is None and plan['wall_guard_hours'] is None and not plan['stop_on_success_goal']
checks['source_snapshot_intact']=all(digest(ROOT/'source'/p)==sha for p,sha in plan['source_identity'].items())
checks['data_manifest_intact']=digest(ROOT/'dataset/native_dataset.json')==plan['dataset_sha256']
initials=[]
first_records={}
for name,n in plan['completed_episodes'].items():
 initial=load_checkpoint(ROOT/name/'initial.pth')
 cfg=initial['config']
 checks[name+'_real_task_limits']=cfg['max_decisions']==196 and cfg['horizon_seconds']==3000 and cfg['completion']==.99
 checks[name+'_unchanged_learning_recipe']=cfg['learning_rate']==1e-4 and cfg['policy_learning_rate']==1e-5 and cfg['batch_size']==16 and cfg['success_imitation_weight']==1
 checks[name+'_empty_initial_replay_success']=not initial['replay'] and not initial['content_replay'] and not initial['success_episodes'] and initial['episode']==0
 initial['role']='equal';initials.append(semantic_digest(preserved_state(initial)));del initial
 rows=[json.loads(s) for s in (ROOT/name/'training.jsonl').read_text().splitlines() if s.strip()]
 rows=[r for r in rows if r['episode']<=n]
 first=rows[0];first_records[name]={k:first[k] for k in ['episode','decisions','success','stop_reason','learner_updates','new_sac_updates']}
 checks[name+'_first_map_and_real_updates']=first['episode']==1 and first['map']['index']==0 and first['decisions']>8 and first['new_sac_updates']==20 and first['learner_updates']==9020
 old=json.loads((REPO/'experiments/native_easy_repair_100_20261017_seed101'/name/'training.jsonl').read_text().splitlines()[0])
 keys=['success','time','coverage','union_coverage','total_distance','return_values','decisions','capped_time','action_seed','learner_updates','new_sac_updates']
 checks[name+'_first_trajectory_matches_old_common_start']=all(first[k]==old[k] for k in keys)
 checks[name+'_http_tensorboard_live_map_and_training_tags']=all(t in tags[name] for t in ['train/success_rate_10_percent','train/success_rate_100_percent','maps/map_00/success_rate_last10_percent'])
checks['identical_initial_learner_and_rng']=initials[0]==initials[1]
protected=read(WORK/'protected_runs.json')
checks['historical_files_unchanged']=all(digest(REPO/p)==sha for p,sha in protected.items())
old=read(REPO/'experiments/native_easy100_cycles_20261018_seed101/pipeline.json')
checks['previous100_remains_saved_and_stopped']=old['status']=='stopped_by_request' and old['completed_episodes']=={'IR2_FIFO':501,'AB_learned':501} and (REPO/'experiments/native_easy100_cycles_20261018_seed101/STOP').exists()
units={}
for unit in ['ir2-native-easy10-cycles-20261018','ir2-native-easy10-cycles-review-20261018','ir2-native-easy10-tensorboard-20261018']:
 text=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','NRestarts'],text=True)
 units[unit]=dict(line.split('=',1) for line in text.splitlines())
checks['services_active_without_restart']=all(s['ActiveState']=='active' and s['NRestarts']=='0' for s in units.values())
report=dict(passed=all(checks.values()),checked_at=plan['updated'],checks=checks,completed_episodes=plan['completed_episodes'],first_records=first_records,services=units,tensorboard_url='http://127.0.0.1:6009',scope='Actual 196-decision ten-map training launch; not convergence evidence')
(WORK/'launch_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
assert report['passed']
