"""Real-PNG engineering smoke check: ten-map wrap, learning and STOP/resume."""
import ast,json,os,subprocess,time,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
ROOT=WORK/'preflight';PY=Path('/home/wyz/anaconda3/envs/ir2_train/bin/python')
os.chdir(REPO)
for p in (REPO/'native_training').rglob('*.py'):ast.parse(p.read_text())
sys.path.insert(0,str(REPO/'native_training'))
from finite_contact.memory import load_checkpoint,semantic_digest
from native_continuation import preserved_state
from native_driver import automatic_review_due,training_endpoint_reached

def read(p):return json.loads(p.read_text())
def launch(resume,target):
 cmd=[str(PY),'driver.py','--finite-contact','--output',str(ROOT)]
 if resume:cmd+=['--resume']
 else:cmd+=['--continuous','--train-maps','10','--from-run','experiments/native_easy_repair_100_20261017_seed101',
  '--dataset','experiments/native_easy_repair_100_20261017_seed101/dataset/native_dataset.json',
  '--num-meta-agent','24','--microbatch','16','--no-auto-evaluation','--preflight']
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/ir2-ten-preflight-mpl')
 with (WORK/'preflight.log').open('a') as out:
  child=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=env)
  started=time.monotonic();sent=False
  while child.poll() is None:
   if (ROOT/'pipeline.json').exists():
    plan=read(ROOT/'pipeline.json')
    if min(plan.get('completed_episodes',{}).values(),default=0)>=target and not sent:
     (ROOT/'STOP').touch();sent=True
   if time.monotonic()-started>240:
    (ROOT/'STOP').touch();child.terminate()
    raise RuntimeError('Engineering check timed out; retained all logs')
   time.sleep(.2)
  assert child.returncode==0,(child.returncode,(WORK/'preflight.log').read_text()[-5000:])
 return read(ROOT/'pipeline.json')

assert not ROOT.exists(),'Use a fresh engineering directory; do not overwrite artifacts'
first=launch(False,20)
assert first['status']=='stopped_by_request'
first_counts=first['completed_episodes'].copy()
(ROOT/'STOP').unlink()
last=launch(True,min(first_counts.values())+3)
checks={}
checks['engineered_short_decision_limit_not_performance_evidence']=True
checks['two_cycles_and_resume']=min(last['completed_episodes'].values())>=23
checks['no_episode_or_walltime_cap']=last['pilot_episodes'] is None and last['wall_guard_hours'] is None
checks['manual_evaluation_only']=not last['automatic_development_evaluation'] and not last['endpoint_development_evaluation']
checks['no_automatic_review_even_at_99']=not automatic_review_due({**last,'automatic_development_evaluation':True},last['completed_episodes'])
checks['no_endpoint_stop']=not training_endpoint_reached(last,last['completed_episodes']) and not last['stop_on_success_goal']
checks['no_eval_or_shared_retraining_artifacts']=not any(ROOT.glob('*/validation_*')) and not (ROOT/'shared_warmup').exists() and not (ROOT/'head_collection').exists()
initials=[]
for name,n in last['completed_episodes'].items():
 rows=[json.loads(s) for s in (ROOT/name/'training.jsonl').read_text().splitlines() if s.strip()]
 checks[name+'_continuous_log']=len(rows)==n and [r['episode'] for r in rows]==list(range(1,n+1))
 checks[name+'_wrap_order']=all(r['map']['index']==(r['episode']-1)%10 for r in rows)
 checks[name+'_only_ten_original_maps']=len({r['map']['layout_id'] for r in rows})==10 and all(r['map']['split']=='train' for r in rows)
 checks[name+'_permap_stats_survive_resume']=sum(v['visits'] for v in last['training_progress'][name]['per_map'].values())==n
 cp=load_checkpoint(ROOT/name/'latest.pth')
 checks[name+'_checkpoint_matches_log']=cp['episode']==n and cp['learner']['updates']==last['sac_updates'][name]
 checks[name+'_exact_updates_per_episode']=cp['learner']['updates']==9000+2*n and all(r['new_sac_updates']==2 for r in rows)
 checks[name+'_only_diagnostic_replay']=len(cp['replay'])==sum(2*r['decisions'] for r in rows)
 checks[name+'_short_preflight_config']=cp['config']['train_maps']==10 and cp['config']['max_decisions']==8 and cp['config']['horizon_seconds']==3000
 initial=load_checkpoint(ROOT/name/'initial.pth');initial.pop('config');initial.pop('time');initial['role']='equal'
 initials.append(semantic_digest(preserved_state(initial)))
 del cp,initial
checks['identical_initial_learner_replay_rng']=initials[0]==initials[1]
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for name in last['groups']:
 ea=EventAccumulator(str(ROOT/'tensorboard'/name));ea.Reload();tags=ea.Tags()['scalars']
 checks[name+'_tensorboard_cycle_and_map_stats']=all(t in tags for t in ['cycle/success_rate_percent','train/success_rate_10_percent','train/success_rate_100_percent','maps/map_00/success_rate_last10_percent','maps/map_09/success'])
 checks[name+'_tensorboard_cycles_match_log']=all(v.step<=last['completed_episodes'][name]//10 for v in ea.Scalars('cycle/success_rate_percent'))
 for v in ea.Scalars('cycle/success_rate_percent'):
  a=rows if False else [json.loads(s) for s in (ROOT/name/'training.jsonl').read_text().splitlines() if s.strip()]
  a=[r for r in a if (v.step-1)*10<r['episode']<=v.step*10]
  assert abs(v.value-100*sum(r['success'] for r in a)/10)<1e-5
protected=read(WORK/'protected_runs.json')
from native_data import digest
checks['historical_results_unchanged']=all(digest(REPO/p)==sha for p,sha in protected.items())
report=dict(checks=checks,passed=all(checks.values()),first_counts=first_counts,last_counts=last['completed_episodes'],scope='Real original PNGs with 8-decision rollouts; engineering only, not learnability evidence')
(WORK/'preflight_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
assert report['passed']
