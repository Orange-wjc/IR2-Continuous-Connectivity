#!/usr/bin/env python3
import datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/home/wyz/anaconda3/envs/ir2_train/bin/python'
def status(**kw):
 (ROOT/'validation_status.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),**kw),indent=2)+'\n')
def run(name,source,groups,maps,trace=False,resume=False):
 status(status='running',stage=name)
 command=[PYTHON,str(ROOT/'run_probe.py'),'--source',str(ROOT/source),'--output',str(ROOT/name),'--groups',groups,'--maps',maps]
 if trace:command+=['--trace']
 if resume:command+=['--resume-check']
 with (ROOT/(name+'.log')).open('w') as log:
  proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,text=True)
  for line in proc.stdout:
   log.write(line);log.flush();print(name+': '+line.rstrip(),flush=True)
  assert proc.wait()==0,name+' failed; see log'
 return json.loads((ROOT/name/'result.json').read_text())
try:
 groups='B0_fifo,A_fifo,B_learned,AB_learned,A_rule,AB_uncoupled'
 b=run('equivalence_baseline','baseline',groups,'0,37,470',True,True)
 o=run('equivalence_optimized','optimized',groups,'0,37,470',True,True)
 assert len(b['rows'])==len(o['rows'])
 checks=[]
 keys=['metrics','policy_inputs_actions_sha256','trajectory_sha256','communication_sha256','final_environment_sha256','learner_replay_rng_sha256']
 for x,y in zip(b['rows'],o['rows']):
  if 'resume_equivalence' in x:
   assert x['resume_equivalence'] and y['resume_equivalence']
   x,y=x['continuation'],y['continuation']
  else:assert (x['group'],x['map_index'],x['episode'])==(y['group'],y['map_index'],y['episode'])
  for k in keys:assert x[k]==y[k],(x.get('group'),x['map_index'],k)
  checks.append(dict(group=x.get('group','AB_learned-resume'),map_index=x['map_index'],exact=True))
 report=dict(status='passed',checks=checks,baseline_peak_rss_mib=b['peak_rss_mib'],optimized_peak_rss_mib=o['peak_rss_mib'],test_policy_evaluations=0)
 (ROOT/'equivalence_review.json').write_text(json.dumps(report,indent=2)+'\n')
 print('All paired semantic hashes and resume checks passed.',flush=True)
 # Unprofiled, untraced timing uses the exact same three-map workload and warmup.
 tb=run('timing_baseline','baseline','AB_learned','0,37,470')
 to=run('timing_optimized','optimized','AB_learned','0,37,470')
 rows=[]
 for x,y in zip(tb['rows'],to['rows']):
  assert x['metrics']==y['metrics']
  rows.append(dict(map_index=x['map_index'],baseline_seconds=x['total_seconds'],optimized_seconds=y['total_seconds'],speedup=x['total_seconds']/y['total_seconds']))
 before=sum(x['baseline_seconds'] for x in rows);after=sum(x['optimized_seconds'] for x in rows)
 timing=dict(status='passed',paired_maps=rows,baseline_seconds=before,optimized_seconds=after,speedup=before/after,time_reduction_percent=100*(1-after/before),baseline_peak_rss_mib=tb['peak_rss_mib'],optimized_peak_rss_mib=to['peak_rss_mib'],sample_size=3,profiled=False,traced=False,includes_checkpoint_io=False)
 (ROOT/'timing_review.json').write_text(json.dumps(timing,indent=2)+'\n')
 status(status='complete',exact_equivalence=True,speedup=timing['speedup'])
 print(json.dumps(timing),flush=True)
except BaseException as exc:
 status(status='failed',error=repr(exc));raise
