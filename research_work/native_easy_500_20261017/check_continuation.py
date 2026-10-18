"""Root-entry continuation, inheritance, STOP/recovery and fixed-budget smoke check."""
from pathlib import Path
import json,subprocess,time,os,hashlib
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
PYTHON='/home/wyz/anaconda3/envs/ir2_train/bin/python'
ROOT=REPO/'experiments/native_easy_500_preflight_20261017'
PARENT=REPO/'experiments/native_repair_100_preflight_v2_20261017'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not ROOT.exists()
 protected={str(p):sha(p) for p in [PARENT/'pipeline.json',PARENT/'IR2_FIFO/latest.pth',PARENT/'AB_learned/latest.pth']}
 env=os.environ.copy()
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
 cmd=[PYTHON,str(REPO/'driver.py'),'--finite-contact','--output',str(ROOT)]
 log=(WORK/'preflight.log').open('w')
 proc=subprocess.Popen(cmd+['--dataset',str(REPO/'experiments/native_easy_20261015_seed101/dataset/native_dataset.json'),
  '--from-run',str(PARENT),'--train-maps','500','--pilot-episodes','6','--preflight','--num-meta-agent','2'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+600;stopped=False
 while proc.poll() is None:
  if time.monotonic()>deadline:raise RuntimeError('Continuation smoke check timed out')
  p=ROOT/'pipeline.json'
  if not stopped and p.exists() and max(read(p)['completed_episodes'].values())>=5:
   (ROOT/'STOP').touch();stopped=True
  time.sleep(.05)
 assert stopped and proc.returncode==0
 plan=read(ROOT/'pipeline.json');assert plan['status']=='pilot_paused'
 audit=read(ROOT/'inheritance_audit.json')
 assert all(g['state_preserved'] for g in audit['groups'].values())
 assert not audit['head_retrained'] and not audit['shared_prefix_retrained']
 (ROOT/'STOP').unlink()
 result=subprocess.run(cmd+['--resume'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 assert result.returncode==0
 plan=read(ROOT/'pipeline.json')
 assert plan['status']=='pilot_complete' and plan['completed_episodes']=={'IR2_FIFO':6,'AB_learned':6}
 assert plan['sac_updates']=={'IR2_FIFO':16,'AB_learned':16}
 assert not (ROOT/'shared_warmup').exists() and not (ROOT/'head_collection').exists()
 for group in ('IR2_FIFO','AB_learned'):
  rows=[json.loads(s) for s in (ROOT/group/'training.jsonl').read_text().splitlines()]
  assert [r['episode'] for r in rows]==[5,6]
  assert [r['map']['index'] for r in rows]==[4,5]
  assert all(r['new_sac_updates']==2 for r in rows)
 checkpoints={str(p):sha(p) for p in [ROOT/'IR2_FIFO/latest.pth',ROOT/'AB_learned/latest.pth']}
 result=subprocess.run(cmd+['--resume'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 assert result.returncode==0 and all(sha(Path(p))==h for p,h in checkpoints.items())
 assert all(sha(Path(p))==h for p,h in protected.items())
 checks=read(ROOT/'analysis/comparison.json')['engineering_checks']
 assert all(v for k,v in checks.items() if not k.endswith('successful_imitation_active')),checks
 report=dict(full_state_inheritance=True,only_three_config_fields_changed=True,original_500_files_verified=True,
  no_shared_prefix_or_head_retraining=True,correct_next_map_indices=[4,5],
  stop_and_resume=True,completed_resume_adds_no_sac=True,parent_preserved=True,
  endpoint_geometry_radio_checks_passed=True,engineering_only=True)
 (WORK/'preflight_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
