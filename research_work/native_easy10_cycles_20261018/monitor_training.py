"""Persistent RAM/progress guard for ten-map diagnostic cycles; no evaluation/messages."""
import fcntl,json,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
ROOT=REPO/'experiments/native_easy10_cycles_20261018_seed101'
UNIT='ir2-native-easy10-cycles-20261018.service'
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,value):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def memory_available():
 info={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
 return info['MemAvailable']/1048576
def unit_state():return subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState','--value'],text=True).strip()
def main():
 lock=(WORK/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 last=0
 while True:
  available=memory_available();resource=ROOT/'RESOURCE_PAUSE'
  plan=read(ROOT/'pipeline.json') if (ROOT/'pipeline.json').exists() else dict(status='preparing',stage='inheritance_verification')
  state=unit_state()
  if (ROOT/'STOP').exists() and state in ('inactive','failed'):
   write(WORK/'monitor_status.json',dict(time=now(),status='stopped_by_user',completed_episodes=plan.get('completed_episodes',{}),automatic_evaluation=False))
   return
  if available<4 and not resource.exists():
   ROOT.mkdir(parents=True,exist_ok=True);resource.touch()
   write(WORK/'memory_guard.json',dict(time=now(),available_memory_gib=available,
    action='Stop only this training unit at saved progress; automatically resume after RAM recovers to 6 GiB; do not evaluate'))
   subprocess.run(['systemctl','--user','stop',UNIT],check=False)
   state=unit_state()
  if resource.exists() and available>=6 and not (ROOT/'STOP').exists():
   resource.unlink()
   write(WORK/'memory_recovery.json',dict(time=now(),available_memory_gib=available,action='Resume existing full checkpoint; no evaluation'))
   subprocess.run(['systemctl','--user','start',UNIT],check=False)
   state=unit_state()
  if time.monotonic()-last>=20:
   write(WORK/'monitor_status.json',dict(time=now(),available_memory_gib=available,
    experiment_status=plan['status'],stage=plan['stage'],completed_episodes=plan.get('completed_episodes',{}),
    training_progress=plan.get('training_progress',{}),unit_state=state,
    fixed_train_maps=10,planned_episode_limit=None,planned_walltime_limit=None,
    automatic_evaluation=False,stop_on_success_goal=False,notifications=False,
    resource_pause=resource.exists()))
   last=time.monotonic()
   if state in ('inactive','failed') and not resource.exists() and not (ROOT/'STOP').exists():
    write(WORK/'needs_review.json',dict(time=now(),status=plan['status'],error=plan.get('error'),
     note='Unexpected stopped/failed training; retain checkpoints, inspect error; no evaluation or additional maps'))
    return
  time.sleep(2)
if __name__=='__main__':main()
