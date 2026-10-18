"""Bounded 1000-map continuation RAM guard and local report finalizer. No messages."""
import fcntl,hashlib,json,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
ROOT=REPO/'experiments/native_easy_1000_20261018_seed101'
UNIT='ir2-native-easy1000-20261018.service'
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finish(plan):
 assert plan['completed_episodes']=={'IR2_FIFO':1000,'AB_learned':1000}
 assert all(sha(REPO/p)==h for p,h in read(WORK/'protected_runs.json').items())
 report=read(ROOT/'analysis/comparison.json');a=report['groups']['IR2_FIFO'];b=report['groups']['AB_learned']
 assert all(report['engineering_checks'].values()),'Completion requires checking the failed engineering item(s)'
 reduction=100*(a['mean_capped_time']-b['mean_capped_time'])/a['mean_capped_time']
 old=report['round_progress']['rounds'][0]
 text=f'''## 原easy累计1000图续训已完成：{now()}

输出：`experiments/native_easy_1000_20261018_seed101`，pilot_complete。两组从各自500图完整端点续训，新增500张原easy训练图、各10000 SAC；累计1000图/29000 SAC，预训练和消息头没有重跑。种子101的一次延续，非新独立种子。旧100图、500图与10060回合结果哈希校验保持。

相同100开发图：基线成功{a['successes']}/100，AB成功{b['successes']}/100；失败罚时均值{a['mean_capped_time']:.3f}→{b['mean_capped_time']:.3f}秒（AB降低{reduction:.2f}%）；共同成功{report['common_success_maps']}图耗时{a['common_success_mean_time']}→{b['common_success_mean_time']}秒，AB快{report['common_ab_faster']}图、慢{report['common_ab_slower']}图。罚时差95%区间{report['paired_bootstrap_95ci_capped_delta']}。父500图端点为基线{old['fifo_successes']}、AB{old['ab_successes']}成功，罚时降低{old['capped_reduction_percent']:.2f}%。两端点共同成功集合不同，四模型同成功交集另存，不能直接跨子集比较耗时比例。

全部工程检查通过；预先小试条件通过：{report['gate_passed']}。单种子和重复查看开发集不足以确认稳定泛化，不宣称复现旧42%收益。原测试0使用，不自动超过1000，不自动进入difficult。服务本轮已结束。详情RESULTS.md、analysis/comparison.json、round_progress.json、逐图数据及PNG/SVG。

'''
 (WORK/'COMPLETION.md').write_text(text)
 marker='<!-- NATIVE_EASY1000_COMPLETION -->'
 for path in [Path('/home/wyz/wjc_ws/papers/多机器人有限通信探索_实验计划与新对话交接.md'),REPO/'finite_contact/EXPERIMENT_HANDOFF.md',REPO/'finite_contact/FEISHU_TASK_CONTEXT.md']:
  original=path.read_text();at=original.find('\n')+1
  if marker not in original:path.write_text(original[:at]+'\n'+marker+'\n'+text+original[at:].lstrip('\n'))
 write(WORK/'completion_checks.json',dict(time=now(),protected_runs_unchanged=True,
  gate_passed=report['gate_passed'],full_dataset_started=False,stage2_started=False,feishu_messages_sent=0))
def main():
 lock=(WORK/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.monotonic();last=0
 while time.monotonic()-started<22*3600:
  info={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
  available=info['MemAvailable']/1048576
  plan=read(ROOT/'pipeline.json') if (ROOT/'pipeline.json').exists() else {'status':'preparing','stage':'inheritance_verification'}
  if available<4:
   ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'STOP').touch()
   write(WORK/'memory_guard_triggered.json',dict(time=now(),available_gib=available,
    action='STOP recorded; terminate only 1000-map continuation group; retain last complete checkpoints'))
   subprocess.run(['systemctl','--user','kill','--signal=SIGTERM',UNIT],check=False)
   return
  if time.monotonic()-last>=20:
   state=subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState','--value'],text=True).strip()
   write(WORK/'monitor_status.json',dict(time=now(),available_memory_gib=available,
    experiment_status=plan['status'],stage=plan['stage'],completed_episodes=plan.get('completed_episodes',{}),
    target=1000,unit_state=state,notifications=False))
   last=time.monotonic()
   if plan['status']=='pilot_complete':finish(plan);return
   if state in ('inactive','failed'):
    write(WORK/'needs_review.json',dict(time=now(),status=plan['status'],unit_state=state,error=plan.get('error'),
     note='Finite run stopped. No source edits, indefinite retries, extra training, or messages.'))
    return
  time.sleep(2)
 (ROOT/'STOP').touch()
 subprocess.run(['systemctl','--user','kill','--signal=SIGTERM',UNIT],check=False)
 write(WORK/'needs_review.json',dict(time=now(),error='Finite monitor reached 22-hour limit; STOP recorded and this training group terminated'))
if __name__=='__main__':main()
