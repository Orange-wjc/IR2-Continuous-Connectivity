"""Finite pilot guard/finalizer. No training, policy evaluation, or messaging."""
import fcntl,hashlib,json,os,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
ROOT=REPO/'experiments/native_easy_repair_100_20261017_seed101'
UNIT='ir2-native-repair100-20261017.service'
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finish(plan):
    protected=read(WORK/'old_run_protected.json')
    assert all(sha(REPO/p)==h for p,h in protected.items())
    report=read(ROOT/'analysis/comparison.json');a=report['groups']['IR2_FIFO'];b=report['groups']['AB_learned']
    text=f'''## 原图100张修复试验已完成：{now()}

新试验：`experiments/native_easy_repair_100_20261017_seed101`，状态pilot_complete。两组各100张原easy训练图适应后，在固定100张原easy开发图比较；原测试使用0，旧10060回合的模型/配置/清单哈希未变。

开发成功：IR2_FIFO {a['successes']}/100，AB {b['successes']}/100。失败罚时均值：{a['mean_capped_time']:.3f}→{b['mean_capped_time']:.3f}秒。共同成功{report['common_success_maps']}图耗时：{a['common_success_mean_time']}→{b['common_success_mean_time']}秒；AB快{report['common_ab_faster']}图、慢{report['common_ab_slower']}图。配对罚时差95%区间：{report['paired_bootstrap_95ci_capped_delta']}。

预先小试条件通过：{report['gate_passed']}。单种子小试不能证明普遍有效，不据此声称复现旧42%结果。服务已完成此轮且不再训练；未开启大规模或difficult。详情与全部逐图结果/图表：本轮RESULTS.md与analysis/。工程修复是否通过及SIL实际更新次数见comparison.json，不能只引用训练奖励或预检。

'''
    (WORK/'COMPLETION.md').write_text(text)
    for path in [Path('/home/wyz/wjc_ws/papers/多机器人有限通信探索_实验计划与新对话交接.md'),
                 REPO/'finite_contact/EXPERIMENT_HANDOFF.md',REPO/'finite_contact/FEISHU_TASK_CONTEXT.md']:
        original=path.read_text();at=original.find('\n')+1
        marker='<!-- NATIVE_REPAIR100_COMPLETION -->'
        if marker not in original:path.write_text(original[:at]+'\n'+marker+'\n'+text+original[at:].lstrip('\n'))
    write(WORK/'completion_checks.json',dict(time=now(),old_run_protected=True,
        old_stop_present=(REPO/'experiments/native_easy_20261015_seed101/STOP').exists(),
        old_supervisor_pause_present=Path('/home/wyz/wjc_ws/training_supervisor_native_easy/PAUSE').exists(),
        gate_passed=report['gate_passed'],large_scale_started=False,feishu_messages_sent=0))

def main():
    lock=(WORK/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();last=0;low_since=None
    while time.monotonic()-start<16*3600:
        info={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
        available=info['MemAvailable']/1048576
        plan=read(ROOT/'pipeline.json')
        if available<4:
            (ROOT/'STOP').touch()
            write(WORK/'memory_guard_triggered.json',dict(time=now(),available_gib=available,
                action='STOP recorded; terminate this pilot control group and retain last complete checkpoints'))
            subprocess.run(['systemctl','--user','kill','--signal=SIGTERM',UNIT],check=False)
            return
        if time.monotonic()-last>=20:
            state=subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState','--value'],text=True).strip()
            write(WORK/'monitor_status.json',dict(time=now(),available_memory_gib=available,
                experiment_status=plan['status'],stage=plan['stage'],warmup_episodes=plan.get('warmup_episodes',0),
                completed_episodes=plan.get('completed_episodes',{}),unit_state=state,notifications=False))
            last=time.monotonic()
            if plan['status']=='pilot_complete':finish(plan);return
            if state in ('inactive','failed'):
                write(WORK/'needs_review.json',dict(time=now(),status=plan['status'],unit_state=state,error=plan.get('error'),
                    note='No automatic extension or hot source edits; inspect retained logs/checkpoints'))
                return
        time.sleep(2)
    write(WORK/'needs_review.json',dict(time=now(),error='Monitor reached its finite 16-hour limit'))
if __name__=='__main__':main()
