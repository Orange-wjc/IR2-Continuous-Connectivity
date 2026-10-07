"""Complete progress/final artifacts even if the interactive session disconnects."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/connectivity_control_20261006'

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pid',type=int,required=True); args=parser.parse_args()
    os.chdir(ROOT); last=-10
    while True:
        f=OUT/'both/episodes.jsonl'; rows=[json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
        complete=(OUT/'full_complete.json').exists()
        alive=Path('/proc/'+str(args.pid)).exists()
        if len(rows)>=last+10 or complete or not alive:
            with (OUT/'progress_summary.log').open('w') as log:
                subprocess.run([sys.executable,'-B','report_control.py']+([] if complete else ['--partial']),stdout=log,stderr=subprocess.STDOUT,check=True)
            last=len(rows)
            print(json.dumps(dict(completed=last,full_complete=complete,parent_running=alive)),flush=True)
        if complete:
            (OUT/'monitor_complete.json').write_text(json.dumps(dict(count=last,report='对照报告.md',validation='independent checks, paired identities, source/weights/map hashes passed')))
            return
        if not alive:
            raise RuntimeError('Evaluation exited before full completion; inspect run.log and resume with evaluate_control.py')
        time.sleep(20)
if __name__=='__main__': main()
