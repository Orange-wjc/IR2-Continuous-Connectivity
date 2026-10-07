"""Single writer for progress/final analysis of the independent study."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/independent_20261006'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);args=parser.parse_args();last=-6
    while True:
        p=OUT/'episodes.jsonl';n=len(p.read_text().splitlines()) if p.exists() else 0
        complete=(OUT/'complete.json').exists()
        if complete or n>=last+6:
            with (OUT/'progress_summary.log').open('a') as log:
                subprocess.run([sys.executable,'-B','report_independent.py','--partial'],cwd=ROOT,stdout=log,stderr=log,check=True)
            last=n
        if complete:
            with (OUT/'validation.log').open('w') as log:
                subprocess.run([sys.executable,'-B','validate_independent.py'],cwd=ROOT,stdout=log,stderr=log,check=True)
            (OUT/'monitor_complete.json').write_text(json.dumps(dict(episodes=n,validated=True)))
            return
        if not Path('/proc/'+str(args.pid)).exists():
            raise RuntimeError('Evaluation exited before completion; inspect run.log')
        time.sleep(20)


if __name__=='__main__':main()
