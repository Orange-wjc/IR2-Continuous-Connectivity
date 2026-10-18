#!/usr/bin/env bash
set -euo pipefail
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/ir2-native-easy10-cycles-mpl PYTHONUNBUFFERED=1
run=experiments/native_easy10_cycles_20261018_seed101
log=research_work/native_easy10_cycles_20261018/training.log
if [[ -e "$run/STOP" ]]; then
    echo 'User STOP exists; leave ten-map diagnostic training stopped.' >> "$log"
    exit 0
fi
# Allow recovery indefinitely when checkpoints advance, but stop a repeated
# reproducible failure after three attempts at the same retained progress.
if ! /home/wyz/anaconda3/envs/ir2_train/bin/python - <<'PY' >> "$log" 2>&1
from pathlib import Path
import json
from datetime import datetime,timezone,timedelta
w=Path('research_work/native_easy10_cycles_20261018');root=Path('experiments/native_easy10_cycles_20261018_seed101')
p=w/'launch_attempts.json';d=json.loads(p.read_text()) if p.exists() else {'attempts':[]}
plan=json.loads((root/'pipeline.json').read_text()) if (root/'pipeline.json').exists() else {}
progress=plan.get('completed_episodes',{'IR2_FIFO':0,'AB_learned':0})
same=0
for row in reversed(d['attempts']):
    if row['progress']!=progress:break
    same+=1
if same>=3:
    (w/'needs_review.json').write_text(json.dumps({'reason':'Three training attempts failed without advancing saved progress','progress':progress,'action':'No infinite crash loop; inspect retained logs/checkpoints'},indent=2)+'\n')
    raise SystemExit(1)
d['attempts'].append(dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),progress=progress))
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
print(json.dumps({'stage':'service_launch','attempt':len(d['attempts']),'retained_progress':progress}),flush=True)
PY
then
    exit 0
fi
if [[ -f "$run/pipeline.json" ]]; then
    exec /home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact --output "$run" --resume >> "$log" 2>&1
fi
exec /home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
    --output "$run" \
    --from-run experiments/native_easy_repair_100_20261017_seed101 \
    --dataset experiments/native_easy_repair_100_20261017_seed101/dataset/native_dataset.json \
    --continuous --train-maps 10 --num-meta-agent 24 --microbatch 16 --no-auto-evaluation \
    >> "$log" 2>&1
