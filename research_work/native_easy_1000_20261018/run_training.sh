#!/usr/bin/env bash
set -euo pipefail
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/ir2-native-1000-mpl PYTHONUNBUFFERED=1
run=experiments/native_easy_1000_20261018_seed101
log=research_work/native_easy_1000_20261018/training.log
if [[ -e "$run/STOP" ]]; then
    echo '1000-map STOP exists; preserving saved checkpoints.' >> "$log"
    exit 0
fi
# Count actual training launches across the service lifetime, not only failures
# within systemd's rate-limit window: first launch + at most two retries.
if ! /home/wyz/anaconda3/envs/ir2_train/bin/python - <<'PY' >> "$log" 2>&1
from pathlib import Path
import json
from datetime import datetime,timezone,timedelta
w=Path('research_work/native_easy_1000_20261018');p=w/'launch_attempts.json'
d=json.loads(p.read_text()) if p.exists() else {'attempts':[]}
if len(d['attempts'])>=3:
    (w/'retry_limit_reached.json').write_text(json.dumps({'attempts':len(d['attempts']),'action':'No further automatic training starts; inspect saved state'},indent=2)+'\n')
    raise SystemExit(1)
d['attempts'].append(datetime.now(timezone(timedelta(hours=8))).isoformat())
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
print(json.dumps({'stage':'service_launch','attempt':len(d['attempts'])}),flush=True)
PY
then
    exit 0
fi
if [[ -f "$run/pipeline.json" ]]; then
    exec /home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact --output "$run" --resume >> "$log" 2>&1
fi
exec /home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
    --output "$run" \
    --from-run experiments/native_easy_500_20261017_seed101 \
    --dataset experiments/native_easy_20261015_seed101/dataset/native_dataset.json \
    --train-maps 1000 --pilot-episodes 1000 --num-meta-agent 24 --microbatch 16 \
    --max-hours 18 >> "$log" 2>&1
