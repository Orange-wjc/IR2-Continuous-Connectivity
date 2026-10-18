#!/usr/bin/env bash
set -euo pipefail
cd /home/wyz/wjc_ws/IR2-Continuous-Connectivity
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/ir2-native-500-mpl PYTHONUNBUFFERED=1
run=experiments/native_easy_500_20261017_seed101
log=research_work/native_easy_500_20261017/training.log
if [[ -e "$run/STOP" ]]; then
    echo '500-map continuation STOP exists; retaining the saved run paused.' >> "$log"
    exit 0
fi
if [[ -f "$run/pipeline.json" ]]; then
    exec /home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact --output "$run" --resume >> "$log" 2>&1
fi
exec /home/wyz/anaconda3/envs/ir2_train/bin/python driver.py --finite-contact \
    --output "$run" \
    --from-run experiments/native_easy_repair_100_20261017_seed101 \
    --dataset experiments/native_easy_20261015_seed101/dataset/native_dataset.json \
    --train-maps 500 --pilot-episodes 500 --num-meta-agent 24 --microbatch 16 \
    --max-hours 14 >> "$log" 2>&1
