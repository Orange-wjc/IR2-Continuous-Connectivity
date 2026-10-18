#!/usr/bin/env bash
# Resume all three existing 1000-map workers, displaying and retaining their output.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
ir2_python=/home/wyz/anaconda3/envs/ir2_train/bin/python
experiment_out="$PWD/experiments/thousand_maps_20261012_seed1"
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=1
elif [[ $# -gt 0 ]]; then
    printf 'Usage: bash run_thousand_maps_foreground.sh [--dry-run]\n' >&2
    exit 2
fi

if [[ "$dry_run" == 0 ]]; then
    exec 9>"$experiment_out/.foreground.lock"
    if ! flock -n 9; then
        printf 'A foreground training session already holds this experiment lock.\n' >&2
        exit 1
    fi
fi

"$ir2_python" - "$experiment_out" "$dry_run" <<'PY'
from pathlib import Path
import datetime,json,sys
from finite_contact_multimap_joint import sha256
from finite_contact_stability import read,write,recover_initialization
from finite_contact_identity import active_identity
root=Path(sys.argv[1]);dry=sys.argv[2]=='1'
plan=read(root/'pipeline.json')
identity=active_identity(root)
pid=read(root/'background_process.json')['pid']
p=Path('/proc')/str(pid)/'cmdline'
if p.exists() and b'finite_contact_scale.py' in p.read_bytes():
    raise SystemExit('The original background coordinator is still running; do not duplicate training')
assert plan['training_maps']==1000 and plan['new_sac_updates_per_group']==20000
if not dry:
    for group in plan['worker_order']:recover_initialization(root/group)
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'entry':'run_thousand_maps_foreground.sh','sha256':sha256('run_thousand_maps_foreground.sh'),
            'scope':'Foreground I/O and sequential queue only; original training sources and budget unchanged.'}
    write(root/'checks/foreground_entry.json',record)
    plan['status']='running';write(root/'pipeline.json',plan)
PY

on_exit() {
    local exit_code=$?
    if [[ "$dry_run" == 0 && "$exit_code" != 0 ]]; then
        "$ir2_python" - "$experiment_out" <<'PY'
from pathlib import Path
import sys
from finite_contact_stability import read,write
root=Path(sys.argv[1]);plan=read(root/'pipeline.json')
if plan['status']!='complete':
    plan['status']='interrupted_or_failed';write(root/'pipeline.json',plan)
PY
    fi
}
trap on_exit EXIT

for group in AB_learned A_rule A_fifo; do
    if "$ir2_python" - "$experiment_out/$group/status.json" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1])
sys.exit(0 if p.exists() and json.loads(p.read_text()).get('status')=='complete' else 1)
PY
    then
        printf 'Skipping completed group: %s\n' "$group"
        continue
    fi
    command=("$ir2_python" -u finite_contact_multimap_joint.py --stage group
             --output "$experiment_out" --group "$group" --device cuda)
    if [[ -f "$experiment_out/$group/latest.pth" ]]; then
        command+=(--resume "$experiment_out/$group/latest.pth")
    fi
    printf '\nTraining group: %s\n' "$group"
    if [[ "$dry_run" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}" 2>&1 | tee -a "$experiment_out/$group.log"
    fi
done

command=("$ir2_python" -u finite_contact_scale.py --output "$experiment_out" --device cuda --resume)
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
