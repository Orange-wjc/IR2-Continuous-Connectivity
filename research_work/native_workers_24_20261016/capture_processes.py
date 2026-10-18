"""Read the selected training process group without exposing unrelated argv."""
import json
import os
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1] / 'experiments/native_easy_20261015_seed101'
EVIDENCE = ROOT / 'checks/parallelism_24_20261016'
launch = json.loads((EVIDENCE / 'launch.json').read_text())
rows = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        if os.getpgid(int(path.name)) != launch['pid']:
            continue
        argv = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        status = (path / 'status').read_text()
        rss = next((int(x.split()[1]) / 1024 for x in status.splitlines() if x.startswith('VmRSS:')), 0)
        role = ('sampling_worker' if 'spawn_main' in argv else
                'resource_tracker' if 'resource_tracker' in argv else 'coordinator')
        rows.append(dict(pid=int(path.name), role=role, rss_mib=rss))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
result = dict(pid=launch['pid'], sampling_workers=sum(x['role'] == 'sampling_worker' for x in rows),
              processes=rows)
assert result['sampling_workers'] == launch['selected_workers'], result
assert sum(x['role'] == 'coordinator' for x in rows) == 1
(EVIDENCE / 'runtime_processes.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
