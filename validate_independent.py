"""Recompute saved raw metrics and point-path legality after all evaluations."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.draw import line
from audit_trajectories import cell_interiors

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/independent_20261006'


def main():
    assert (OUT/'complete.json').exists()
    rows=[json.loads(l) for l in (OUT/'episodes.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['case_id'] for r in rows})==180
    maps={};cache={};totals=Counter();validated=[]
    for row in rows:
        raw=json.loads((OUT/'raw'/(row['case_id']+'.json')).read_text());assert raw['row']==row
        name=row['map_path']
        if name not in maps:
            with Image.open(ROOT/name) as im:maps[name]=np.asarray(im.convert('L'))>150
        free=maps[name];denom=int(free.sum())
        with np.load(OUT/'raw'/(row['case_id']+'_maps.npz')) as saved:personal=saved['personal']
        rates=[int(np.count_nonzero((m==255)&free))/denom for m in personal]
        assert rates==row['individual_explored']
        assert not any(np.any((m==255)&~free) for m in personal)
        distances=[math.fsum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(route,route[1:])) for route in raw['paths']]
        assert math.isclose(max(distances),row['max_distance_coordinate_units'],rel_tol=1e-12,abs_tol=1e-8)
        assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8) for a,b in zip(distances,raw['independent']['distances']))
        assert row['success']==bool(row['execution_ok'] and all(r>=.99 for r in rates))
        if row['execution_ok']:
            assert all(len(route)-1==row['steps'] for route in raw['paths'])
            assert len(raw['events'])==5*row['steps']==row['action_checks']
        assert row['copy_violations']==0 and all(raw['checks'].values())
        counts=Counter()
        for route in raw['paths']:
            for p in route:
                assert all(math.isfinite(v) and float(v).is_integer() for v in p)
                assert 0<=p[0]<free.shape[1] and 0<=p[1]<free.shape[0] and free[int(p[1]),int(p[0])]
                counts['free_poses']+=1
            for a,b in zip(route,route[1:]):
                key=(name,tuple(a),tuple(b))
                if key not in cache:
                    rr,cc=line(int(a[1]),int(a[0]),int(b[1]),int(b[0]));inside=cell_interiors(a,b)
                    cache[key]=bool(np.all(free[rr,cc]) and np.all(free[inside[:,1],inside[:,0]]))
                assert cache[key],('Obstacle interior/path crossing',row['case_id'],a,b)
                counts['legal_segments']+=1
        totals.update(counts);validated.append(row['case_id'])
    result=dict(episodes=180,validated_cases=validated,totals=dict(totals),unique_segments=len(cache),all_pass=True,checks='raw JSON/JSONL equality; NPZ coverage and zero false free; independent path lengths and rounds; success and copy isolation; finite free endpoints; 8-connected raster and continuous point-segment obstacle interiors',limits='Point paths only, no robot footprint/clearance guarantee',protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ['episodes','totals','unique_segments','all_pass']}))


if __name__=='__main__':main()
