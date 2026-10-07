"""Offline ground-truth legality audit of saved paths; no policy evaluation."""
import json, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.draw import line

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reproduction_results/credibility_20261006'


def four_connected(a, b):
    # Independent enumeration of the repository's 4-connected raster convention.
    x, y = a
    tx, ty = b
    dx, dy = abs(tx-x), abs(ty-y)
    error = dx-dy
    result = [(x, y)]
    while (x, y) != (tx, ty):
        if error > 0:
            x += 1 if tx > x else -1
            error -= 2*dy
        else:
            y += 1 if ty > y else -1
            error += 2*dx
        result.append((x, y))
        if len(result) > dx+dy+1:
            raise RuntimeError('Raster failed to reach endpoint')
    return np.asarray(result)


def cell_interiors(a, b):
    # Partition a center-to-center segment at every pixel boundary, then inspect
    # the midpoint of each positive-length interval. Corner contacts have no
    # interval and therefore do not count as crossing an obstacle interior.
    a = np.asarray(a, dtype=float)
    delta = np.asarray(b, dtype=float)-a
    crossings = [0.0, 1.0]
    for axis in range(2):
        if delta[axis] != 0:
            low,high = sorted([a[axis], a[axis]+delta[axis]])
            boundaries = np.arange(math.floor(low)+0.5, math.ceil(high), 1.0)
            crossings.extend(((boundaries-a[axis])/delta[axis]).tolist())
    ts = np.sort(np.asarray(crossings))
    left,right = ts[:-1],ts[1:]
    keep = (right-left)>1e-10
    mid = (left[keep]+right[keep])/2
    return np.floor(a[None,:]+mid[:,None]*delta[None,:]+0.5).astype(int)


def main():
    started = time.monotonic()
    control = ROOT/'reproduction_results/connectivity_control_20261006'
    files = [(mode, p) for mode in ['both', 'map_only'] for p in sorted((control/mode).glob('*.json')) if p.stem not in ['episodes']]
    original = ROOT/'reproduction_results/audit_20261006'
    cases = [('corridor',0),('corridor',9),('hybrid',0),('hybrid',1),('complex',0),('complex',5)]
    files += [('original', original/(kind+'_'+str(index)+'.json')) for kind,index in cases]
    maps, cache, rows = {}, {}, []
    for mode,path in files:
        data = json.loads(path.read_text())
        row = data.get('row', data)
        name = (row['map_type'], row['map_file'])
        if name not in maps:
            with Image.open(ROOT/'DungeonMaps/test'/name[0]/name[1]) as im:
                maps[name] = np.asarray(im.convert('L')) > 150
        free = maps[name]
        counts = Counter(); examples = []
        for robot, route in enumerate(data['paths']):
            points = []
            for step,p in enumerate(route):
                valid = len(p)==2 and all(math.isfinite(v) and float(v).is_integer() for v in p)
                in_bounds = valid and 0<=p[0]<free.shape[1] and 0<=p[1]<free.shape[0]
                clear = in_bounds and bool(free[int(p[1]),int(p[0])])
                counts['poses'] += 1
                counts['invalid_poses'] += not clear
                if not clear and len(examples)<20:
                    examples.append(dict(robot=robot,step=step,pose=p,issue='invalid or occupied endpoint'))
                points.append(tuple(int(v) for v in p) if valid else None)
            for step,(a,b) in enumerate(zip(points,points[1:])):
                counts['segments'] += 1
                if a is None or b is None: continue
                key = (name,a,b)
                if key not in cache:
                    rr,cc = line(a[1],a[0],b[1],b[0])
                    coords = four_connected(a,b)
                    interior = cell_interiors(a,b)
                    def obstructed(xs,ys):
                        if np.any(xs<0) or np.any(ys<0) or np.any(xs>=free.shape[1]) or np.any(ys>=free.shape[0]):
                            return True
                        return not bool(np.all(free[ys,xs]))
                    cache[key] = (obstructed(cc,rr), obstructed(coords[:,0],coords[:,1]), obstructed(interior[:,0],interior[:,1]))
                eight_bad,four_bad,interior_bad = cache[key]
                counts['obstructed_segments_8_connected'] += eight_bad
                counts['obstructed_segments_4_connected'] += four_bad
                counts['obstacle_interior_crossings'] += interior_bad
                if (eight_bad or four_bad or interior_bad) and len(examples)<20:
                    reverse = four_connected(b,a)
                    reverse_bad = bool(np.any(~free[reverse[:,1],reverse[:,0]]))
                    examples.append(dict(robot=robot,step=step,start=a,end=b,raster8=eight_bad,raster4=four_bad,raster4_reverse=reverse_bad,obstacle_interior=interior_bad))
        rows.append(dict(mode=mode,map_type=row['map_type'],map_index=row['map_index'],map_file=row['map_file'],counts=dict(counts),examples=examples))
    totals = {mode:dict(sum((Counter(r['counts']) for r in rows if r['mode']==mode), Counter())) for mode in ['both','map_only','original']}
    result = dict(records=len(rows),counts_by_mode=dict(Counter(r['mode'] for r in rows)),totals=totals,rows=rows,unique_segments=len(cache),method='Raw PNG grayscale >150 free; integer finite in-bounds free endpoints; independent skimage 8-connected raster plus 4-connected raster matching source convention, both including endpoints; continuous point-segment traversal partitioned at all half-integer cell boundaries to detect positive-length obstacle interior crossings',limitations='Point-robot grid legality only; corner contacts imply zero clearance, not proof of interior collision; no robot footprint, continuous dynamics or Gazebo/real-world collision proof; original source raw paths available for six cases only',wall_seconds=time.monotonic()-started)
    (OUT/'trajectory_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ['records','counts_by_mode','totals','unique_segments','wall_seconds']}),flush=True)


if __name__ == '__main__':
    main()
