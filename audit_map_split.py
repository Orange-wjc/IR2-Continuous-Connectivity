"""Geometry split audit: ignores starting-marker gray values; no GPU needed."""
import hashlib,json,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/credibility_20261006'
def digest(mask):
    return hashlib.sha256(np.asarray(mask.shape,dtype='<u4').tobytes()+np.packbits(mask,bitorder='little').tobytes()).hexdigest()
def fingerprints(path):
    with Image.open(path) as im: free=np.asarray(im.convert('L'))>150
    transforms=[np.rot90(free,k) for k in range(4)]
    transforms += [np.fliplr(m) for m in transforms]
    hashes=[digest(m) for m in transforms]
    return hashes[0],min(hashes),list(free.shape)
def main():
    OUT.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    training=sorted((ROOT/'DungeonMaps/train').rglob('*.png')); exact=defaultdict(list); symmetric=defaultdict(list)
    for i,path in enumerate(training):
        e,s,shape=fingerprints(path); name=str(path.relative_to(ROOT)); exact[e].append(name); symmetric[s].append(name)
        if (i+1)%1000==0: print('training scanned',i+1,'/',len(training),flush=True)
    rows=[json.loads(l) for l in (ROOT/'reproduction_results/pretrained_stage2_20261005/episodes.jsonl').read_text().splitlines()]
    matches=[]; selected=defaultdict(list); selected_sym=defaultdict(list)
    for row in rows:
        path=ROOT/'DungeonMaps/test'/row['map_type']/row['map_file']; name=str(path.relative_to(ROOT)); e,s,shape=fingerprints(path)
        selected[e].append(name); selected_sym[s].append(name)
        if e in exact or s in symmetric: matches.append(dict(test=name,exact_geometry_training_matches=exact[e],rotation_or_reflection_training_matches=symmetric[s]))
    all_tests=sorted((ROOT/'DungeonMaps/test').rglob('*.png')); all_matches=[]; disjoint=[]; all_sym=defaultdict(set); all_counts=Counter()
    for path in all_tests:
        e,s,shape=fingerprints(path); name=str(path.relative_to(ROOT)); kind=path.parent.name
        all_counts[kind]+=1; all_sym[kind].add(s)
        if s in symmetric:
            all_matches.append(dict(test=name,exact_geometry_training_matches=exact[e],rotation_or_reflection_training_matches=symmetric[s]))
        else: disjoint.append(name)
    result=dict(training_png_count=len(training),training_counts_by_folder=dict(Counter(str(p.parent.relative_to(ROOT/'DungeonMaps/train')) for p in training)),selected_test_count=len(rows),unique_training_geometry=len(exact),unique_training_geometry_up_to_symmetry=len(symmetric),training_geometry_duplicate_groups=[v for v in exact.values() if len(v)>1],test_training_matches=matches,selected_test_geometry_duplicate_groups=[v for v in selected.values() if len(v)>1],selected_test_symmetry_duplicate_groups=[v for v in selected_sym.values() if len(v)>1],all_test_counts=dict(all_counts),all_test_unique_geometry_up_to_symmetry={k:len(v) for k,v in all_sym.items()},all_test_training_matches=all_matches,eligible_geometry_disjoint_test_maps=disjoint,method='PIL grayscale >150 occupancy; start markers treated as free; hash dimensions + packed binary pixels; all 8 rotations/reflections',limitations='No fuzzy/perceptual near-duplicate detection; cannot prove the released checkpoint was trained only on these provided training maps',wall_seconds=time.monotonic()-start)
    (OUT/'map_split_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ['training_png_count','selected_test_count','unique_training_geometry','unique_training_geometry_up_to_symmetry','wall_seconds']}),flush=True)
    print('matched selected tests',len(matches),flush=True)
if __name__=='__main__': main()
