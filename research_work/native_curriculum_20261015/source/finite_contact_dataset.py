#!/usr/bin/env python3
"""Materialize and audit diverse simple maps without policy evaluation on test maps."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/ir2-finite-contact-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.maps import configured_map, manifest


def audit_dataset(config, output):
    config.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new dataset directory')
    entries = manifest(config)
    assert len({e['sha256'] for e in entries}) == len(entries), 'Duplicate geometry'
    assert len({e['canonical_sha256'] for e in entries}) == len(entries), 'Rotated/reflected duplicate'
    rows, maps, starts = [], [], []
    for info in entries:
        truth, positions, _ = configured_map(config, info['split'], info['index'])
        env = Environment(config, info['split'], info['index'])
        coverage, union = env.coverage()
        clearance = distance_transform_edt(truth == 255)[positions[:,1].astype(int), positions[:,0].astype(int)]
        actions = [int(np.sum(env.observation(r)[4] == 0)) for r in range(2)]
        assert min(actions) > 1, f'No move from spawn: {info}'
        assert min(clearance) >= 4, f'Spawn against obstacle: {info}'
        assert union < .85, f'Trivial initial coverage: {info}'
        rows.append({'split':info['split'],'index':info['index'],'initial_coverage':coverage.tolist(),
                     'initial_union_coverage':union,'start_clearance':clearance.tolist(),'legal_actions':actions})
        maps.append(truth); starts.append(positions)
        if len(rows) % 20 == 0:
            print(json.dumps({'audited_maps':len(rows),'total':len(entries)}),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    encoded = json.dumps(entries,sort_keys=True,separators=(',',':')).encode()
    summary = {'passed':True,'family':config.map_family,'dataset_seed':config.map_seed,
        'counts':{'train':config.train_maps,'validation':config.validation_maps,'test':config.test_maps},
        'unique_geometry':len(entries),'unique_up_to_rotation_reflection':len(entries),
        'room_counts':dict(Counter(e['rooms'] for e in entries)),
        'rotation_counts':dict(Counter(e['rotation'] for e in entries)),
        'manifest_sha256':hashlib.sha256(encoded).hexdigest(),
        'checks':rows,'scope':'Test geometry quality checked; no test policy rollouts or test-based tuning.'}
    (output/'manifest.json').write_text(json.dumps(entries,indent=2))
    (output/'quality.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(output/'maps.npz',truth=np.stack(maps),starts=np.stack(starts))
    selected = list(range(min(12,config.train_maps))) + list(range(config.train_maps,config.train_maps+min(4,config.validation_maps)))
    fig,axes = plt.subplots(4,4,figsize=(10,10))
    for ax in axes.flat: ax.axis('off')
    for ax,idx in zip(axes.flat,selected):
        ax.imshow(maps[idx],cmap='gray',vmin=1,vmax=255)
        ax.scatter(starts[idx][:,0],starts[idx][:,1],s=12,c=['tab:blue','tab:orange'])
        ax.set_title(f"{entries[idx]['split']} {entries[idx]['index']} / {entries[idx]['rooms']} rooms",fontsize=9)
    fig.tight_layout(); fig.savefig(output/'preview.png',dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--map-seed',type=int,default=20261011)
    p.add_argument('--train-maps',type=int,default=100)
    args=p.parse_args()
    if Path(sys.prefix).resolve()!=Path('/home/wyz/anaconda3/envs/ir2_train').resolve():
        p.error('Use ir2_train Python')
    audit_dataset(Config(map_family='simple_rooms_v1',map_seed=args.map_seed,
                        train_maps=args.train_maps,validation_maps=20,test_maps=20),args.output)


if __name__=='__main__': main()
