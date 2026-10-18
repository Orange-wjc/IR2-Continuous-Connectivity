"""Original DungeonMaps inventory and deterministic train/development separation.

No generated maps, resampling, geometry repair, or test-set training.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from skimage import io


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original(path):
    a = io.imread(str(path), as_gray=True)
    # Exactly the upstream import_ground_truth conversion, including RGBA PNGs.
    if np.issubdtype(a.dtype, np.floating) and a.max() <= 1:
        a = a * 255
    gray = a.astype(int)
    marker = np.argwhere(gray == 208)
    if len(marker) < 128:
        raise ValueError('Missing original spawn marker: ' + str(path))
    start = marker[127][::-1].astype(np.float64)
    truth = np.where(gray > 150, 255, 1).astype(np.uint8)
    return truth, start


def canonical(truth):
    return min(hashlib.sha256(np.packbits(np.rot90(a,k)).tobytes()+str(np.rot90(a,k).shape).encode()).hexdigest()
               for a in (truth==255,np.fliplr(truth==255)) for k in range(4))


def prepare_dataset(repo, output, development_maps=100):
    """Every original training file belongs to train or development, never dropped."""
    repo, output = Path(repo), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result={'repository':str(repo.resolve()),'families':{},'test_policy_executions':0,
            'geometry':'Original pixels and dimensions; upstream >150 threshold',
            'development_selection':'Native geometry groups kept together; SHA256 geometry rank, whole groups until at least 100 files. No geometry used for training is resampled.',
            'test_usage':'Original test directories reserved; never used for labels, gradients or checkpoint selection'}
    for family in ('easy','difficult'):
        files=sorted((repo/'DungeonMaps/train'/family).glob('*.png'),reverse=True)
        assert len(files)==(9999 if family=='easy' else 6000)
        entries=[]
        for path in files:
            relative=str(path.relative_to(repo));h=digest(path)
            truth,start=load_original(path)
            # A grouping signature only: detect the dataset's 500/1000 copies.
            normalized=truth[::2,::2] if truth.shape==(1000,1000) else truth
            group=canonical(normalized)
            entries.append({'path':relative,'file_sha256':h,'geometry_group':group,
                            'shape':list(truth.shape),'free_cells':int((truth==255).sum())})
        grouped={}
        for e in entries:grouped.setdefault(e['geometry_group'],[]).append(e['path'])
        dev_paths=set()
        for key in sorted(grouped,key=lambda g:hashlib.sha256(('native_dev_20261015:'+g).encode()).hexdigest()):
            if len(dev_paths)>=development_maps:break
            dev_paths.update(grouped[key])
        splits={'train':[e for e in entries if e['path'] not in dev_paths],
                'validation':[e for e in entries if e['path'] in dev_paths]}
        assert len(splits['train'])>1000
        assert not ({e['geometry_group'] for e in splits['train']} & {e['geometry_group'] for e in splits['validation']})
        result['families'][family]={'original_training_files':len(entries),'distinct_geometry_groups':len(grouped),'splits':splits}
    result['original_test_counts']={name:len(list((repo/'DungeonMaps/test'/name).glob('*.png')))
                                    for name in ('hybrid','corridor','complex')}
    result['original_test_files']={name:[dict(path=str(p.relative_to(repo)),file_sha256=digest(p))
        for p in sorted((repo/'DungeonMaps/test'/name).glob('*.png'),reverse=True)]
        for name in ('hybrid','corridor','complex')}
    path=output/'native_dataset.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    return path


@lru_cache(maxsize=4)
def dataset(path):
    return json.loads(Path(path).read_text())


def original_map(config,split,index):
    d=dataset(config.dataset_manifest)
    if split not in ('train','validation','test'):
        raise ValueError('The reserved original test set requires a separate final evaluation command')
    family=config.map_family.removeprefix('dungeon_') if hasattr(str,'removeprefix') else config.map_family[len('dungeon_'):]
    if split=='test':
        if config.native_test_set not in ('hybrid','corridor','complex'):
            raise ValueError('Original test set is reserved; an explicit final test set is required')
        entry=d['original_test_files'][config.native_test_set][index]
    else:entry=d['families'][family]['splits'][split][index]
    path=Path(d['repository'])/entry['path']
    assert digest(path)==entry['file_sha256'], 'Original map file changed'
    truth,start=load_original(path)
    starts=np.stack([start,start])
    info=dict(split=split,index=index,layout_id=entry['path'],family=config.map_family,
              source_file=entry['path'],file_sha256=entry['file_sha256'],
              sha256=hashlib.sha256(truth.tobytes()).hexdigest(),units='native map pixels',
              native_shape=list(truth.shape),starts=starts.tolist())
    return truth,starts,info
