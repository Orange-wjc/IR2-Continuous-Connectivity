"""Lossless observation storage and bounded-buffer loading of trusted checkpoints.

The model, replay order, successful episodes and random draws do not change.
Compressed observations are self-contained in the checkpoint, without sidecars.
"""
import hashlib
import pickle
from pathlib import Path
import sys
import zipfile
import zlib

import numpy as np
import torch


class PackedObservation:
    __slots__ = ('schema', 'payload')

    def __init__(self, schema, payload):
        self.schema, self.payload = schema, payload

    def unpack(self):
        buffer = zlib.decompress(self.payload)
        result, offset = [], 0
        for dtype, shape in self.schema:
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            result.append(np.frombuffer(buffer, dtype=dtype, count=int(np.prod(shape)),
                                        offset=offset).reshape(shape))
            offset += size
        if offset != len(buffer):
            raise ValueError('Compressed observation size does not match its schema')
        return tuple(result)

    def __reduce__(self):
        return (type(self), (self.schema, self.payload))


def pack_observation(observation):
    if isinstance(observation, PackedObservation):
        return observation
    schema = tuple((a.dtype.str, a.shape) for a in observation)
    return PackedObservation(schema, zlib.compress(b''.join(a.tobytes(order='C') for a in observation), 1))


def compact_state(state):
    """Replace only stored observations, preserving all scalar values and order."""
    memo = {}
    before_bytes = after_bytes = observations = 0
    def packed(obs):
        nonlocal before_bytes, after_bytes, observations
        if isinstance(obs, PackedObservation):
            return obs
        key = id(obs)
        if key not in memo:
            encoded = pack_observation(obs)
            memo[key] = encoded
            before_bytes += sum(a.nbytes for a in obs)
            after_bytes += len(encoded.payload)
            observations += 1
        return memo[key]
    def transition(t):
        return (packed(t[0]), *t[1:4], packed(t[4]), *t[5:])
    if 'replay' in state:
        state['replay'] = [transition(t) for t in state['replay']]
    for episode in state.get('success_episodes', []):
        episode['transitions'] = [transition(t) for t in episode['transitions']]
    state['observation_storage'] = 'lossless-zlib-v1'
    return {'unique_observations_encoded': observations, 'original_array_bytes': before_bytes,
            'compressed_payload_bytes': after_bytes}


def semantic_digest(value):
    """Hash logical values, ignoring observation encoding and object aliasing."""
    digest = hashlib.sha256()
    def visit(v):
        if isinstance(v, PackedObservation):
            v = v.unpack()
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            digest.update(b'tensor'); visit(a)
        elif isinstance(v, np.ndarray):
            digest.update(b'array'+v.dtype.str.encode()+repr(v.shape).encode())
            digest.update(v.tobytes(order='C'))
        elif isinstance(v, dict):
            digest.update(b'dict'+str(len(v)).encode())
            for k in sorted(v, key=lambda k:(type(k).__name__, repr(k))):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            digest.update(type(v).__name__.encode()+str(len(v)).encode())
            for x in v: visit(x)
        else:
            digest.update((type(v).__name__+':'+repr(v)).encode())
    # This field describes representation, not a changed training value.
    visit({k:v for k,v in value.items() if k != 'observation_storage'})
    return digest.hexdigest()


def load_checkpoint(path):
    """Stream the pickle instead of torch.load's full data.pkl BytesIO copy.

    Trusted local PyTorch ZIP checkpoints only; CPU map_location. Storage
    rebuilding follows the installed PyTorch 2.3 loader, including shared
    storages and historical Storage globals. No policy is instantiated here.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return torch.load(path, map_location='cpu')
    with zipfile.ZipFile(path) as archive, torch.serialization._open_zipfile_reader(str(path)) as reader:
        if reader.has_record('byteorder') and reader.get_record('byteorder').decode() != sys.byteorder:
            raise ValueError('Cross-endian checkpoints require the standard PyTorch loader')
        stored = {}
        def persistent_load(saved):
            kind, storage_type, key, location, numel = saved
            if kind not in ('storage', b'storage'):
                raise ValueError('Unknown checkpoint persistent object')
            if key not in stored:
                dtype = torch.uint8 if storage_type is torch.UntypedStorage else storage_type.dtype
                nbytes = numel * torch._utils._element_size(dtype)
                storage = reader.get_storage_from_record('data/'+key, nbytes, torch.UntypedStorage)._typed_storage()._untyped_storage
                stored[key] = torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)
            return stored[key]
        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if 'Storage' in name:
                    try:
                        return torch.serialization.StorageType(name)
                    except KeyError:
                        pass
                if module == 'torch.tensor':
                    module = 'torch._tensor'
                return super().find_class(module, name)
        entry = next(x for x in archive.namelist() if x.endswith('/data.pkl'))
        with archive.open(entry) as handle:
            unpickler = Unpickler(handle, encoding='utf-8')
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
        torch._utils._validate_loaded_sparse_tensors()
        return result
