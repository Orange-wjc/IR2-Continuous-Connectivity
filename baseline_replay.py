"""Lossless storage of original IR2 replay masks, decoded only when sampled.

The 360x360 edge masks are binary float tensors. Store their exact values in
bits and zlib instead of retaining two dense float arrays per transition.
Non-binary inputs use a lossless raw-byte fallback, including signed zero.
"""
import zlib
import numpy as np
import torch


class PackedMask:
    __slots__ = ('shape', 'dtype', 'binary', 'payload')

    def __init__(self, shape, dtype, binary, payload):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.binary = binary
        self.payload = payload

    @classmethod
    def from_tensor(cls, tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        binary = bool(np.all((array == 0) | (array == 1)))
        if binary and np.issubdtype(array.dtype, np.floating):
            binary = not bool(np.any((array == 0) & np.signbit(array)))
        raw = np.packbits(array.reshape(-1).astype(np.uint8)).tobytes() if binary else array.tobytes()
        return cls(array.shape, array.dtype.str, binary, zlib.compress(raw, 1))

    def unpack(self):
        raw = zlib.decompress(self.payload)
        count = int(np.prod(self.shape))
        if self.binary:
            if len(raw) != (count + 7) // 8:
                raise ValueError('Packed mask bit payload has the wrong size')
            array = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:count].astype(np.dtype(self.dtype))
        else:
            if len(raw) != count * np.dtype(self.dtype).itemsize:
                raise ValueError('Packed mask byte payload has the wrong size')
            array = np.frombuffer(raw, dtype=np.dtype(self.dtype)).copy()
        return torch.from_numpy(array.reshape(self.shape))


def pack_mask_rows(tensor):
    return [PackedMask.from_tensor(row) for row in tensor]


def stack_masks(rows):
    """Same dense tensors as torch.stack, including historical dense entries."""
    return torch.stack([row.unpack() if isinstance(row, PackedMask) else row for row in rows])
