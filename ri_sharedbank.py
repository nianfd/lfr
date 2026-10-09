"""Shared-RO mmap bank loader -- drop-in replacement for load_bank().

WHY THIS EXISTS (2026-09-29, Real-IAD chain)
--------------------------------------------
`load_bank(use_mmap=True)` calls `torch.load(path, map_location='cpu', mmap=True)`.
On this box that call FAILS for the Real-IAD bank and the failure is swallowed by
`load_bank`'s own `except Exception` fallback, which then tries the eager path.
Measured on amax (kernel 5.15, 125 GB RAM, vm.overcommit_memory=0):

    torch.load(..., mmap=True)
      -> RuntimeError: unable to mmap 351744311196 bytes from file
         <...bank.pth>: Cannot allocate memory (12)

    351744311196 == os.path.getsize(bank.pth), i.e. torch maps the WHOLE file in
    one call, and it does so with an ACCOUNTABLE (private, VM_MAYWRITE) mapping.
    Under the heuristic overcommit policy (vm.overcommit_memory=0) the kernel
    refuses any accountable mapping larger than RAM+swap:

        mmap whole file, ACCESS_COPY (private)   -> OSError ENOMEM   [reproduced]
        mmap whole file, ACCESS_READ (shared RO) -> OK, len 351744311196
        anon 200 GB                              -> OSError ENOMEM
        anon 100 GB                              -> OK

    So the 352 GB private mapping is refused, `load_bank` falls back to eager
    loading, 352 GB does not fit in 125 GB, and every Real-IAD run dies on the
    OOM killer after ~4.5 min at ~124 GB RSS (17 consecutive kills in dmesg).
    The VisA bank (77 GiB) never hit this because 77 GiB < RAM, so BOTH paths
    fit and the broken mmap went unnoticed.

WHAT THIS DOES
--------------
Maps the same file with mmap.ACCESS_READ -- i.e. MAP_SHARED|PROT_READ, which is
NOT overcommit-accounted and therefore succeeds at full size -- and hands out
plain views into it.  This is not a reinterpretation of the checkpoint: it is
byte-for-byte the same file, zero-copy, and precisely what load_bank's own
docstring says the mmap path is for ("两者读的是同一份字节，数值完全一致").

The container is an uncompressed zip, so the raw storage bytes sit at
contiguous file offsets; the mapping and the per-storage offsets/strides are
recovered from the checkpoint's own pickle, never assumed.  Numerically
identical to a successful `torch.load(mmap=True)`; the only difference is that
this one fits in the address space the kernel is willing to hand out.
"""
import io
import mmap
import os
import pickle
import struct
import zipfile

import torch

_LOG = []
_TENSOR_KEYS = {}


class _Ref:
    """A storage reference produced by the checkpoint pickle."""

    __slots__ = ('key', 'dtype', 'numel')

    def __init__(self, key, dtype, numel):
        self.key = key
        self.dtype = dtype
        self.numel = numel

    def __repr__(self):
        return f'<storage {self.key} {self.dtype} numel={self.numel}>'


def _dtype_from_name(name):
    n = str(name).lower()
    if 'untyped' in n:
        # the zip container names the storage but not its element type; the
        # builder wrote float32 and _load_shared_ro asserts the byte count
        # against the entry size, so a wrong guess fails loudly, not silently.
        return torch.float32
    if 'double' in n:
        return torch.float64
    if 'half' in n:
        return torch.float16
    if 'bfloat16' in n:
        return torch.bfloat16
    if 'float' in n:
        return torch.float32
    if 'long' in n:
        return torch.int64
    if 'int' in n:
        return torch.int32
    if 'bool' in n:
        return torch.bool
    raise RuntimeError(f'[sharedbank] unsupported storage dtype {name!r}')


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
                       backward_hooks, *a, **k):
    return ('tensor', storage, tuple(int(s) for s in size),
            tuple(int(s) for s in stride), int(storage_offset))


class _StructureUnpickler(pickle.Unpickler):
    """Reads the checkpoint's pickle WITHOUT allocating any storage."""

    def persistent_load(self, pid):
        # pid == ('storage', storage_type, key, location, numel)
        _, stype, key, _loc, numel = pid
        name = getattr(stype, '__name__', str(stype))
        return _Ref(key, _dtype_from_name(name), int(numel))

    def find_class(self, module, name):
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return _rebuild_tensor_v2
        return super().find_class(module, name)


def _entry_offsets(path):
    """data-offset / stored-size of every `data/<n>` entry (must be STORED).

    The 32-bit size fields in a local file header are zero here because this
    checkpoint is >4 GiB and uses ZIP64: the real sizes live in the extra field
    and are already resolved for us in the central directory, so take them from
    ZipInfo and use the local header only to locate the payload.
    """
    out = {}
    with zipfile.ZipFile(path) as z:
        infos = [zi for zi in z.infolist()
                 if zi.filename.split('/')[-2:-1] == ['data']
                 and zi.filename.split('/')[-1].isdigit()]
        with open(path, 'rb') as fh:
            for zi in infos:
                fh.seek(zi.header_offset)
                hdr = fh.read(30)
                (sig, _ver, _flag, comp, _t, _d, _crc, _csize, _usize,
                 fnlen, extralen) = struct.unpack('<IHHHHHIIIHH', hdr)
                if sig != 0x04034B50:
                    raise RuntimeError(f'[sharedbank] bad local header for '
                                       f'{zi.filename}')
                if zi.compress_type != 0 or comp != 0:
                    raise RuntimeError(
                        f'[sharedbank] {zi.filename} is compressed '
                        f'(method={comp}); cannot map it directly')
                if zi.compress_size != zi.file_size:
                    raise RuntimeError(f'[sharedbank] {zi.filename}: '
                                       f'csize={zi.compress_size} != '
                                       f'usize={zi.file_size}')
                off = zi.header_offset + 30 + fnlen + extralen
                out[str(zi.filename.split('/')[-1])] = (off, zi.file_size)
    return out


def _basename_of_pickle(path):
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith('/data.pkl'):
                return n
    raise RuntimeError('[sharedbank] no data.pkl in the checkpoint')


def _read_structure(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(_basename_of_pickle(path))
    return _StructureUnpickler(io.BytesIO(raw)).load()


def _materialise(obj, make):
    if isinstance(obj, tuple) and obj and obj[0] == 'tensor':
        return make(obj)
    if isinstance(obj, dict):
        return type(obj)((k, _materialise(v, make)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(_materialise(v, make) for v in obj)
    return obj


def load_bank(bank_path, use_mmap=True, print_fn=print):
    """Signature-compatible with rad_mvtec_visa_3dadam.load_bank()."""
    if not use_mmap:
        return torch.load(bank_path, map_location='cpu'), 'eager RAM'
    try:
        ckpt, mode = _load_shared_ro(bank_path, print_fn)
        return ckpt, mode
    except Exception as exc:                       # noqa: BLE001
        print_fn(f'[bank] shared-RO mmap 加载失败({type(exc).__name__}: {exc})，'
                 f'回退常规加载')
    return torch.load(bank_path, map_location='cpu'), 'eager RAM'


def _load_shared_ro(bank_path, print_fn=print):
    size = os.path.getsize(bank_path)
    fh = open(bank_path, 'rb')
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)   # MAP_SHARED|PROT_READ
    _LOG.append((fh, mm, bank_path))                          # keep alive
    mv = memoryview(mm)

    offs = _entry_offsets(bank_path)
    struct_obj = _read_structure(bank_path)

    def make(desc):
        _, ref, shape, stride, soff = desc
        key = str(ref.key)
        if key not in _TENSOR_KEYS:
            if key not in offs:
                raise RuntimeError(f'[sharedbank] storage {key!r} not in the zip '
                                   f'(have {sorted(offs)})')
            off, usize = offs[key]
            want = ref.numel * ref.dtype.itemsize
            if want != usize:
                raise RuntimeError(f'[sharedbank] storage {key}: pickle says '
                                   f'{want} bytes ({ref.numel} x {ref.dtype}), '
                                   f'entry holds {usize}')
            flat = torch.frombuffer(mv[off:off + usize], dtype=ref.dtype)
            _TENSOR_KEYS[key] = flat
        flat = _TENSOR_KEYS[key]
        numel = 1
        for s in shape:
            numel *= s
        return torch.as_strided(flat, size=shape, stride=stride,
                                storage_offset=soff)

    ck = _materialise(struct_obj, make)
    n_map = sum(1 for l in open('/proc/self/maps') if bank_path in l.split()[-1:])
    print_fn(f'[bank] shared-RO mmap: {size / 2**30:.2f} GiB mapped, '
             f'{len(_TENSOR_KEYS)} storages, {n_map} mapping(s), '
             f'dtype={next(iter(_TENSOR_KEYS.values())).dtype}')
    return ck, 'mmap shared-RO (lazy)'


def shared_bank_tensors(path, print_fn=print):
    """Explicit entry point used by the validator / launcher."""
    ck, _ = _load_shared_ro(path, print_fn)
    return ck


if __name__ == '__main__':
    import sys
    ck = shared_bank_tensors(sys.argv[1])
    for k, v in ck.items():
        if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            print(k, 'x', len(v), [tuple(t.shape) for t in v])
        else:
            print(k, '=', v)
