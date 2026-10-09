#!/usr/bin/env python
"""Launch `curvot_rad.py` with a shared-RO mmap bank loader.

WHY (2026-09-29, Real-IAD chain, diagnosed and reproduced)
----------------------------------------------------------
`load_bank(use_mmap=True)` calls `torch.load(path, map_location='cpu', mmap=True)`,
which maps the WHOLE checkpoint in one *private/accountable* mapping.  Under this
box's heuristic overcommit policy (vm.overcommit_memory=0) the kernel refuses any
accountable mapping larger than RAM+swap:

    torch.load(mmap=True) on the 352 GB Real-IAD bank
        -> RuntimeError: unable to mmap 351744311196 bytes ... (errno 12)
    python mmap, whole file, ACCESS_COPY (private)   -> OSError ENOMEM
    python mmap, whole file, ACCESS_READ (shared RO) -> OK (351744311196 bytes)
    anonymous 200 GB                                 -> OSError ENOMEM
    anonymous 100 GB                                 -> OK

`load_bank` swallows that RuntimeError in its own `except Exception` fallback and
retries with `torch.load(...)` eager, so every Real-IAD class died on the OOM
killer after ~4.5 min at ~124 GB RSS (17 kills in dmesg; VmData 83 GB, ZERO
bank.pth entries in /proc/<pid>/maps).  VisA's 77 GiB bank never showed it
because 77 GiB fits in RAM and both paths worked.

WHAT THIS DOES
--------------
Swaps in `ri_sharedbank.load_bank`, a signature-compatible drop-in that maps the
same file with mmap.ACCESS_READ (MAP_SHARED|PROT_READ -- not overcommit-accounted,
so it succeeds at full size) and hands out zero-copy views into it.  Same bytes,
same dtype, same shape, same order -- this only restores what `load_bank`'s own
docstring says the mmap path is for.

Verified before use (exp/q_bankverify.py):
  * pickle structure / offsets / strides read from the checkpoint, not assumed
  * all 8 storages bit-equal to a plain read at the same file offset
  * CLS class-block purity 1.0000 -- the pipeline's own audit (alphabetical
    order scores 0.7716, so a wrong storage->attribute assignment cannot pass)
  * RSS 336 MB with the 352 GB bank mapped, 1 file-backed mapping

USAGE: identical to `python curvot_rad.py <args...>`.  The launcher's filename
still ends in `curvot_rad.py` on purpose: the drivers guard single-writer access
with `pgrep -f 'curvot_rad\.py'`, and that must keep matching.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ri_sharedbank as _sb                            # noqa: E402
import rad_mvtec_visa_3dadam as _lib                   # noqa: E402

_lib.load_bank = _sb.load_bank

import curvot_rad                                      # noqa: E402

# curvot_rad did `from rad_mvtec_visa_3dadam import load_bank`, which binds its
# own name -- that binding is what its main() actually calls, so patch it too.
curvot_rad.load_bank = _sb.load_bank


def main():
    sys.argv[0] = os.path.join(_REPO, 'curvot_rad.py')
    print('[launch] bank loader = shared-RO mmap '
          '(exp/ri_sharedbank.py); torch.load(mmap=True) is unusable for the '
          '352 GB Real-IAD bank on this box', flush=True)
    curvot_rad.main()


if __name__ == '__main__':
    main()
