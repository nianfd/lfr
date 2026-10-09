"""GPU A/B + bit-check for the fast LONM window (--lonm_fast 1).

SAFETY
------
The card is shared with the running Real-IAD experiment, so this process is
hard-capped with ``torch.cuda.set_per_process_memory_fraction`` and the bank
size is chosen to stay far inside that cap.  It also keeps the real protocol
shapes wherever they matter for the timing:

    Kr = 900   (k_image)      real
    C  = 768   (DINOv3 ViT-B/16 feature width)   real
    rho = 4 -> Kp = 81        real
    L  = H*W   reduced (both paths are O(L), so the RATIO is L-invariant)

Measured on 2026-10-04 with the RTX 4090 busy at 67% (so absolute numbers are
a lower bound); the ratio is what we report.

    python bench_window_fast.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curvot_utils import (                                     # noqa: E402
    build_cand_context, collect_candidates, lonm_score,
    window_top_candidates)

CAP_FRAC = 0.05          # <=1228 MiB of the 24564 MiB card; the run needs ~800
REPEATS = 5
CHUNK = 32
M = 24
ARM_B = dict(M=M, d_max=4, quadratic=False, norm_scale=True,
             norm_mode='raw', scale_q=0.9, chunk=CHUNK)


def traffic_gb(Kr, L, C, Kp):
    """Bytes the gather path touches per layer (read + write + permute + bmm
    read), for the honest bandwidth comparison."""
    return 5.0 * Kr * L * Kp * C * 4 / 1e9


def run_case(H, W, r, Kr, C, label):
    dev = 'cuda'
    L = H * W
    Kp = (2 * r + 1) ** 2
    g = torch.Generator(device='cpu').manual_seed(0)
    bank = torch.nn.functional.normalize(
        torch.randn(Kr, L, C, generator=g), dim=-1).to(dev)
    q = torch.nn.functional.normalize(
        torch.randn(L, C, generator=g), dim=-1).to(dev)
    ctx = build_cand_context(H, W, r, dev)
    Msel = min(M, Kr * ctx['Kp'])
    torch.cuda.synchronize()
    peak0 = torch.cuda.max_memory_allocated() / 2 ** 20

    print(f'\n== {label} ==')
    print(f'   H={H} W={W} L={L} r={r} Kp={ctx["Kp"]} Kr={Kr} C={C} '
          f'(gather path traffic ~{traffic_gb(Kr, L, C, Kp):.1f} GB/layer)')

    got = {}
    for fl in (False, True):
        tag = 'FAST' if fl else 'gather'
        for _ in range(2):                                     # warm-up
            for _, _, Z in window_top_candidates(q, bank, ctx, Msel,
                                                 chunk=CHUNK, fast=fl):
                got.setdefault(tag, []).append(Z.clone())
            got[tag] = got[tag][-L // CHUNK - 1:]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            for _, _, Z in window_top_candidates(q, bank, ctx, Msel,
                                                 chunk=CHUNK, fast=fl):
                pass
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / REPEATS
        got[tag + '_t'] = dt
        print(f'   {tag:7s} window_top_candidates  {dt*1e3:9.2f} ms/layer   '
              f'({traffic_gb(Kr,L,C,Kp)/max(dt,1e-9):7.1f} GB/s effective)')

    sp = got['gather_t'] / max(got['FAST_t'], 1e-9)
    print(f'   -> speed-up on this stage: {sp:6.1f}x   '
          f'({got["gather_t"]*1e3:.1f} ms -> {got["FAST_t"]*1e3:.2f} ms)')

    # ---- bit-check on real GPU kernels, real Kr/C/Kp ----------------------
    zs = {}
    for fl in (False, True):
        out = []
        for _, _, Z in window_top_candidates(q, bank, ctx, Msel,
                                             chunk=CHUNK, fast=fl):
            out.append(Z)
        zs[fl] = torch.cat(out, dim=0)
    zd = float((zs[False] - zs[True]).abs().max())
    print(f'   Z  max|fast-gather| = {zd:.3e}   '
          f'bit-equal = {bool(torch.equal(zs[False], zs[True]))}')

    s = {fl: lonm_score(q, bank, ctx, fast_window=fl, **ARM_B)
         for fl in (False, True)}
    # self-consistency control: if lonm_score is not bit-stable against ITSELF,
    # the fast-vs-gather difference has to be read against that floor and not
    # against zero.
    s2 = {fl: lonm_score(q, bank, ctx, fast_window=fl, **ARM_B)
          for fl in (False, True)}
    for fl in (False, True):
        tag = 'FAST' if fl else 'gather'
        d = float((s[fl] - s2[fl]).abs().max())
        print(f'   {tag:7s} self-consistency max|run1-run2| = {d:.3e}')
    sd = float((s[False] - s[True]).abs().max())
    den = s[False].abs().clamp_min(1e-12)
    rel = float(((s[False] - s[True]).abs() / den).max())
    print(f'   lonm_score max|fast-gather| = {sd:.3e}  max rel {rel:.3e}  '
          f'bit-equal = {bool(torch.equal(s[False], s[True]))}')
    print(f'   accepted precedent for this layout change (test_agg.py): '
          f'rel ~4e-10, hard max 1e-7  -> '
          f'{"WITHIN" if rel < 1e-7 else "OUTSIDE"}')

    # ---- whole-stage timing: window + post-processing together -----------
    # The gather path is pinned to one patch position per iteration by
    # collect_candidates' mem_mb=256 guard (sub == 1 at rho=4), so its
    # post-processing runs 784 single-position kernel launches per layer.
    # The fast path batches `chunk` positions, so the SAME arithmetic gets
    # real batch sizes -- that is why chunk has to move together with the flag.
    print('   -- whole lonm_score stage (per layer) --')
    for tag, fl, ck in (('gather  ck=32 ', False, 32),
                        ('FAST    ck=32 ', True, 32),
                        ('FAST    ck=128', True, 128)):
        flags = dict(ARM_B, chunk=ck)
        for _ in range(2):
            lonm_score(q, bank, ctx, fast_window=fl, **flags)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            lonm_score(q, bank, ctx, fast_window=fl, **flags)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / REPEATS
        print(f'      {tag}  {dt*1e3:9.2f} ms/layer')
        got[f'lonm_{tag}'] = dt
    base = got['lonm_gather  ck=32 ']
    for tag in ('FAST    ck=32 ', 'FAST    ck=128'):
        print(f'      -> {tag}: {base / max(got["lonm_" + tag], 1e-9):5.2f}x '
              f'vs the shipped path (whole stage)')
    sp = base / max(got['lonm_FAST    ck=128'], 1e-9)
    print(f'   ==> LONM stage speed-up to take to the ETA: {sp:.2f}x')

    del bank, q, ctx, got, zs, s
    torch.cuda.empty_cache()
    return sp


def main():
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device')
    torch.cuda.set_per_process_memory_fraction(CAP_FRAC)
    p = torch.cuda.get_device_properties(0)
    print(f'device: {p.name}  {p.total_memory/2**30:.1f} GiB  '
          f'cap: {CAP_FRAC*100:.0f}% = {CAP_FRAC*p.total_memory/2**20:.0f} MiB')
    torch.cuda.reset_peak_memory_stats()

    sps = []
    sps.append(run_case(14, 14, 4, 900, 768, 'real Kr/C/Kp, L reduced 1/4'))
    if '--small' not in sys.argv:
        try:
            sps.append(run_case(28, 28, 4, 100, 768,
                                'real L and Kp, Kr reduced'))
        except torch.OutOfMemoryError as e:
            # the card is shared; a smaller-bank second case is optional
            print(f'   [skip] second case OOM (shared card): '
                  f'{str(e).splitlines()[0]}')
            torch.cuda.empty_cache()
    print(f'\npeak torch allocation this process: '
          f'{torch.cuda.max_memory_allocated()/2**20:.0f} MiB (cap '
          f'{CAP_FRAC*p.total_memory/2**20:.0f} MiB)')
    print(f'geometric mean speed-up: '
          f'{float(torch.tensor(sps).log().mean().exp()):.1f}x')


if __name__ == '__main__':
    main()
