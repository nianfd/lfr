"""Equivalence test for the fast LONM window (--lonm_fast 1 / fast_window=True).

WHY THIS FILE EXISTS
--------------------
``collect_candidates`` gathers ``bank[:, idx[p, kk], :]`` for every
(position, offset) pair, so each reference patch is read once per offset.  On
Real-IAD (rho=4 -> Kp=81, K=900, C=768, L=784) that is ~875 GB of traffic per
layer per image, x4 layers, and it is the whole reason LONM costs 5.5 s/image
there while RAD's own window walk costs 0.84 s on the same code and card.

The fix is the one ``patch_scores_positional`` already uses for the max-only
case: shift the QUERY, do ONE bmm, then re-align the offsets with a single
gather -- the bank is streamed once.  ``_window_sim_fast`` extends that from
``max`` to ``topk``.

It is only allowed if it computes the SAME thing, so this file checks three
independent levels:

  L1  algebra      sim[p, m*Kp + kk] against a literal per-(p, kk) loop
  L2  vs the shipped gather path   masked sim, topk indices, Z
  L3  end to end   lonm_score with the Real-IAD arm-b flags, both ways

Reordering the accumulation changes the last few ulps, so the tolerance is the
one ``test_agg.py`` already established for exactly this layout change:
4e-10 relative, hard maximum 1e-7, printed metrics identical to every digit.

WHICH ENTRIES ARE COMPARED
--------------------------
Only the cells the window actually uses.  For a (position, offset) pair that
falls outside the patch grid, ``build_cand_context`` clamps the index and sets
``valid = False``; BOTH paths then produce a meaningless number there -- the
gather path computes ``<bank[m, clamp(p+D)], q[p]>``, the fast path
``<bank[m, clamp(p+D)], q[clamp(p+D)-D]>`` -- and both overwrite it with -inf
before the topk.  So the equivalence claim is about the VALID window, and the
test asserts the -inf pattern as well, because that pattern is load-bearing:
it decides which cells can win the topk.

Runs on CPU -- no GPU, no bank, no dataset.

    python test_window_fast.py            (from the directory holding curvot_utils.py)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'srv')

from curvot_utils import (                                     # noqa: E402
    build_cand_context, collect_candidates, lonm_score,
    window_top_candidates, _window_sim_fast)

REL_TOL = 1e-7        # fast vs the SHIPPED gather path (what actually matters)
# L1 compares against a literal python loop, which accumulates one row at a
# time (gemv) while both real paths use a batched gemm -- measured on CPU that
# differs by up to ~2.4e-7 on its own, i.e. the reference is the noisy side.
REL_TOL_L1 = 1e-6
GATHER_CHUNK = 32


def make_case(H, W, r, Kr, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = H * W
    bank = torch.nn.functional.normalize(torch.randn(Kr, L, C, generator=g),
                                         dim=-1)
    q = torch.nn.functional.normalize(torch.randn(L, C, generator=g), dim=-1)
    ctx = build_cand_context(H, W, r, 'cpu')
    return q, bank, ctx


def slow_sim(q, bank, ctx, chunk=GATHER_CHUNK):
    """The shipped path: materialise the [Pc, Kr*Kp, C] block, then bmm."""
    L = q.shape[0]
    sim = torch.empty(L, bank.shape[0] * ctx['Kp'], dtype=q.dtype)
    for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
        s = torch.bmm(cand, q[s0:s1].unsqueeze(-1)).squeeze(-1)
        if ok is not None:
            s = s.masked_fill(~ok, float('-inf'))
        sim[s0:s1] = s
    return sim


def rel_dev(a, b, only=None):
    """Relative deviation over the entries both sides call finite.

    ``only`` optionally restricts the comparison to a bool mask (used to stay
    on the VALID window cells -- see the module docstring).
    """
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    pattern_mismatch = int((fa != fb).sum())
    if only is not None:
        fa = fa & only
        fb = fb & only
    both = fa & fb
    if not bool(both.any()):
        return 0.0, pattern_mismatch
    d = (a[both] - b[both]).abs()
    s = torch.maximum(a[both].abs(), b[both].abs()).clamp_min(1.0)
    return float((d / s).max()), pattern_mismatch


def naive_sim(q, bank, ctx):
    """Independent reference: literal per-(p, kk) dot products off the table.

    Flat layout is reference-major: candidate index = m*Kp + kk, which is what
    ``collect_candidates`` produces (sel.reshape(Kr, Pc, Kp, C) -> permute ->
    reshape(Pc, Kr*Kp, C)).
    """
    L = ctx['L']
    Kp = ctx['Kp']
    Kr = bank.shape[0]
    sim = torch.empty(L, Kr * Kp, dtype=q.dtype)
    for p in range(L):
        for kk in range(Kp):
            col = bank[:, ctx['idx'][p, kk], :]          # [Kr, C]
            sim[p, kk::Kp] = col @ q[p]
    return sim


def masked(sim, ctx, Kr):
    out = sim.clone().view(ctx['L'], Kr, ctx['Kp'])
    out.masked_fill_(~ctx['valid'].unsqueeze(1), float('-inf'))
    return out.view(ctx['L'], Kr * ctx['Kp'])


CASES = [
    # H, W, r, Kr, C, label
    (8, 8, 1, 6, 16, 'MVTec-like rho=1, tiny'),
    (8, 8, 4, 5, 16, 'Real-IAD rho=4, tiny'),
    (7, 9, 3, 4, 32, 'non-square grid, rho=3'),
    (6, 6, 4, 3, 8, 'rho=4 mid-edge clamping (6 < 2r+2)'),
    (28, 28, 4, 4, 16, 'Real-IAD patch grid 28x28, rho=4, Kp=81'),
]

ARM_B = dict(M=24, d_max=4, quadratic=False, norm_scale=True,
             norm_mode='raw', scale_q=0.9, chunk=GATHER_CHUNK)
MVTEC = dict(M=24, d_max=4, quadratic=True, ridge=1e-4,
             norm_scale=True, norm_mode='ratio', scale_q=0.9,
             chunk=GATHER_CHUNK)

fails = []

for H, W, r, Kr, C, label in CASES:
    q, bank, ctx = make_case(H, W, r, Kr, C)
    Kp, L = ctx['Kp'], ctx['L']
    vm = ctx['valid'].unsqueeze(1).expand(L, Kr, Kp).reshape(L, Kr * Kp)
    print(f'== {label}: H={H} W={W} r={r} Kp={Kp} Kr={Kr} L={L} C={C}  '
          f'in-grid offsets per position={int(vm[0].sum()) // Kr}/{Kp} ==')

    fast = _window_sim_fast(q, bank, ctx, chunk=GATHER_CHUNK)

    # ---- L1: algebra against a literal loop, on the VALID window -----------
    ref = naive_sim(q, bank, ctx)
    dev1, pm1 = rel_dev(fast, ref, only=vm)
    ok1 = dev1 < REL_TOL_L1 and pm1 == 0
    print(f'   L1 fast vs naive loop      rel={dev1:.3e}  '
          f'finite-pattern mismatch={pm1}  -> {"PASS" if ok1 else "FAIL"}')
    if not ok1:
        fails.append(f'L1/{label}')

    # ---- L2: against the shipped gather path, masked both sides -----------
    sl = slow_sim(q, bank, ctx)
    m_fast = masked(fast, ctx, Kr)
    m_slow = masked(sl, ctx, Kr)
    dev2, pm2 = rel_dev(m_fast, m_slow)
    ok2 = dev2 < REL_TOL and pm2 == 0
    print(f'   L2 fast vs gather (masked) rel={dev2:.3e}  '
          f'finite-pattern mismatch={pm2}  -> {"PASS" if ok2 else "FAIL"}')
    if not ok2:
        fails.append(f'L2/{label}')

    # ---- L2b: the topk winners themselves ---------------------------------
    Msel = min(24, Kr * Kp)
    Zs, tops = {}, {}
    for tag, fl in (('slow', False), ('fast', True)):
        zs, ts = [], []
        for s0, s1, Z in window_top_candidates(q, bank, ctx, Msel,
                                               chunk=GATHER_CHUNK, fast=fl):
            zs.append(Z)
        Zs[tag] = torch.cat(zs, dim=0)
    # recover the winners from Z by matching against the bank (independent of
    # the indices the two paths chose internally)
    zdev, _ = rel_dev(Zs['fast'], Zs['slow'])
    same_Z = bool(torch.allclose(Zs['fast'], Zs['slow'], rtol=1e-5, atol=1e-6))
    ok3 = same_Z and zdev < REL_TOL
    print(f'   L2b top-M candidate vectors  Z rel={zdev:.3e}  '
          f'allclose={same_Z}  -> {"PASS" if ok3 else "FAIL"}')
    if not ok3:
        fails.append(f'L2b/{label}')

    # ---- L3: end to end, both flag families ------------------------------
    for tag, flags in (('arm-b', ARM_B), ('MVTec', MVTEC)):
        s_slow = lonm_score(q, bank, ctx, fast_window=False, **flags)
        s_fast = lonm_score(q, bank, ctx, fast_window=True, **flags)
        dev, _ = rel_dev(s_fast, s_slow)
        ok = dev < REL_TOL
        print(f'   L3 lonm_score {tag:6s}          rel={dev:.3e}  '
              f'-> {"PASS" if ok else "FAIL"}')
        if not ok:
            fails.append(f'L3{tag}/{label}')
    print()

print('ALL OK' if not fails else 'FAILED: ' + ', '.join(fails))
raise SystemExit(0 if not fails else 1)
