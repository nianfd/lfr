"""Bit-exactness + branch test for the ``unbounded`` / ``clip_stat`` patch.

Two things must hold before the radius ablation can be trusted:

  (1) With ``unbounded=False`` (the default) ``lonm_score`` must be
      BIT-IDENTICAL to the pre-patch code.  This isolates the patch from every
      older refactor -- notably the 2026-10-04 ``--lonm_fast`` window rewrite,
      which moves results at the 1e-12 level and therefore rules out a
      historical-run comparison.

  (2) With ``unbounded=True`` and clipping actually active, the score must
      equal the independent closed form  s = ||q - (z0 + T^T a)||  (no clip),
      while the bounded arm must equal  s = ||q - (z0 + T^T (alpha a))||.

  (3) ``clip_stat`` must be numerically INERT -- only the diagnostics move.
      The radius arms carry ``--lonm_clipstat 1`` and the anchors do not, so
      without this the paired radius difference would carry a second change.

Case B uses STRUCTURED features (a shared low-rank direction plus small
normal scatter, and queries that travel far along that direction) because
independent random unit vectors in C = 768 are mutually near-orthogonal and
never leave the fitted radius -- the random case cannot exercise the branch.

Run:  PYTHONPATH=. python exp/test_unbounded_equiv.py
"""
import sys

import torch
import torch.nn.functional as F

import curvot_utils as cu
import cu_ref

L, Kr, C, RHO = 784, 16, 768, 1


def random_case(seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = F.normalize(torch.randn(L, C, generator=g, device='cuda'), dim=-1)
    bank = F.normalize(torch.randn(Kr, L, C, generator=g, device='cuda'), dim=-1)
    return q, bank


def structured_case(seed=1, spread=0.01, excursion=1.0, kr=16):
    """Normals vary by ~``spread`` along a 3-d basis; the query travels
    ~``excursion`` along the same basis.  The query stays cosine-close enough
    to be selected among the M nearest, yet its retained coordinate exceeds
    the normals' own extent -- which is exactly when alpha < 1 and the finite
    radius changes the readout.  Independent random unit vectors in C = 768
    are mutually near-orthogonal and can never produce this.
    """
    g = torch.Generator(device='cuda').manual_seed(seed)
    b = F.normalize(torch.randn(C, generator=g, device='cuda'), dim=-1)     # [C]
    U = torch.linalg.qr(torch.randn(C, 3, generator=g, device='cuda'))[0]   # [C,3]
    coef = spread * torch.randn(kr, L, 3, generator=g, device='cuda')
    bank = F.normalize(b + coef @ U.t(), dim=-1)                           # [kr,L,C]
    qc = torch.zeros(L, 3, device='cuda')
    qc[:, 0] = excursion
    q = F.normalize(b + qc @ U.t(), dim=-1)                                # [L,C]
    return q, bank


def prod_cfg(**over):
    kw = dict(M=24, d_max=4, energy=0.9, trust_q=0.9, quadratic=False,
              norm_scale=True, norm_mode='raw')
    kw.update(over)
    return kw


@torch.no_grad()
def closed_form(q, bank, ctx, kw, chunk=128):
    """s_inf and s_R re-derived from the raw tensors, mirroring the code."""
    M, d_max, energy, trust_q = kw['M'], kw['d_max'], kw['energy'], kw['trust_q']
    eps, min_var = 1e-8, 1e-12
    Lq = q.shape[0]
    d_R = torch.empty(Lq, device=q.device, dtype=q.dtype)
    d_inf = torch.empty(Lq, device=q.device, dtype=q.dtype)
    dmax = d_max if kw.get('quadratic', False) else max(d_max, 1)
    Msel = min(M, bank.shape[0] * int(ctx['Kp']))
    for s0, s1, Z in cu.window_top_candidates(q, bank, ctx, Msel, chunk=chunk):
        qq = q[s0:s1]
        z0 = Z.mean(dim=1, keepdim=True)
        X = Z - z0
        G = torch.bmm(X, X.transpose(1, 2))
        evals, evecs = torch.linalg.eigh(G)
        evals = evals.flip(-1).clamp_min(0.0)
        evecs = evecs.flip(-1)
        d = min(dmax, Msel)
        Sd = evals[:, :d].clamp_min(min_var).sqrt()
        Ud = evecs[:, :, :d]
        T = (torch.bmm(X.transpose(1, 2), Ud) / Sd.unsqueeze(1)).transpose(1, 2)
        y = Ud * Sd.unsqueeze(1)
        ev = evals[:, :d]
        cum = torch.cumsum(ev, dim=-1) / ev.sum(dim=-1, keepdim=True).clamp_min(min_var)
        d_p = ((cum < energy).sum(dim=-1) + 1).clamp(min=2, max=d)
        jj = torch.arange(d, device=q.device).view(1, d)
        mask = (jj < d_p.view(-1, 1)).to(q.dtype).unsqueeze(1)
        yn = torch.linalg.norm(y * mask, dim=-1)
        kq = min(Msel - 1, max(0, int(round(trust_q * (Msel - 1)))))
        r_trust = torch.sort(yn, dim=-1).values[:, kq]
        yq = torch.bmm(qq.unsqueeze(1) - z0, T.transpose(1, 2)).squeeze(1)
        yq = yq * mask.squeeze(1)
        nyq = torch.linalg.norm(yq, dim=-1).clamp_min(eps)
        scale = torch.clamp(r_trust / nyq, max=1.0).unsqueeze(-1)
        hat_R = z0.squeeze(1) + torch.bmm((yq * scale).unsqueeze(1), T).squeeze(1)
        hat_i = z0.squeeze(1) + torch.bmm(yq.unsqueeze(1), T).squeeze(1)
        d_R[s0:s1] = torch.linalg.norm(qq - hat_R, dim=-1)
        d_inf[s0:s1] = torch.linalg.norm(qq - hat_i, dim=-1)
    return d_inf, d_R


def check(label, q, bank, kw, ctx, do_closed_form=True):
    fails = 0
    a = cu.lonm_score(q, bank, ctx, **kw)
    b = cu_ref.lonm_score(q, bank, ctx, **kw)
    same = bool(torch.equal(a, b))
    print(f'[{label}]')
    print(f'   unbounded=0 bit-identical to pre-patch : {same}'
          f'  (max|d| = {float((a - b).abs().max()):.3e})')
    fails += (not same)

    # clip_stat must be numerically INERT: the radius arms are run with
    # --lonm_clipstat 1 while the anchors are not, so unless this holds the
    # radius ablation would not be a single-knob comparison.  The counters must
    # nevertheless be populated.
    stat0 = {}
    d = cu.lonm_score(q, bank, ctx, clip_stat=stat0, **kw)
    inert = bool(torch.equal(a, d))
    counted = int(stat0.get('n', 0)) > 0
    print(f'   clip_stat=dict() bit-identical         : {inert}'
          f'  (counters filled: {counted}, n = {stat0.get("n", 0)})')
    if not inert or not counted:
        fails += 1

    stat = {}
    c = cu.lonm_score(q, bank, ctx, unbounded=True, clip_stat=stat, **kw)
    frac = stat['clip'] / stat['n']
    diff = not bool(torch.equal(a, c))
    print(f'   clipping active on                     : {frac:.4%} of coords '
          f'(max ||a||/R = {stat["ratio_max"]:.3f})')
    print(f'   R=inf arm differs from bounded arm     : {diff}'
          f'  (max|d| = {float((a - c).abs().max()):.3e})')
    if frac > 0 and not diff:
        print('   ERROR: clipping active but the unbounded arm did not change')
        fails += 1
    if do_closed_form:
        d_inf, d_R = closed_form(q, bank, ctx, kw)
        e_i = float((c - d_inf).abs().max())
        e_R = float((a - d_R).abs().max())
        print(f'   ||s_inf - closed form||_inf            : {e_i:.3e}')
        print(f'   ||s_R   - closed form||_inf            : {e_R:.3e}')
        if e_i > 1e-5 or e_R > 1e-5:
            fails += 1
    print()
    return fails


def main():
    ctx = cu.build_cand_context(28, 28, RHO, 'cuda')
    fails = 0

    q, bank = random_case(0)
    fails += check('A  random features, production config', q, bank,
                   prod_cfg(), ctx)
    q, bank = random_case(0)
    fails += check('B  random features, beta=0.5', q, bank,
                   prod_cfg(trust_q=0.5), ctx)
    q, bank = structured_case(1)
    fails += check('C  STRUCTURED features (clipping fires)', q, bank,
                   prod_cfg(), ctx)
    q, bank = structured_case(2, spread=0.004, excursion=2.0)
    fails += check('D  STRUCTURED, tighter bank, beta=0.5', q, bank,
                   prod_cfg(trust_q=0.5), ctx)
    # cosine readout: only the bit-identity / branch parts apply
    q, bank = structured_case(1)
    fails += check('E  structured, cosine readout (no closed form)', q, bank,
                   prod_cfg(norm_scale=False, norm_mode='ratio'), ctx,
                   do_closed_form=False)

    print('RESULT:', 'ALL OK' if fails == 0 else f'{fails} CHECK(S) FAILED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
