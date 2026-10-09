"""Definition-level checks for the four matched baselines.

Each check verifies the property that DEFINES the method, not merely that the
function runs.  Run:  PYTHONPATH=. python exp/test_baselines.py
"""
import sys

import torch
import torch.nn.functional as F

import baseline_scores as bs

C, L, RHO = 768, 784, 1
fails = 0


def ok(name, cond, extra=''):
    global fails
    print(f'  [{"OK " if cond else "FAIL"}] {name}   {extra}')
    if not cond:
        fails += 1


def refs(N, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    return F.normalize(torch.randn(N, L, C, generator=g, device='cuda'), dim=-1)


def lowrank_refs(N, l, c, rank=4, spread=0.5, noise=1e-3, seed=7):
    """Normals concentrated on a shared low-rank subspace, as l2-normalised
    deep descriptors are; this is what makes the per-position covariance
    nearly rank deficient."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    b = F.normalize(torch.randn(c, generator=g, device='cuda'), dim=-1)
    U = torch.linalg.qr(torch.randn(c, rank, generator=g, device='cuda'))[0]
    coef = spread * torch.randn(N, l, rank, generator=g, device='cuda')
    x = (b.view(1, 1, c) + coef @ U.t()
         + noise * torch.randn(N, l, c, generator=g, device='cuda'))
    return F.normalize(x, dim=-1)


# ---------------------------------------------------------------- SubspaceAD
def test_subspacead():
    print('SubspaceAD (category-level PCA residual)')
    r = refs(32, 1)
    st = bs.fit_subspacead([r], rows=None, bank_cache=None, d=16)
    mu, V = st[0]
    ok('output shapes', mu.shape == (C,) and V.shape == (16, C),
       f'mu{tuple(mu.shape)} V{tuple(V.shape)}')
    ok('basis is orthonormal', float((V @ V.t()
       - torch.eye(16, device='cuda')).abs().max()) < 1e-4)

    # defining property: the score is the squared norm of the component of
    # (q - mu) outside the span, i.e. the Pythagorean remainder
    q = F.normalize(torch.randn(L, C, device='cuda'), dim=-1)
    t = q - mu
    proj = (t @ V.t()) @ V
    expect = t.pow(2).sum(-1) - proj.pow(2).sum(-1)
    got = bs.subspacead_score(q, mu, V, power=2.0)
    ok('score == squared out-of-span component',
       float((got - expect).abs().max()) < 1e-4,
       f'max={float((got - expect).abs().max()):.3e}')
    # moving the query inside the span must not change the score
    y = torch.randn(L, 16, device='cuda') * 0.5
    q_shift = q + y @ V
    ok('shift along the span leaves the score unchanged',
       float((bs.subspacead_score(q_shift, mu, V, 2.0) - got).abs().max()) < 1e-4)
    # and an orthogonal displacement must raise it
    o = torch.randn(L, C, device='cuda')
    o = o - (o @ V.t()) @ V
    ok('orthogonal displacement raises the score',
       float(bs.subspacead_score(q + 0.3 * o, mu, V, 2.0).min()) > float(got.max()))
    ok('power=1 == sqrt(power=2)',
       float((bs.subspacead_score(q, mu, V, 1.0) - got.sqrt()).abs().max()) < 1e-6)


# -------------------------------------------------------------------- PaDiM
def test_padim():
    print('PaDiM (position-specific Gaussian, Mahalanobis)')
    N = 64
    r = refs(N, 2)
    st = bs.fit_padim(r, d=32, shrink='ridge', ridge_rel=1e-12, ridge_abs=0.0)
    ok('output shapes', st['sel'].shape == (L, 32)
       and st['mu'].shape == (L, 32)
       and st['prec'].shape == (L, 32, 32), str(tuple(st['prec'].shape)))
    ok('precision is symmetric', float((st['prec'] - st['prec'].transpose(1, 2))
       .abs().max()) < 1e-3)

    # with an identity covariance the Mahalanobis distance is the plain
    # Euclidean distance in the selected dimensions
    st_i = dict(sel=st['sel'], mu=st['mu'],
                prec=torch.eye(32, device='cuda').expand(L, 32, 32).contiguous())
    q = F.normalize(torch.randn(L, C, device='cuda'), dim=-1)
    m2 = bs.padim_score(q, st_i, sqrt=False)
    m1 = bs.padim_score(q, st_i, sqrt=True)
    euc2 = (torch.gather(q, 1, st['sel']) - st['mu']).pow(2).sum(-1)
    ok('identity-cov, squared form == squared Euclidean',
       float((m2 - euc2).abs().max()) < 1e-5,
       f'max={float((m2 - euc2).abs().max()):.3e}')
    ok('sqrt readout == sqrt of the squared form',
       float((m1 - m2.sqrt()).abs().max()) < 1e-6)

    # the fitted precision must invert the EMPIRICAL covariance of the selected
    # dimensions -- that is PaDiM's definition of the reference
    x = r.detach().permute(1, 0, 2).float()                          # [L, N, C]
    xs = torch.gather(x, 2, st['sel'].unsqueeze(1).expand(-1, N, -1))
    mu_ref = xs.mean(dim=1)
    xc = xs - mu_ref.unsqueeze(1)
    cov_ref = torch.bmm(xc.transpose(1, 2), xc) / (N - 1)
    ok('mean equals the empirical mean',
       float((mu_ref - st['mu']).abs().max()) < 1e-5)
    cov_fit = torch.linalg.inv(st['prec'])
    rel = float((cov_fit - cov_ref).abs().max()
                / cov_ref.diagonal(dim1=-2, dim2=-1).mean())
    ok('unregularised precision inverts the empirical covariance', rel < 1e-4,
       f'rel={rel:.2e}')
    # and the covariance really is non-diagonal, i.e. this is not a whitened
    # diagonal model in disguise
    off = cov_ref - torch.diag_embed(cov_ref.diagonal(dim1=-2, dim2=-1))
    ok('covariance is non-diagonal (Mahalanobis, not whitened)',
       float(off.abs().max()) > 0.0
       and float(off.abs().max() / cov_ref.diagonal(dim1=-2, dim2=-1).mean()) > 1e-3)
    ok('scores are finite and non-negative',
       bool(torch.isfinite(m2).all()) and float(m2.min()) >= 0.0)

    # ---- CALIBRATION -------------------------------------------------------
    # A fitted Gaussian must place HELD-OUT normal patches at a mean quadratic
    # form of order d.  Using held-out rather than training patches matters: a
    # training point's own quadratic form is biased upward by roughly n/d even
    # for a perfect covariance, so it cannot be used as the invariant.
    #
    # Random unit vectors in C = 768 are nearly orthogonal, so their
    # per-position covariance stays well conditioned even at N/d ~ 2 and an
    # under-regularised inverse happens to be harmless.  Real l2-normalised
    # DINOv3 descriptors are NOT like that: they concentrate on a low-rank
    # subspace, the sample covariance is then nearly rank deficient, and the
    # inverse inflates the distance by about 1/eigenvalue_min.  That is the
    # failure measured on bottle (held-out mean Mahalanobis^2 = 5467 at d = 64,
    # i.e. 85x), which is why the default regulariser is Ledoit-Wolf rather
    # than the paper's absolute 0.01*I or a negligible relative ridge.
    for lab, big in (('random bank  ', refs(128, 2)),
                     ('low-rank bank', lowrank_refs(128, L, C, rank=4,
                                                    spread=0.5, noise=1e-3))):
        rs, held = big[:64], big[64:]
        s_n = bs.fit_padim(rs, d=32, shrink='ridge', ridge_rel=1e-12,
                           ridge_abs=0.0)
        s_l = bs.fit_padim(rs, d=32, shrink='lw')
        nv = float(bs.padim_score(held[0], s_n, sqrt=False).mean())
        lw = float(bs.padim_score(held[0], s_l, sqrt=False).mean())
        print(f'  {lab}: held-out mean M^2  unreg {nv:11.1f} '
              f'(cond {s_n["cond_med"]:.2e}) | LW {lw:7.1f} '
              f'(delta {s_l["shrink_delta"]:.3f}, cond '
              f'{s_l["cond_med"]:.2e}) | d = 32')
        if lab.startswith('low-rank'):
            ok('low-rank bank: unregularised inverse IS over-confident',
               nv / 32.0 > 3.0, f'{nv / 32.0:.1f}x')
            ok('LW reduces the low-rank inflation by >= 10x',
               lw < nv / 10.0, f'{nv:.1f} -> {lw:.1f}')
            ok('Ledoit-Wolf shrinkage coefficient is a probability',
               0.0 <= s_l['shrink_delta'] <= 1.0)
        else:
            ok('random bank: LW is calibrated (|mean M^2 - d| < d)',
               abs(lw - 32.0) / 32.0 < 1.0, f'{lw:.1f} vs 32')


# ---------------------------------------------------------------- PatchCore
def test_patchcore():
    print('PatchCore (coreset memory, k-NN + reweighting)')
    r = refs(24, 3)
    feat = r.reshape(-1, C)
    mem = bs.patchcore_coreset(feat, 500, seed=0)
    ok('coreset size', mem.shape == (500, C), str(tuple(mem.shape)))
    # every coreset entry must come from the memory
    d2 = torch.cdist(mem.float(), feat.float()).pow(2)
    worst = float(d2.min(dim=1).values.max())
    ok('coreset entries are memory patches', worst < 1e-5,
       f'worst squared distance to the memory = {worst:.3e}')

    # the accelerated path (random projection + a capped candidate set) must
    # still return real memory patches, must honour the cap, and must be
    # deterministic given the seed
    mem_a = bs.patchcore_coreset(feat, 400, seed=0, proj_dim=32, sub_cap=5000)
    d2a = torch.cdist(mem_a.float(), feat.float()).pow(2)
    worst_a = float(d2a.min(dim=1).values.max())
    ok('accelerated path returns memory patches', worst_a < 1e-5
       and mem_a.shape == (400, C),
       f'worst={worst_a:.3e} shape={tuple(mem_a.shape)}')
    mem_b = bs.patchcore_coreset(feat, 400, seed=0, proj_dim=32, sub_cap=5000)
    ok('accelerated path is deterministic', bool((mem_a == mem_b).all()))
    small = bs.patchcore_coreset(feat, 5000, seed=0, proj_dim=32, sub_cap=5000)
    ok('requested size above the candidate cap is clamped',
       small.shape[0] <= 5000, str(tuple(small.shape)))

    q = F.normalize(torch.randn(L, C, device='cuda'), dim=-1)
    s = bs.patchcore_score(q, mem, k=9, rw=1.0)
    ok('score shape/finite', s.shape == (L,) and bool(torch.isfinite(s).all()))
    # a query equal to a memory entry must score ~0
    s0 = bs.patchcore_score(mem[:4].clone(), mem, k=9, rw=1.0)
    ok('exact memory hit scores ~0', float(s0.max()) < 1e-6,
       f'max={float(s0.max()):.3e}')
    # with k=1 the reweighting factor is a constant, so the score is monotone
    # in the nearest distance
    s1 = bs.patchcore_score(q, mem, k=1, rw=1.0)
    d1 = torch.cdist(q.float(), mem.float()).min(dim=1).values
    const = 1.0 - torch.exp(torch.tensor(-1.0, device='cuda'))
    ok('k=1 == d1 * (1 - e^-1)', float((s1 - d1 * const).abs().max()) < 1e-4,
       f'max={float((s1 - d1 * const).abs().max()):.3e}')


# -------------------------------------------------------------------- ANoCo
def test_anoco():
    print('ANoCo (anchored bipartite Laplacian drift)')
    import curvot_utils as cu
    r = refs(12, 4)
    ctx = cu.build_cand_context(28, 28, RHO, 'cuda')
    q = F.normalize(torch.randn(L, C, device='cuda'), dim=-1)
    e = bs.anoco_score(q, r, ctx, lambda_rel=1.0, chunk=32)
    ok('score shape/finite', e.shape == (L,) and bool(torch.isfinite(e).all())
       and float(e.min()) >= 0.0)

    # ---- independent check: solve the DENSE anchored Laplacian energy ------
    # (Lqq + Lambda) F~q = Lambda Fq - Lqr Fr  with Lqq = diag(D), Lqr = -W.
    # The point is that this uses a dense linear solve instead of the
    # elementwise division the closed form relies on.
    s0, s1, cand, okm = next(iter(cu.collect_candidates(r, ctx, chunk=32)))
    qq = q[s0:s1]
    sim = torch.bmm(cand, qq.unsqueeze(-1)).squeeze(-1)
    sim = sim.masked_fill(~okm, float('-inf'))
    js = sim.argmax(dim=-1, keepdim=True)
    zs = torch.gather(cand, 1, js.unsqueeze(-1).expand(-1, -1, C)).squeeze(1)
    a = torch.bmm(cand, zs.unsqueeze(-1)).squeeze(-1)
    keep = (a > sim.gather(1, js)) & torch.isfinite(sim)
    w = torch.where(keep, sim.clamp_min(0.0), torch.zeros_like(sim))
    D = w.sum(-1, keepdim=True)
    deg = D.squeeze(1) < 1e-8
    if bool(deg.any()):
        wu = keep.to(sim.dtype)
        w = torch.where(deg.view(-1, 1), wu, w)
        D = torch.where(deg.view(-1, 1), wu.sum(-1, keepdim=True), D).clamp_min(1e-8)
    D = D.clamp_min(1e-8)
    lam = 1.0 * D
    A = torch.diag_embed((D + lam).squeeze(-1))                    # [Pc, Pc]
    rhs = lam * qq + torch.bmm(w.unsqueeze(1), cand).squeeze(1)    # [Pc, C]
    ft = torch.linalg.solve(A, rhs)
    drift = (ft - qq).pow(2).sum(-1)
    cos = (ft * qq).sum(-1) / ft.norm(dim=-1).clamp_min(1e-8)
    E = (drift * (1.0 - cos).clamp_min(0.0)).to(q.dtype)
    df = float((E - e[s0:s1]).abs().max())
    ok('closed form == dense anchored-Laplacian solve', df < 5e-3,
       f'max|d|={df:.3e}')

    # a query that IS normal (a copy of a reference patch) must drift less
    q_ref = r[0, :L].clone()
    e_ref = bs.anoco_score(q_ref, r, ctx, lambda_rel=1.0, chunk=32)
    ok('normal query drifts less than a random query',
       float(e_ref.mean()) < float(e.mean()),
       f'mean {float(e_ref.mean()):.4g} < {float(e.mean()):.4g}')

    # the anchor rule must keep at least the anchor itself
    keep_any = True
    for s0, s1, cand, okm in cu.collect_candidates(r, ctx, chunk=256):
        sim = torch.bmm(cand, q[s0:s1].unsqueeze(-1)).squeeze(-1)
        sim = sim.masked_fill(~okm, float('-inf'))
        js = sim.argmax(dim=-1, keepdim=True)
        zs = torch.gather(cand, 1, js.unsqueeze(-1).expand(-1, -1, C)).squeeze(1)
        a = torch.bmm(cand, zs.unsqueeze(-1)).squeeze(-1)
        keep = (a > sim.gather(1, js)) & torch.isfinite(sim)
        keep_any &= bool(keep.sum(-1).min() >= 1)
    ok('anchor rule keeps >= 1 neighbour everywhere', keep_any)


if __name__ == '__main__':
    assert torch.cuda.is_available()
    test_subspacead()
    test_padim()
    test_patchcore()
    test_anoco()
    print('\nRESULT:', 'ALL OK' if fails == 0 else f'{fails} CHECK(S) FAILED')
    sys.exit(1 if fails else 0)
