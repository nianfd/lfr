"""Matched-feature baselines for the LFR paper.

The reviewer's first request is a controlled comparison that is NOT "LFR vs
RAD only": re-run the four reference constructions the paper discusses at the
level of its own related-work table, with the frozen representation, the
normal memory, the output processing and the metric evaluator held fixed.

What is shared with LFR and the RAD rerun
-----------------------------------------
* frozen DINOv3 ViT-B/16, the same four layers, l2-normalised descriptors;
* the same normal memory (the full, uncompressed bank) and the same
  per-category bank blocks;
* the same anomaly-map recipe -- layer mean, bilinear upsample to 256x256,
  5x5 Gaussian with sigma 1.0 -- and the same top-1% image pooling;
* the same evaluator (P-AUROC / P-AP / P-F1-max / AUPRO, I-AUROC / I-AP / I-F1).

What differs is only the reference construction and the readout, which is
precisely the object of the comparison.

Deviation from each paper's original protocol
---------------------------------------------
Every one of these methods was published with a different backbone, and RAD
itself is re-run here rather than quoted, so the point of the table is a
matched comparison under one representation, not a reproduction of each
paper's headline number.  Two further points are stated explicitly because
they could otherwise be mistaken for a straw man:

* PatchCore's own memory is a coreset of the normal patches, and its original
  pooling is a maximum.  We keep its coreset construction (including the
  random-projection acceleration the authors use), its coreset ratio and its
  k-NN reweighting, but pool with the pipeline's top-1% mean so that the
  image-level column is comparable across rows.  Its localization column is
  unaffected by pooling.  The coreset size actually used per category and layer
  is printed in the run log and quoted in the paper's table note.
* ANoCo is published for K<=4 normal images and retrieves over all reference
  patches.  Under this protocol the reference set is the same candidate pool
  RAD/LFR use (retrieved images x spatial neighbourhood), because a
  full-bank bipartite graph is not the memory any other row gets.  Its
  closed form, anchor rule and drift score are unchanged.
* PaDiM's published regulariser is an ABSOLUTE ``0.01 * I`` calibrated to the
  un-normalised WideResNet50 activations of its own protocol, and no ridge at
  all leaves the sample covariance nearly singular here (N/d ~ 3, effective
  rank far below d for l2-normalised descriptors).  The three regularisation
  regimes score very differently on real data, so the row is NOT picked by
  convenience: ``--baseline padim`` runs the paper verbatim (default), and the
  scale-free Ledoit--Wolf estimator and the scale-equivalent relative ridge are
  both run as sensitivity checks and reported with the table.  The published
  readout (the square root of the quadratic form) is used throughout.

Entry points
------------
``fit_*`` build the offline reference for the methods that estimate it once
per category; ``*_score`` returns a per-patch anomaly value for one layer and
mirrors ``curvot_utils.lonm_score``'s contract (``[L]``, larger = more
anomalous).  Pool-based methods take ``(q_norm, bank, ctx)`` exactly as the
LONM readout does.
"""

import torch
import torch.nn.functional as F

NAME = 'matched baselines'


# ---------------------------------------------------------------------------
# 1) SubspaceAD -- category-level PCA, scored by the reconstruction residual
# ---------------------------------------------------------------------------
@torch.no_grad()
def fit_subspacead(patch_layers, rows, bank_cache, d=64, chunk_mb=256):
    """One PCA model per layer, fitted to the category's normal patches.

    SubspaceAD (Lendering et al., CVPR 2026) fits a PCA model to patch-level
    features from the normal images and scores a test patch by its
    reconstruction residual with respect to the estimated subspace.  We fit on
    the same category block of the same bank the other rows retrieve from, so
    the normal memory is matched; ``d`` components are kept.

    Returns ``[(mu [C], V [d, C]), ...]``, one pair per layer.
    """
    stats = []
    for li, patches in enumerate(patch_layers):
        x = patches.reshape(-1, patches.shape[-1])                # [N*L, C]
        n, c = x.shape
        mu = x.mean(dim=0)
        cov = torch.zeros(c, c, device=x.device, dtype=torch.float32)
        step = max(1, int(chunk_mb * 1024 * 1024 / 4 // max(c, 1)))
        for s0 in range(0, n, step):
            xc = (x[s0:s0 + step] - mu).float()
            cov += xc.transpose(0, 1) @ xc
        cov /= max(1, n - 1)
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.flip(-1).clamp_min(0.0)
        V = evecs.flip(-1)[:, :d].transpose(0, 1).contiguous()     # [d, C]
        stats.append((mu.to(x.dtype), V.to(x.dtype)))
    return stats


@torch.no_grad()
def subspacead_score(q_norm, mu, V, power=2.0):
    """Residual of the query patch to the fitted principal subspace.

    ``power=2`` is the squared orthogonal projection residual used by
    SubspaceAD; ``power=1`` is the unsquared form, which is the readout the
    paper's own LFR-vs-subspace limit uses, and is reported as a sensitivity
    check.
    """
    t = q_norm - mu
    t = t - (t @ V.t()) @ V
    s2 = t.pow(2).sum(dim=-1)
    return s2 if power == 2 else s2.sqrt()


# ---------------------------------------------------------------------------
# 2) PaDiM -- position-specific Gaussian, Mahalanobis readout
# ---------------------------------------------------------------------------
@torch.no_grad()
def fit_padim(patch_layer, d=64, ridge_rel=0.0, ridge_abs=0.01, shrink='published'):
    """Position-specific mean and covariance over the category's normals.

    PaDiM (Defard et al., ICPR 2021) estimates a mean and covariance at every
    patch position and scores a test patch by its Mahalanobis distance.  The
    covariance is the mechanism: displacement is scaled by the position's own
    second moment, so movement along a high-variance direction is discounted.

    Features arrive as ``[N, L, C]`` for one layer.  The per-position feature
    dimension is first reduced to ``d`` by the largest marginal variance --
    deterministic, where PaDiM's own random selection is a strictly weaker
    choice on ViT features -- and then the FULL d x d covariance inverse is
    used, so the off-diagonal structure PaDiM relies on is preserved.

    Regularisation is a genuine fork in this method and is therefore not
    silently resolved.  The published code adds an ABSOLUTE ``0.01 * I`` to
    every position's covariance; that constant is calibrated to un-normalised
    WideResNet50 activations, whereas every row of this table consumes
    l2-normalised descriptors whose per-dimension variance is ~1/C.  The three
    regimes are materially different on real data (bottle, MVTec-AD, d = 64):

        published   0.01 * I, the paper's own constant   over-regularised at
                                                        this feature scale, so
                                                        the readout approaches
                                                        Euclidean in the
                                                        selected dimensions
        lw          Ledoit-Wolf towards mu * I           scale free, automatic,
                                                        but delta ~ 0.05 here,
                                                        so the inverse stays
                                                        heavy tailed
        ridge_rel   ridge scaled to the mean diagonal    the scale-equivalent
        (e.g. 1e-3) reading of the paper's constant      inverse is nearly
                                                        singular and inflates
                                                        the distance by ~2
                                                        orders of magnitude

    ``shrink='published'`` is the default because it applies the method exactly
    as published; the other two are run as a sensitivity check so that the row
    cannot be read as a choice of convenient regulariser.
    """
    N, L, C = patch_layer.shape
    dev, dt = patch_layer.device, torch.float32
    x = patch_layer.detach().permute(1, 0, 2).to(dt)               # [L, N, C]
    d = int(min(d, max(8, N - 2), C))
    sel = torch.topk(x.var(dim=1), d, dim=-1).indices              # [L, d]
    xs = torch.gather(x, 2, sel.unsqueeze(1).expand(-1, N, -1))    # [L, N, d]
    mu = xs.mean(dim=1)                                            # [L, d]
    xc = (xs - mu.unsqueeze(1)).double()
    n = xc.shape[1]
    eye = torch.eye(d, device=dev, dtype=torch.float64).unsqueeze(0)

    if shrink == 'lw':
        # Ledoit-Wolf (2004) shrinkage towards mu * I, identity target.
        S = torch.bmm(xc.transpose(1, 2), xc) / n                  # biased S
        mu_s = S.diagonal(dim1=-2, dim2=-1).mean(-1)               # [L]
        x2 = xc.pow(2)
        S2 = torch.bmm(x2.transpose(1, 2), x2) / n
        pi_hat = (S2 - S.pow(2)).flatten(1).mean(-1)               # [L]
        gam_hat = ((S - mu_s.view(-1, 1, 1) * eye).pow(2)
                   ).flatten(1).mean(-1)                           # [L]
        delta = (pi_hat / (n * gam_hat).clamp_min(1e-300)
                 ).clamp(0.0, 1.0)                                 # [L]
        cov = ((1 - delta).view(-1, 1, 1) * S
               + (delta * mu_s).view(-1, 1, 1) * eye)
        diag = {'shrink_delta': float(delta.mean())}
    elif shrink in ('published', 'ridge'):
        cov = torch.bmm(xc.transpose(1, 2), xc) / max(1, n - 1)
        tr = cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
        cov = cov + (ridge_rel * tr + ridge_abs).view(-1, 1, 1) * eye
        diag = {'shrink_delta': 0.0}
    else:
        raise ValueError(f'unknown padim shrink {shrink!r}')

    ev = torch.linalg.eigvalsh(cov).clamp_min(1e-30)
    diag['cond_mean'] = float((ev[:, -1] / ev[:, 0]).mean())
    diag['cond_med'] = float((ev[:, -1] / ev[:, 0]).median())
    prec = torch.linalg.inv(cov).to(dt)
    return dict(sel=sel, mu=mu, prec=prec, **diag)


@torch.no_grad()
def padim_score(q_norm, st, sqrt=True):
    """Mahalanobis distance in the fitted position-specific Gaussian.

    ``sqrt=True`` reproduces the published readout (``scipy.mahalanobis``
    returns the square root of the quadratic form).  The distinction matters
    only for the image-level tail aggregation, which is a mean over the top 1%
    of map values and is therefore not invariant under the square root; every
    ranking metric is unaffected.
    """
    qs = torch.gather(q_norm, 1, st['sel'])                        # [L, d]
    t = (qs - st['mu']).to(torch.float32)
    m2 = torch.einsum('ld,ldd,ld->l', t, st['prec'], t)
    m2 = m2.clamp_min(0.0)
    return m2.sqrt() if sqrt else m2


# ---------------------------------------------------------------------------
# 3) PatchCore -- coreset memory, k-NN with local neighbourhood reweighting
# ---------------------------------------------------------------------------
@torch.no_grad()
def patchcore_coreset(feat, m, seed=0, proj_dim=64, sub_cap=120000,
                      verbose=False):
    """Greedy k-center coreset over the normal patch features.

    PatchCore (Roth et al., CVPR 2022) keeps a representative subset of the
    normal patches; a test patch is scored by the distance to its nearest
    stored patch, reweighted by the distance to its b-th nearest stored patch
    so that a patch sitting in an isolated corner of the memory is not trusted.

    Greedy k-center is ``O(m * n)`` distance evaluations.  At this protocol's
    memory scale that is prohibitive: a VisA category holds about 0.56M patches
    and a 1% coreset is about 5.6k points, i.e. about 1.6 TB of reads per layer
    and roughly 20 h per category.  Two accelerations are applied, both of them
    standard for this construction:

    * ``proj_dim`` -- a Gaussian random projection before the selection loop,
      which is the acceleration the PatchCore authors themselves apply when the
      patch dimension is large.  The selected indices still refer to the
      original descriptors, so the stored memory is unchanged; only the
      ordering used by the greedy step is approximated, and a Johnson--
      Lindenstrauss embedding preserves it with high probability.
    * ``sub_cap`` -- the selection candidates are capped by a uniform
      subsample of the category's patches.  Centers are then chosen among the
      subsample but the retained points are still real bank patches.  With
      ``sub_cap`` far above ``m`` the cover is a good approximation of the
      full-set coreset; the value used is recorded in the run log.

    Both are reported in the paper's table note so that this row cannot be read
    as a straw man.
    """
    n = feat.shape[0]
    m = int(max(1, min(m, n)))
    if m >= n:
        return feat.clone()

    g = torch.Generator(device=feat.device).manual_seed(seed + 11)
    cand_idx = None
    if sub_cap and n > sub_cap:
        cand_idx = torch.randperm(n, generator=g, device=feat.device)[:sub_cap]
        cand_idx, _ = torch.sort(cand_idx)

    work = feat if cand_idx is None else feat[cand_idx]
    nw = work.shape[0]
    if proj_dim and 0 < proj_dim < work.shape[1]:
        G = torch.randn(work.shape[1], proj_dim, generator=g,
                        device=feat.device, dtype=work.dtype)
        G = G / G.norm(dim=0, keepdim=True).clamp_min(1e-12)
        work = work @ G

    g2 = torch.Generator(device=feat.device).manual_seed(seed)
    first = int(torch.randint(nw, (1,), generator=g2, device=feat.device))
    mind = (work - work[first]).pow(2).sum(-1)                     # [nw]
    keep = torch.empty(m, dtype=torch.long, device=feat.device)
    keep[0] = first
    mind[first] = -1.0
    for i in range(1, m):
        j = int(torch.argmax(mind))
        keep[i] = j
        d = (work - work[j]).pow(2).sum(-1)
        mind = torch.minimum(mind, d)
        mind[j] = -1.0
        if verbose and (i % max(1, m // 10) == 0):
            print(f'    [coreset] {i}/{m}', flush=True)
    if cand_idx is not None:
        keep = cand_idx[keep]
    return feat[keep]


@torch.no_grad()
def patchcore_score(q_norm, mem, k=9, rw=1.0, chunk=128, eps=1e-12):
    """Nearest-memory-patch distance with PatchCore's reweighting.

        s = ( 1 - exp( -d1^2 / (w * db^2) ) ) * d1

    with ``d1`` the distance to the nearest and ``db`` the distance to the
    ``k``-th nearest stored patch.  Suffices are squared distances; ``d1`` is
    returned unsquared so the map has the same units as the other rows' maps.
    """
    L = q_norm.shape[0]
    out = torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
    mm = mem.shape[0]
    kk = int(min(k, mm))
    for s0 in range(0, L, chunk):
        q = q_norm[s0:s0 + chunk].float()
        d2 = torch.cdist(q, mem.float()).pow(2)                    # [Pc, M]
        top = torch.topk(d2, kk, dim=-1, largest=False).values
        d1 = top[:, 0].clamp_min(eps)
        db = top[:, -1].clamp_min(eps)
        out[s0:s0 + chunk] = ((1.0 - torch.exp(-d1 / (rw * db))) * d1.sqrt()).to(
            q_norm.dtype)
    return out


# ---------------------------------------------------------------------------
# 4) ANoCo -- anchored bipartite graph Laplacian drift
# ---------------------------------------------------------------------------
@torch.no_grad()
def anoco_score(q_norm, bank, ctx, lambda_rel=1.0, chunk=64, eps=1e-8,
                anchor=True):
    """Closed-form anchored Laplacian drift over the matched candidate pool.

    ANoCo (Seo et al., CVPR 2026) scores a patch by how much it must move to
    conform to the normal manifold.  With ``s_ij = cos(q_i, z_j)``:

      * anchor      ``j*(i) = argmax_j s_ij``;
      * neighbour   keep ``j`` iff ``cos(z_{j*}, z_j) > s_i,j*``
                    (anchor consistency; ``anchor=False`` keeps the pool);
      * weights     ``w_ij = s_ij * alpha_ij`` with
                    ``alpha_ij = 2|q||z| / (|q|^2 + |z|^2)``, which equals one
                    here because every descriptor is l2-normalised;
      * closed form ``f~_i = (D_ii + lam_i)^{-1} (lam_i f_i + sum_j w_ij z_j)``
                    with ``D_ii = sum_j w_ij`` the bipartite degree, the
                    reference nodes anchored, and query-query edges absent so
                    the query block is diagonal;
      * score       ``E_i = ||f~_i - f_i||^2 * (1 - cos(f~_i, f_i))``, i.e.
                    both the magnitude and the directional part of the drift.

    ``lam_i = lambda_rel * D_ii`` keeps the regularisation scale free.  The
    published note does not state lambda's value, so ``lambda_rel = 1`` is
    fixed a priori rather than tuned on the test set; a sensitivity check is
    run instead.
    """
    L, C = q_norm.shape
    out = torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
    for s0, s1, cand, ok in cu_collect(bank, ctx, chunk=chunk):
        q = q_norm[s0:s1]
        sim = torch.bmm(cand, q.unsqueeze(-1)).squeeze(-1)          # [Pc, Kc]
        if ok is not None:
            sim = sim.masked_fill(~ok, float('-inf'))
        live = torch.isfinite(sim)
        if anchor:
            jstar = sim.argmax(dim=-1, keepdim=True)               # [Pc, 1]
            zstar = torch.gather(
                cand, 1, jstar.unsqueeze(-1).expand(-1, -1, C)).squeeze(1)
            a = torch.bmm(cand, zstar.unsqueeze(-1)).squeeze(-1)   # [Pc, Kc]
            keep = (a > sim.gather(1, jstar)) & live
        else:
            keep = live
        w = torch.where(keep, sim.clamp_min(0.0), torch.zeros_like(sim))
        D = w.sum(dim=-1, keepdim=True)                            # [Pc, 1]
        # Degenerate case: every retained similarity is <= 0 (possible on
        # all-negative windows).  Fall back to a uniform weight over the kept
        # set -- the anchor itself is always kept, so D >= 1 afterwards.
        deg = D.squeeze(1) < eps
        if bool(deg.any()):
            wu = keep.to(sim.dtype)
            w = torch.where(deg.view(-1, 1), wu, w)
            D = torch.where(deg.view(-1, 1), wu.sum(dim=-1, keepdim=True),
                            D).clamp_min(eps)
        D = D.clamp_min(eps)
        lam = lambda_rel * D                                       # [Pc, 1]
        agg = torch.bmm(w.unsqueeze(1), cand).squeeze(1)           # [Pc, C]
        ft = (lam * q + agg) / (D + lam)                           # [Pc, C]
        drift = (ft - q).pow(2).sum(dim=-1)                        # [Pc]
        nq = ft.norm(dim=-1).clamp_min(eps)
        cos = (ft * q).sum(dim=-1) / nq                            # |q| == 1
        out[s0:s1] = (drift * (1.0 - cos).clamp_min(0.0)).to(q_norm.dtype)
    return out


def cu_collect(bank, ctx, chunk=64):
    """Local import shim so this module stays importable next to curvot_utils."""
    import curvot_utils as cu
    return cu.collect_candidates(bank, ctx, chunk=chunk)
