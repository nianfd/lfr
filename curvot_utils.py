"""
CurvOT-RAD — core utilities.

Two mechanisms, both strictly training-free (no SGD, no adapter, no prompt):

1) APTR  (Anomaly-Resilient Partial-Transport Retrieval)
   Coarse CLS retrieval is kept as a cheap shortlist, then references are
   re-ranked by an *entropic partial* optimal-transport structural cost in
   which anomalous query mass is explicitly allowed to stay unmatched.

2) LONM  (Local Osculating Normal Manifold)
   Instead of a single 1-NN normal patch, the normality of a query patch is
   measured as its distance to a locally fitted second-order (osculating)
   normal manifold recovered in closed form from a few nearby normal patches.

The partial-OT solver is a batched, log-domain Sinkhorn on an *exact*
balanced-OT reduction (two dustbin nodes), so it needs no external library
and runs on GPU for a whole shortlist at once.
"""

import math

import torch
import torch.nn.functional as F


# ============================================================================
# 0) sanity helpers
# ============================================================================
def _as_3d(x):
    return x if x.dim() == 3 else x.unsqueeze(0)


# ============================================================================
# 1) coarse token pooling  (28x28 -> 7x7)
# ============================================================================
def grid_pool_tokens(x, gh, gw, out_grid):
    """x: [B, gh*gw, C] -> [B, out_grid*out_grid, C] by average pooling."""
    if out_grid == gh and out_grid == gw:
        return x
    assert gh % out_grid == 0 and gw % out_grid == 0, (gh, gw, out_grid)
    B, L, C = x.shape
    assert L == gh * gw
    k = gh // out_grid
    x = x.transpose(1, 2).reshape(B, C, gh, gw)
    x = F.avg_pool2d(x, kernel_size=k, stride=k)
    return x.reshape(B, C, out_grid * out_grid).transpose(1, 2).contiguous()


# ============================================================================
# 2) entropic PARTIAL optimal transport  (batched, log-domain)
# ============================================================================
@torch.no_grad()
def partial_ot_plan(C, reg=0.05, mass=0.8, n_iter=120, forbid=1e3,
                    eps_scaling=True, return_plan=True):
    """Batched entropic partial OT.

    Problem (per batch element, a and b uniform distributions):
        min_{P>=0}  <P, C> + reg * sum_ij P_ij (log P_ij - 1)
        s.t.        P 1  <= a,   P^T 1 <= b,   1^T P 1 = mass

    Exact reduction to a balanced problem: append one *dump* sink that absorbs
    the ``1 - mass`` unmatched source mass, and one *phantom* source that fills
    the ``1 - mass`` unfilled sink demand (at zero cost).  The phantom->dump
    edge is forbidden, which forces exactly ``mass`` unit of mass through the
    real real->real block.

    Args
    ----
    C     : [B, n, m] cost matrix (>= 0)
    reg   : entropic regularisation (>= 0)
    mass  : transported mass fraction in (0, 1]; mass=1 recovers *full* OT
    n_iter: Sinkhorn sweeps per temperature stage

    Returns
    -------
    P    : [B, n, m] transport plan of the real block (sums to ``mass``)
    cost : [B]  <P, C> / mass      (average cost per transported unit mass)
    """
    C = _as_3d(C).detach()
    B, n, m = C.shape
    dev, dt = C.device, C.dtype

    a = torch.full((n,), 1.0 / n, device=dev, dtype=dt)
    b = torch.full((m,), 1.0 / m, device=dev, dtype=dt)
    slack = max(0.0, 1.0 - float(mass))
    if slack > 0:
        a_aug = torch.cat([a, torch.full((1,), slack, device=dev, dtype=dt)])
        b_aug = torch.cat([b, torch.full((1,), slack, device=dev, dtype=dt)])
    else:                                     # m == 1 -> plain balanced OT
        a_aug, b_aug = a, b

    nu, nv = a_aug.numel(), b_aug.numel()
    Caug = torch.zeros(B, nu, nv, device=dev, dtype=dt)
    Caug[:, :n, :m] = C
    if slack > 0:
        Caug[:, n, m] = forbid                # phantom source must not be dumped

    loga = torch.log(a_aug.clamp_min(1e-30)).view(1, nu, 1)
    logb = torch.log(b_aug.clamp_min(1e-30)).view(1, 1, nv)

    f = torch.zeros(B, nu, 1, device=dev, dtype=dt)
    g = torch.zeros(B, 1, nv, device=dev, dtype=dt)

    if eps_scaling and reg > 0:
        stages = [4.0, 2.0, 1.0, 1.0]
    else:
        stages = [1.0]
    for sc in stages:
        r = max(reg * sc, 1e-6)
        logK = (-Caug / r).contiguous()
        for _ in range(max(1, n_iter)):
            f = loga - torch.logsumexp(g + logK, dim=2, keepdim=True)
            g = logb - torch.logsumexp(f + logK, dim=1, keepdim=True)

    if not return_plan:
        P = torch.exp(torch.clamp(f + g + logK, min=-80.0))
        return None, (P[:, :n, :m] * C).sum(dim=(1, 2)).div(max(mass, 1e-8))

    P = torch.exp(torch.clamp(f + g + logK, min=-80.0))[:, :n, :m]
    cost = (P * C).sum(dim=(1, 2)) / max(mass, 1e-8)
    return P, cost


def partial_ot_rerank(q_coarse, ref_coarse, reg=0.05, mass=0.8, lam_pos=0.05,
                      n_iter=120, return_plan=False, return_match=False):
    """Re-rank reference images by anomaly-resilient partial-transport cost.

    Args
    ----
    q_coarse   : [n_tok, C]           query coarse tokens (raw, will be normalised)
    ref_coarse : [K0, n_tok, C]       shortlisted reference coarse tokens
    lam_pos    : weight of the *soft* spatial bias  lam * ||p_i - p_j||^2

    Returns
    -------
    cost  : [K0]        average partial-transport cost (lower = better match)
    plan  : [K0, n, m]  (optional)
    match : [n]         (optional) argmax correspondence into the coarse grid,
                        aggregated (summed) over references
    """
    K0, n_tok, C = ref_coarse.shape
    dev, dt = ref_coarse.device, ref_coarse.dtype

    qn = F.normalize(q_coarse, dim=-1)                     # [n_tok, C]
    rn = F.normalize(ref_coarse, dim=-1)                   # [K0, n_tok, C]

    cost_feat = 1.0 - torch.einsum('ic,kjc->kij', qn, rn)  # [K0, n, m]

    if lam_pos > 0:
        g = int(round(math.sqrt(n_tok)))
        assert g * g == n_tok, f'coarse token count {n_tok} is not a square grid'
        coords = torch.stack(torch.meshgrid(
            torch.arange(g, device=dev, dtype=dt),
            torch.arange(g, device=dev, dtype=dt), indexing='ij'), dim=-1)
        coords = (coords.reshape(-1, 2) + 0.5) / g                        # [n,2]
        pd = torch.cdist(coords, coords) ** 2                             # [n,n]
        cost_feat = cost_feat + lam_pos * pd.view(1, n_tok, n_tok)

    plan, cost = partial_ot_plan(cost_feat, reg=reg, mass=mass, n_iter=n_iter,
                                 return_plan=True)

    out = (cost,)
    if return_plan:
        out += (plan,)
    if return_match:
        # how much mass each query coarse token sends, and where it lands
        w = plan.sum(dim=0)                                               # [n, m]
        out += (w.argmax(dim=-1),)                                        # [n]
    return out if len(out) > 1 else out[0]


# ============================================================================
# 3) transport-guided candidate context  (replaces RAD's fixed rho window)
# ============================================================================
def unmatched_mass_field(plan_sel, G, H, W):
    """Per-query-node *unmatched* mass of the partial plan, lifted to the fine grid.

    This is the second, independent use of the *partial* formulation: a query
    coarse cell that cannot be matched to any normal structure keeps its mass
    unmatched, so ``a_i - sum_j P_ij`` is itself a coarse anomaly statistic.
    Full OT (``mass = 1``) forces every cell to send all of its mass and
    therefore has no such signal -- which is exactly the ``m = 1`` ablation.

    plan_sel : [Kr, n, m]  plans of the *selected* references (n = m = G*G)
    returns  : [H*W] in [0, 1]  (normalised by the uniform node mass 1/n)
    """
    n = G * G
    if plan_sel.numel() == 0:
        return torch.zeros(H * W, device=plan_sel.device, dtype=plan_sel.dtype)
    sent = plan_sel.sum(dim=-1).mean(dim=0)                  # [n] mass sent per node
    un = (1.0 / n - sent).clamp_min(0.0) * n                 # -> [0,1]
    up = F.interpolate(un.view(1, 1, G, G), size=(H, W), mode='nearest')
    return up.reshape(-1)


def build_cand_context(H, W, r, device, center_rc=None):
    """Static gather tables for the local candidate pool.

    For every query position p (row-major over HxW) and every offset
    ``d`` in the (2r+1)^2 window, produce the *bank* flat index of the patch at
    ``p + center[p] + d`` (clamped) plus a validity flag.

    center_rc : optional [H*W, 2] per-position (dr, dc) shift -- this is what
                the partial-transport correspondence supplies instead of RAD's
                constant (0, 0).
    """
    rows = torch.arange(H, device=device)
    cols = torch.arange(W, device=device)
    RR = rows[:, None].expand(H, W).reshape(-1)          # [L]
    CC = cols[None, :].expand(H, W).reshape(-1)          # [L]
    if center_rc is not None:
        RR = RR + center_rc[:, 0].to(device)
        CC = CC + center_rc[:, 1].to(device)

    idx_list, valid_list, off_list = [], [], []
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            rr = RR + dr
            cc = CC + dc
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rr = rr.clamp(0, H - 1)
            cc = cc.clamp(0, W - 1)
            idx_list.append((rr * W + cc).contiguous())
            valid_list.append(ok.contiguous())
            off_list.append((dr, dc))
    return {
        'H': H, 'W': W, 'r': r, 'L': H * W, 'Kp': len(off_list),
        'idx': torch.stack(idx_list, dim=1),       # [L, Kp]
        'valid': torch.stack(valid_list, dim=1),   # [L, Kp]
        'offsets': off_list,
    }


@torch.no_grad()
def collect_candidates(bank, ctx, chunk=128, valid_mask=True, mem_mb=256):
    """Gather the local candidate feature pool for every query position.

    bank : [Kr, L, C]  (L2-normalised reference patch features)
    ctx  : build_cand_context(...)
    Yields (start, end, cand, ok) with cand [Pc, Kr*Kp, C], ok [Pc, Kr*Kp].

    The gather is split so that the transient [Pc, Kr*Kp, C] block stays under
    ``mem_mb``.  This matters because Kr and Kp both grow QUADRATICALLY with the
    window radius (Kp = (2*rho+1)^2), so the block grows as rho^4: it is ~30 MB
    at rho=1 but ~2.7 GB at rho=3, which raises CUDA OOM on a 24 GB card as soon
    as anything else shares the device -- and that is exactly why the rho=2/3
    runs of this project died while rho=1 was fine.  The yielded (start, end)
    contract is unchanged, so every caller, which writes into out[s0:s1], is
    unaffected.
    """
    Kr, L, C = bank.shape
    idx, valid = ctx['idx'], ctx['valid']
    Kp = ctx['Kp']
    per = max(1, Kr * Kp * C)
    sub = max(1, int((float(mem_mb) * 1024 * 1024 / 4) // per))
    for s0 in range(0, L, chunk):
        s1 = min(s0 + chunk, L)
        ii = idx[s0:s1]                                   # [Pc, Kp]
        Pc = s1 - s0
        for t0 in range(0, Pc, sub):
            t1 = min(t0 + sub, Pc)
            iis = ii[t0:t1]                               # [Pt, Kp]
            sel = bank.index_select(1, iis.reshape(-1))    # [Kr, Pt*Kp, C]
            cand = sel.reshape(Kr, t1 - t0, Kp, C).permute(1, 0, 2, 3).reshape(
                t1 - t0, Kr * Kp, C)
            ok = valid[s0 + t0:s0 + t1].unsqueeze(1).expand(
                t1 - t0, Kr, Kp).reshape(t1 - t0, Kr * Kp)
            yield s0 + t0, s0 + t1, cand, (ok if valid_mask else None)


# ----------------------------------------------------------------------------
# 3b) window similarities WITHOUT materialising the candidate block
# ----------------------------------------------------------------------------
# ``collect_candidates`` gathers bank[:, idx[p, kk], :] for every (p, kk) pair,
# i.e. it re-reads the SAME reference patch once per window offset.  At
# Real-IAD's rho=4 / K=900 / C=768 / L=784 that is
#     900 * 784 * 81 * 768 * 4 B = 175 GB of reads
# plus an equal volume of writes, plus the (Kr, Pt, Kp, C) -> (Pt, Kr, Kp, C)
# permute, which is a second full copy.  ~875 GB of traffic per layer per
# image; x4 layers that is what makes LONM cost 5.5 s/image on Real-IAD while
# RAD's own window walk (``patch_scores_positional``) costs 0.84 s on the same
# code, same card, same class.
#
# ``patch_scores_positional`` already shows the fix for the max-only case and
# its docstring states it outright: "整图整层只读写记忆库一次，而不是每个偏移各读
# 一次" -- shift the QUERY, then re-align the offsets, so the bank is streamed
# once.  ``_window_sim_fast`` / ``window_top_candidates`` are that same
# construction extended from ``max`` to ``topk``, so LONM can keep the top-M
# candidate VECTORS it needs.
#
# Arithmetic: the same inner products, summed in a different order.
# ``test_agg.py`` measured the deviation for this exact change of layout on
# MVTec at 4e-10 relative (hard maximum 1e-7) with every printed metric
# identical to the last digit.
@torch.no_grad()
def _window_sim_fast(q_norm, bank, ctx, chunk=32):
    """[L, Kr*Kp] window similarities with the bank read ONCE.

    ``sim[p, m*Kp + kk] == <bank[m, idx[p, kk]], q_norm[p]>`` -- numerically the
    same tensor ``collect_candidates`` + ``bmm`` produce, and in the same flat
    (reference-major, then offset) order, so a downstream ``topk`` breaks ties
    identically.

    Returns the FULL [L, Kr*Kp] tensor; callers that only need per-position
    reductions can walk it in ``chunk``-sized slices.
    """
    H, W, r = int(ctx['H']), int(ctx['W']), int(ctx['r'])
    idx, offsets = ctx['idx'], ctx['offsets']
    Kp = int(ctx['Kp'])
    Kr, Lb, C = bank.shape
    L = q_norm.shape[0]
    if Lb != L:
        raise RuntimeError(
            'fast window needs bank L == query L, got %d vs %d' % (Lb, L))
    dt, dev = q_norm.dtype, q_norm.device

    # Query-side window: Qc[p, :, kk] = q_norm[p - offset_kk]  (zero-padded and
    # therefore clamped at the border, exactly like RAD's own tables).
    qp = F.pad(q_norm.reshape(H, W, C).permute(2, 0, 1), (r, r, r, r))
    Qc = torch.empty(L, C, Kp, dtype=dt, device=dev)
    for kk, (dr, dc) in enumerate(offsets):
        Qc[:, :, kk] = qp[:, r - dr:r - dr + H, r - dc:r - dc + W].reshape(
            C, L).t()

    # sim[p, m, kk] = <bank[m, idx[p, kk]], q[p]>.  The middle step below forms
    # <bank[m, a], q[a - offset_kk]> for every grid position a with ONE matmul
    # (the dual role of `a` is what lets the re-alignment be a pure gather):
    # for a valid (p, kk) pair idx[p, kk] == p + offset_kk, so a - offset_kk
    # collapses back to p.  Clamped (invalid) cells get a meaningless value and
    # are overwritten with -inf by the caller, exactly as the gather path does.
    sim = torch.empty(L, Kr * Kp, dtype=dt, device=dev)
    for b0 in range(0, Kr, chunk):
        b1 = min(b0 + chunk, Kr)
        m = b1 - b0
        # [L, m, C] @ [L, C, Kp] -> [L, m, Kp]
        out = torch.bmm(bank[b0:b1].permute(1, 0, 2).contiguous(), Qc)
        sel = torch.gather(out, 0, idx.unsqueeze(1).expand(L, m, Kp))
        sim[:, b0 * Kp:b1 * Kp] = sel.reshape(L, m * Kp)
    return sim


@torch.no_grad()
def window_top_candidates(q_norm, bank, ctx, Msel, chunk=32, fast=False):
    """Yield (s0, s1, Z) with Z [Pc, Msel, C] the top-``Msel`` candidate vectors.

    Drop-in replacement for the ``bmm -> masked_fill -> topk -> gather(cand)``
    triple that ``lonm_score`` used to run on top of ``collect_candidates``:
    the yielded Z is the same tensor, the iteration spans the same [0, L) in
    the same order, and the mask/ordering are applied at the same points.

    ``fast=False`` walks the historical gather path (kept so every existing
    table stays reproducible on the code that produced it); ``fast=True``
    never materialises the [Pc, Kr*Kp, C] block.
    """
    L = q_norm.shape[0]
    if not fast:
        for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
            sim = torch.bmm(cand, q_norm[s0:s1].unsqueeze(-1)).squeeze(-1)
            if ok is not None:
                sim = sim.masked_fill(~ok, float('-inf'))
            _, topi = sim.topk(Msel, dim=-1)
            C = cand.shape[-1]
            yield s0, s1, torch.gather(
                cand, 1, topi.unsqueeze(-1).expand(-1, -1, C))
        return

    Kr, Lb, C = bank.shape
    Kp = int(ctx['Kp'])
    idx, valid = ctx['idx'], ctx['valid']
    bank_flat = bank.reshape(Kr * L, C)

    sim = _window_sim_fast(q_norm, bank, ctx, chunk=chunk)
    # in-place: `sim` is freshly allocated above and never reused
    sim.view(L, Kr, Kp).masked_fill_(~valid.unsqueeze(1), float('-inf'))

    for s0 in range(0, L, chunk):
        s1 = min(s0 + chunk, L)
        _, topi = sim[s0:s1].topk(Msel, dim=-1)                  # [Pc, Msel]
        mm = torch.div(topi, Kp, rounding_mode='floor')
        kk = topi - mm * Kp
        row = mm * L + torch.gather(idx[s0:s1], 1, kk)
        yield s0, s1, bank_flat.index_select(
            0, row.reshape(-1)).reshape(s1 - s0, Msel, C)


# ============================================================================
# 4) LONM — local osculating (second-order) normal manifold score
# ============================================================================
def _quadratic_basis(y):
    """y: [P, M, d] -> [P, M, d(d+1)/2] with orthonormal-ish quadratic terms."""
    P, M, d = y.shape
    terms = []
    for i in range(d):
        terms.append(y[..., i:i + 1] ** 2)
    s2 = math.sqrt(2.0)
    for i in range(d):
        for j in range(i + 1, d):
            terms.append(s2 * y[..., i:i + 1] * y[..., j:j + 1])
    return torch.cat(terms, dim=-1) if terms else y[..., :0]


def _radial_basis(y):
    """y: [P, M, d] -> [P, M, 1].

    Isotropic (direction-averaged) second-order term ``||y||^2``.  Fitting a
    single curvature coefficient per normal direction is the numerically stable
    way to estimate a local second fundamental form: with ``nb = 1`` basis
    function and ``c`` normal directions the chart needs only ``c`` numbers,
    which is what M ~ 24 .. 48 local samples can actually support.  The full
    quadratic basis needs d(d+1)/2 * c parameters and is badly under-determined
    in the same regime (that is the absolute-ridge failure mode).
    """
    return (y ** 2).sum(dim=-1, keepdim=True)


@torch.no_grad()
def window_1nn(q_norm, bank, ctx, chunk=128, eps=1e-8):
    """Exact 1-NN dissimilarity restricted to the candidate window.

    Same candidate pool as ``lonm_score`` (so the only difference against the
    full model is the scoring rule), used for the "transport-guided window"
    ablation without the manifold term.
    """
    L = q_norm.shape[0]
    best = torch.full((L,), -1.0, device=q_norm.device, dtype=q_norm.dtype)
    for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
        sim = torch.bmm(cand, q_norm[s0:s1].unsqueeze(-1)).squeeze(-1)
        if ok is not None:
            sim = sim.masked_fill(~ok, float('-inf'))
        torch.maximum(best[s0:s1], sim.max(dim=1).values, out=best[s0:s1])
    return 1.0 - best


@torch.no_grad()
def window_ia_score(q_norm, bank, ctx, chunk=128, K_ia=0, eps=1e-8):
    """Order-statistic ("interval average") aggregation over the reference bank.

    RAD reduces the local candidate pool with a hard max over BOTH axes at once
    (reference image x window offset).  That max is the K = 1 order statistic:
    the score of every patch is decided by a single (image, offset) pair, so one
    reference image that happens to contain a similar-looking *normal* pattern
    collapses the score of a genuinely anomalous patch -- the same failure mode
    that motivated partial transport, but on the aggregation axis rather than on
    the retrieval axis.

    MuSc (ICLR 2024) replaces it with the "interval average": take the best
    match inside the window FOR EACH reference image, then average the K best
    images instead of keeping only the very best one.  K is a small fraction of
    the bank, so the statistic stays sharp on true anomalies while its variance
    drops by roughly a factor K.

    K_ia == 1 is bit-exactly RAD's hard max -- ``per_image_best.max()`` -- so the
    whole family is a strict generalisation of the baseline rule rather than a
    different rule.  ``test_agg.py`` asserts that identity.

    The pooled candidate order produced by ``collect_candidates`` is
    image-major (flat index = m * Kp + kk), which is what makes the split into
    (image, offset) axes below a pure ``view``.
    """
    Kr, L, C = bank.shape
    Kp = ctx['Kp']
    K = int(K_ia) if int(K_ia) > 0 else max(1, int(round(0.10 * Kr)))
    K = max(1, min(K, Kr))

    per_img = torch.zeros(L, Kr, device=q_norm.device, dtype=q_norm.dtype)
    for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
        sim = torch.bmm(cand, q_norm[s0:s1].unsqueeze(-1)).squeeze(-1)  # [P, Kr*Kp]
        if ok is not None:
            sim = sim.masked_fill(~ok, float('-inf'))
        P = s1 - s0
        best_in_window = sim.view(P, Kr, Kp).max(dim=2).values
        # `valid` is shared across reference images, and the zero-offset cell is
        # always valid, so a row can never be all -inf; nan_to_num is a guard.
        per_img[s0:s1] = torch.nan_to_num(best_in_window, neginf=0.0)

    top, _ = torch.sort(per_img, dim=1, descending=True)
    return 1.0 - top[:, :K].mean(dim=1)


def _maha_normalise(rq, R, r_iso, Msel, d_max, rank=0, ridge_rel=0.1,
                    min_var=1e-12, eps=1e-8):
    """Per-DIRECTION normalisation of the normal-space residual (Mahalanobis).

    rq    : [P, C]     the query's residual  q - q_hat  (NOT normalised)
    R     : [P, M, C]  the normal candidates' residuals  X - y_m T.  These span
           the local *normal* bundle the same way the tangent basis spans the
           tangent bundle, so their spectrum is a per-direction estimate of how
           much deviation is still normal.
    r_iso : [P]        ||rq||, the isotropic fallback magnitude.

    Why this is NOT the 13-times-failed "divide by the local scale": every one
    of those divided ALL components by ONE scalar, which rescales the defect
    signal together with the noise and destroyed the ranking.  Here each
    principal direction of the normal residual bundle is divided by its OWN
    standard deviation, so a direction in which normal patches routinely wobble
    by 0.02 cannot fire on a 0.02 wobble, while a direction that is normally
    rigid fires immediately.  The directions and their scales are estimated from
    the *normal* candidates only, so nothing about the query leaks in.

    Only min(M-1-d, C) directions are observable (M ~ 24 candidates cannot span
    the 768-dim normal bundle), so the energy that lies outside the sampled
    subspace is normalised isotropically by the mean directional variance
    instead of being amplified by the smallest one.
    """
    P = R.shape[0]
    cap = int(rank) if (rank and rank > 0) else (Msel - 1 - int(d_max))
    cap = max(1, min(cap, Msel - 1))

    GR = torch.bmm(R, R.transpose(1, 2))                     # [P,M,M]
    ev, U = torch.linalg.eigh(GR)                            # ascending
    ev = ev.flip(-1).clamp_min(0.0)
    U = U.flip(-1)
    S = ev[:, :cap].clamp_min(min_var).sqrt()                # [P,cap]
    Uc = U[:, :, :cap]
    V = torch.bmm(R.transpose(1, 2), Uc) / S.unsqueeze(1)    # [P,C,cap]

    t = torch.bmm(V.transpose(1, 2), rq.unsqueeze(-1)).squeeze(-1)   # [P,cap]
    var = (S ** 2) / float(max(1, Msel - 1))                 # [P,cap]
    vmean = var.mean(dim=-1, keepdim=True)
    var = var + float(ridge_rel) * vmean + min_var

    m2 = (t ** 2 / var).sum(dim=-1)                          # [P]
    in2 = (t ** 2).sum(dim=-1)
    out2 = (r_iso ** 2 - in2).clamp_min(0.0)                 # outside span(R)
    return (m2 + out2 / vmean.squeeze(-1)).clamp_min(0.0).sqrt()


def _finalize_norm(num, scale, eps=1e-8, norm_mode='ratio', scale_beta=0.0):
    """Turn a raw manifold distance into a *locally calibrated* non-conformity.

    ``scale`` is the local null envelope (a high quantile of the normal
    candidates' own residual distances).  Dividing by it pointwise is
    scale-invariant, which is exactly what breaks P-AP: in a region whose noise
    floor is 1e-3, a query that deviates by 3e-3 -- utterly negligible in cosine
    units -- is scored as high as a real defect in a region whose floor is 0.2.
    ``scale_beta > 0`` shrinks the local scale towards the image-level median
    scale, i.e. ``scale_eff = scale + beta * median(scale)`` (the standard
    James-Stein style shrinkage for a per-cell scale estimate, here with a fixed
    image-level target).  ``norm_mode='sat'`` additionally bounds the result to
    [0, 1), which keeps the 4-layer equal-weight fusion well behaved.
    """
    if norm_mode == 'maha':
        # The Mahalanobis form is already final: the *directional* normalisation
        # happened inside the loop, where the local normal-space spectrum lives.
        # Nothing is divided by a scalar here -- that is the whole point.
        return num
    if norm_mode == 'raw':
        # plain distance to the fitted local normal subspace.  This is NOT the
        # same statistic as 1 - cos(q, q_hat): the cosine form also mixes in the
        # *tangential* magnitude of the query, so two patches at the same
        # perpendicular distance get different scores purely because of where
        # they sit inside the local chart.
        return num
    if scale_beta and scale_beta > 0:
        se = scale + scale_beta * scale.median()
    else:
        se = scale
    if norm_mode == 'sub':
        # subtract the local noise floor instead of dividing by it: this keeps
        # the absolute scale (so a near-flat region cannot blow up) while still
        # removing the sampling-density-dependent floor of the raw distance.
        return (num - se).clamp_min(0.0)
    if norm_mode == 'sat':
        return num / (num + se + eps)
    return num / (se + eps)


@torch.no_grad()
def calibrated_1nn(q_norm, bank, ctx, M=24, chunk=128, eps=1e-8,
                   scale_q=0.9, norm_mode='ratio', scale_beta=0.0):
    """Locally calibrated distance-to-normal-mean (the ``d_max == 0`` LONM).

    Isolates the effect of *calibration* from the effect of the local subspace:
    the raw statistic stays first order -- ``||q - z0||`` with ``z0`` the mean of
    the M nearest normal candidates -- but it is divided by the ``scale_q``
    quantile of the normal candidates' own distances to ``z0``.  This is the
    no-geometry control for the local non-conformity score.
    """
    L = q_norm.shape[0]
    C = q_norm.shape[1]
    nums = torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
    scs = torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
    for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
        q = q_norm[s0:s1]
        sim = torch.bmm(cand, q.unsqueeze(-1)).squeeze(-1)
        if ok is not None:
            sim = sim.masked_fill(~ok, float('-inf'))
        Msel = min(M, cand.shape[1])
        _, topi = sim.topk(Msel, dim=-1)
        Z = torch.gather(cand, 1, topi.unsqueeze(-1).expand(-1, -1, C))
        z0 = Z.mean(dim=1, keepdim=True)                       # [Pc,1,C]
        nums[s0:s1] = torch.linalg.norm(q - z0.squeeze(1), dim=-1)
        ri = torch.linalg.norm(Z - z0, dim=-1)                 # [Pc,M]
        kq = min(Msel - 1, max(0, int(round(scale_q * (Msel - 1)))))
        scs[s0:s1] = torch.sort(ri, dim=-1).values[:, kq]
    return _finalize_norm(nums, scs, eps, norm_mode, scale_beta)


@torch.no_grad()
def lonm_score(q_norm, bank, ctx, M=24, d_max=4, energy=0.9, ridge=1e-4,
               trust_q=0.9, chunk=128, eps=1e-8, min_var=1e-12,
               quadratic=True, ridge_rel=0.0, res_rank=0, curv_gain=1.0,
               radial=False, norm_scale=False, scale_q=0.9, norm_mode='ratio',
               scale_beta=0.0, maha=False, maha_ridge=0.1, maha_rank=0,
               anc_beta=0.0, fast_window=False, unbounded=False, clip_stat=None):
    """Local Osculating Normal Manifold anomaly score, vectorised.

    q_norm : [L, C]      query patch features (L2-normalised)
    bank   : [Kr, L, C]  candidate reference patch features (L2-normalised)
    ctx    : build_cand_context(...)

    Returns
    -------
    score : [L]   1 - cos(q, q_hat)  with q_hat the query's projection onto the
                  locally fitted second-order normal manifold.

    ``quadratic=False`` keeps only the first-order (tangent plane) term, i.e. a
    local-PCA / SubspaceAD-style reconstruction control.  ``d_max == 0``
    degenerates to plain 1-NN dissimilarity.

    ``unbounded=True`` removes the finite coordinate radius (alpha == 1), i.e.
    the R = inf limit of the reference set.  Default False leaves every stored
    result bit-identical.  ``clip_stat``, when a dict is passed, accumulates the
    radius-clipping diagnostics (``n``, ``clip``, ``ratio_sum``, ``ratio_max``)
    over all chunks.

    Regularisation of the curvature fit
    -----------------------------------
    ``ridge_rel`` > 0 makes the Tikhonov term relative to the scale of
    ``Phi^T Phi`` (recommended: 1e-2 .. 1e0).  ``res_rank`` > 0 fits the
    curvature only inside the top-``res_rank`` directions of the perpendicular
    residual bundle, which is the numerically stable way to estimate a local
    second fundamental form from M << C samples.  ``curv_gain`` scales the
    curvature correction (0 recovers the tangent-only model).  ``radial``
    switches the curvature basis from the full quadratic monomials to the
    isotropic ``||y||^2`` term (recommended together with ``res_rank = d``).
    """
    L, C = q_norm.shape

    if d_max == 0:                                    # 1-NN control
        if norm_scale:
            return calibrated_1nn(q_norm, bank, ctx, M=M, chunk=chunk, eps=eps,
                                  scale_q=scale_q, norm_mode=norm_mode,
                                  scale_beta=scale_beta)
        return window_1nn(q_norm, bank, ctx, chunk=chunk, eps=eps)

    score = torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
    num_all = (torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
               if (norm_scale or maha) else None)
    sc_all = (torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
              if (norm_scale or maha) else None)
    dt = q_norm.dtype
    dev = q_norm.device

    # ``window_top_candidates`` yields exactly the Z the gather path produced
    # (same flat (reference, offset) order, same -inf masking before the topk),
    # so everything below is untouched.  ``fast_window=True`` only changes HOW
    # the similarities are formed -- see its docstring and test_window_fast.py.
    Msel = min(M, bank.shape[0] * int(ctx['Kp']))
    for s0, s1, Z in window_top_candidates(q_norm, bank, ctx, Msel,
                                           chunk=chunk, fast=fast_window):
        Pc = s1 - s0
        q = q_norm[s0:s1]                                      # [Pc, C]

        # ---- neighbourhood coherence weighting (contribution (2), the
        # ANoCo (CVPR 2026) anchor-consistency idea transplanted onto the
        # multi-layer bank)
        #
        # The M nearest bank patches are currently averaged with EQUAL weights,
        # so `z0` -- the anchor of the local chart -- is a plain centroid.  The
        # M nearest neighbours of a query patch are not equally trustworthy
        # though: ANoCo builds its normal manifold only from reference patches
        # that are compatible with the query's single best match (the
        # "anchor"), because the raw top-K set can be "locally plausible but
        # not mutually coherent" once one reference image is itself defective
        # -- which is exactly diagnostic B of this project, and the measured
        # cause of the upper-tail false positives that cap pixel-AP.
        #
        # ANoCo cuts with a hard threshold (keep j iff cos(anchor, f_j) >
        # cos(query, anchor)).  With only M = 24 slots a hard cut throws away
        # too much support, so this is the soft form: weight each neighbour by
        # its affinity to the anchor and take the WEIGHTED centroid / weighted
        # covariance.  anc_beta = 0 gives w == 1/M and sw == 1 exactly, so the
        # whole branch is bit-identical to the uniform version (asserted in
        # test_dad.py).  The sign of anc_beta is the actual question the data
        # has to answer: beta > 0 sharpens the chart onto near-identical
        # patches, beta < 0 broadens it onto the full spread of normal
        # appearance inside the window.
        sw = None
        if anc_beta != 0.0:
            a = torch.bmm(Z, Z[:, :1, :].transpose(1, 2)).squeeze(-1)  # [Pc,M]
            w = torch.softmax(anc_beta * a, dim=-1)                    # sums to 1
            z0 = (Z * w.unsqueeze(-1)).sum(dim=1, keepdim=True)        # [Pc,1,C]
            sw = torch.sqrt(w * Msel)                                  # [Pc,M]
            X = (Z - z0) * sw.unsqueeze(-1)                            # [Pc,M,C]
        else:
            z0 = Z.mean(dim=1, keepdim=True)                        # [Pc,1,C]
            X = Z - z0                                              # [Pc,M,C]

        # ---- local tangent space (eigh on the tiny MxM Gram matrix)
        G = torch.bmm(X, X.transpose(1, 2))                     # [Pc,M,M]
        evals, evecs = torch.linalg.eigh(G)                     # ascending
        evals = evals.flip(-1).clamp_min(0.0)
        evecs = evecs.flip(-1)                                  # [Pc,M,M] U
        d = min(d_max, Msel)
        Sd = evals[:, :d].clamp_min(min_var).sqrt()             # [Pc,d]
        Ud = evecs[:, :, :d]                                    # [Pc,M,d]

        # right singular vectors V_d = X^T U_d / sigma_d
        Vd = torch.bmm(X.transpose(1, 2), Ud) / Sd.unsqueeze(1)  # [Pc,C,d]
        T = Vd.transpose(1, 2)                                   # [Pc,d,C]
        y = Ud * Sd.unsqueeze(1)                                 # [Pc,M,d]
        # X = sum_k y_k V_k, so y holds the coordinates of the *weighted*
        # offsets; dividing by sw recovers the coordinates of the true offset
        # Z_j - z0, which is the space the query is projected into.  sw == 1
        # whenever anc_beta == 0, so this is a no-op on every existing row.
        y_off = y if sw is None else y / sw.unsqueeze(-1)        # [Pc,M,d]

        # ---- per-patch intrinsic dimension d_p (energy criterion)
        ev = evals[:, :d]
        cum = torch.cumsum(ev, dim=-1) / ev.sum(dim=-1, keepdim=True).clamp_min(min_var)
        d_p = (cum < energy).sum(dim=-1) + 1                     # [Pc]
        d_p = d_p.clamp(min=2, max=d)
        jj = torch.arange(d, device=dev).view(1, d)
        mask = (jj < d_p.view(-1, 1)).to(dt).unsqueeze(1)        # [Pc,1,d]

        y_m = y * mask                                           # [Pc,M,d]
        R = X - torch.bmm(y_m, T)                                # perpendicular residual

        # ---- curvature fit, optionally inside the dominant residual directions
        VR = None
        if quadratic and curv_gain != 0.0:
            c = 0
            if res_rank and res_rank > 0:
                c = int(min(res_rank, d, Msel))
            elif radial:
                # the radial basis has a single column, so without a residual
                # restriction the solve would regress C = 768 outputs; cap it.
                c = int(min(d, Msel))
            if c > 0:
                GR = torch.bmm(R, R.transpose(1, 2))             # [Pc,M,M]
                evR, evecR = torch.linalg.eigh(GR)
                evR = evR.flip(-1).clamp_min(0.0)
                evecR = evecR.flip(-1)
                SR = evR[:, :c].clamp_min(min_var).sqrt()        # [Pc,c]
                UR = evecR[:, :, :c]                             # [Pc,M,c]
                VR = torch.bmm(R.transpose(1, 2), UR) / SR.unsqueeze(1)  # [Pc,C,c]
                Rr = torch.bmm(R, VR)                            # [Pc,M,c]
            else:
                Rr = R
            Phi = _radial_basis(y_m) if radial else _quadratic_basis(y_m)  # [Pc,M,nb]
            nb = Phi.shape[-1]
            PtP = torch.bmm(Phi.transpose(1, 2), Phi)
            if ridge_rel and ridge_rel > 0:
                lam = ridge_rel * (PtP.diagonal(dim1=-2, dim2=-1).mean(-1)
                                   .view(-1, 1, 1)) + min_var
            else:
                lam = ridge * torch.ones(Pc, 1, 1, device=dev, dtype=dt)
            A = torch.linalg.solve(
                PtP + lam * torch.eye(nb, device=dev, dtype=dt).unsqueeze(0),
                torch.bmm(Phi.transpose(1, 2), Rr))              # [Pc,nb,c]

        # ---- query projection with a local trust region
        yn = torch.linalg.norm(y_off * mask, dim=-1)             # [Pc,M]
        kq = min(Msel - 1, max(0, int(round(trust_q * (Msel - 1)))))
        r_trust = torch.sort(yn, dim=-1).values[:, kq]           # [Pc]

        yq = torch.bmm(q.unsqueeze(1) - z0, T.transpose(1, 2)).squeeze(1)  # [Pc,d]
        yq = yq * mask.squeeze(1)
        nyq = torch.linalg.norm(yq, dim=-1).clamp_min(eps)        # [Pc]
        scale = torch.clamp(r_trust / nyq, max=1.0).unsqueeze(-1)
        if clip_stat is not None:
            # Radius-clipping diagnostics.  ``ratio`` is ||a|| / R, so
            # ratio > 1 means the query coordinate exceeded the fitted
            # extent and ``scale`` dropped below 1.  Recorded for both the
            # bounded and the unbounded arm so the two are directly
            # comparable; the activated fraction is what decides whether a
            # finite radius can matter numerically at all.
            ratio = (nyq / r_trust.clamp_min(eps)).clamp_max(1e3)
            clip_stat['n'] = clip_stat.get('n', 0) + int(ratio.numel())
            clip_stat['clip'] = clip_stat.get('clip', 0) + int((ratio > 1.0).sum().item())
            clip_stat['ratio_sum'] = clip_stat.get('ratio_sum', 0.0) + float(ratio.sum())
            clip_stat['ratio_max'] = max(clip_stat.get('ratio_max', 0.0),
                                         float(ratio.max()))
        if unbounded:
            # R = inf: alpha == 1 for every query.  The reference becomes the
            # unbounded local affine span, so the readout is the orthogonal
            # residual only.  This is the radius control the paper needs; it
            # changes nothing unless unbounded=1 (default 0).
            scale = torch.ones_like(scale)
        yq = yq * scale

        qhat = z0.squeeze(1) + torch.bmm(yq.unsqueeze(1), T).squeeze(1)
        if quadratic and curv_gain != 0.0:
            bq = (_radial_basis(yq.unsqueeze(1)) if radial
                  else _quadratic_basis(yq.unsqueeze(1)))
            pred = torch.bmm(bq, A).squeeze(1)                    # [Pc,c]
            if VR is not None:
                pred = torch.bmm(pred.unsqueeze(1), VR.transpose(1, 2)).squeeze(1)
            qhat = qhat + curv_gain * pred

        if norm_scale or maha:
            # norm_scale: local scale = how well the *normal* candidates
            # themselves are explained by the fitted chart (leave-one-out so the
            # scale is not biased by in-sample fitting).  Dividing by it turns
            # the raw manifold distance into a local non-conformity, which
            # removes the systematic score inflation of high-variance but normal
            # regions.
            c_hat = torch.bmm(y_m, T)
            if quadratic and curv_gain != 0.0:
                pm = torch.bmm(Phi, A)                                    # [Pc,M,c]
                if VR is not None:
                    pm = torch.bmm(pm, VR.transpose(1, 2))
                c_hat = c_hat + curv_gain * pm
            res_i = X - c_hat                                             # [Pc,M,C]
            if sw is not None:
                # undo the coherence weighting so that the local scale stays the
                # same statistic as before: a quantile over the neighbours' OWN
                # residuals.  Weighting it would make the scale shrink with the
                # spread of `sw`, i.e. bake the mechanism into two places.
                res_i = res_i / sw.unsqueeze(-1)
            if quadratic and curv_gain != 0.0:
                Inv = torch.linalg.inv(
                    PtP + lam * torch.eye(nb, device=dev, dtype=dt).unsqueeze(0))
                Hd = torch.bmm(Phi, torch.bmm(Inv, Phi.transpose(1, 2)))
                Hd = Hd.diagonal(dim1=-2, dim2=-1).clamp(max=1.0 - 1e-2)
                res_i = res_i / (1.0 - Hd).unsqueeze(-1)
            ri = torch.linalg.norm(res_i, dim=-1)                         # [Pc,M]
            ks = min(Msel - 1, max(0, int(round(scale_q * (Msel - 1)))))
            scale = torch.sort(ri, dim=-1).values[:, ks]                   # [Pc]
            np_ = torch.linalg.norm(q - qhat, dim=-1)
            if maha:
                # replace the isotropic magnitude by the directionally
                # normalised one (the local scalar scale is then unused)
                np_ = _maha_normalise(q - qhat, R, np_, Msel, d_max,
                                      maha_rank, maha_ridge, min_var, eps)
            num_all[s0:s1] = np_
            sc_all[s0:s1] = scale
            continue

        qhat = F.normalize(qhat, dim=-1, eps=1e-8)
        cos = (q * qhat).sum(dim=-1)
        score[s0:s1] = 1.0 - cos

    if norm_scale or maha:
        return _finalize_norm(num_all, sc_all, eps, norm_mode, scale_beta)
    return score


# ============================================================================
# 5) transport-guided correspondence field for the full-res patch grid
# ============================================================================
def transport_offset_field(plan_sum, coarse_g, H, W, max_shift):
    """Turn the aggregated coarse transport plan into a per-patch (dr, dc) shift.

    plan_sum : [G*G, G*G]  aggregated transport plan (query coarse cell x ref coarse cell)
    coarse_g : G   (7)
    Returns  shift [H*W, 2] long tensor
    """
    dev = plan_sum.device
    G = coarse_g
    tgt = plan_sum.argmax(dim=-1)                    # [G*G] -> ref coarse cell id
    tr = (tgt // G).to(torch.long)
    tc = (tgt % G).to(torch.long)
    # centre of the matched coarse cell in the fine grid
    step = H // G
    half = step // 2
    tr_f = tr * step + half
    tc_f = tc * step + half

    rr = torch.arange(H, device=dev).view(H, 1).expand(H, W).reshape(-1)
    cc = torch.arange(W, device=dev).view(1, W).expand(H, W).reshape(-1)
    qr = rr // step
    qc = cc // step
    tid = qr * G + qc

    dr = (tr_f[tid] - rr).clamp(-max_shift, max_shift)
    dc = (tc_f[tid] - cc).clamp(-max_shift, max_shift)
    return torch.stack([dr, dc], dim=1).to(torch.long)


# ============================================================================
# Per-image null calibration
# ----------------------------------------------------------------------------
# The pixel-AP of the RAD protocol pools EVERY test image of a class into a
# single ranking (curvot_rad.py:
#   ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel()) ).
# A per-image *monotone* transform therefore leaves both the image-level AUROC
# and the within-image ordering untouched, yet it CAN change the pooled AP,
# because it decides whether the normal-score tail of one image reaches above
# the defects of another image.  Texture strength varies enormously from image
# to image (a strongly textured 'leather' crop sits at a far higher absolute
# distance everywhere, normal patches included), and RAD leaves that inter-image
# offset in the score.  Removing it is the one thing the failed *pointwise*
# calibration (divide by the local scale) was never going to achieve.
# ============================================================================
def image_level_calibrate(scores, mode='none', q=0.5, eps=1e-6):
    """scores: [B, L] patch scores -> [B, L], per-image offset/scale removal.

    'sub_med'  subtract the per-image median
    'sub_q'    subtract the mean of the lowest q-fraction  (the normal floor;
               robust even when defects cover a large area of the image)
    'znorm'    (s - median) / (IQR + eps)  -- bounded removal, reported for
               completeness; expect this one to hurt for the same reason the
               pointwise local scale did.
    """
    if mode == 'none' or scores is None:
        return scores
    if mode == 'sub_med':
        return scores - scores.median(dim=-1, keepdim=True).values
    if mode == 'sub_q':
        k = max(1, int(round(float(q) * scores.shape[-1])))
        ref = torch.sort(scores, dim=-1).values[:, :k].mean(dim=-1, keepdim=True)
        return scores - ref
    if mode == 'znorm':
        med = scores.median(dim=-1, keepdim=True).values
        q75 = torch.quantile(scores, 0.75, dim=-1, keepdim=True)
        q25 = torch.quantile(scores, 0.25, dim=-1, keepdim=True)
        return (scores - med) / ((q75 - q25) + eps)
    raise ValueError(f'unknown img_cal mode {mode}')


def matched_pit(scores, u_field, u_q=0.5):
    """Per-image probability-integral transform w.r.t. the plan-vouched null.

    The between-image mismatch that limits the pooled P-AP can be removed by ANY
    per-image monotone transform, because none of them can touch the ordering
    inside an image.  Subtracting a scalar (``matched_null_offset``) is the
    crudest such transform: it shifts the null but keeps its width, so an image
    whose normal region is simply *noisier* still has a fatter null tail
    competing with other images' defects.  This maps each patch score to the
    fraction of that image's confidently matched (normal) patches sitting at or
    below it, i.e. to its own null CDF: every image's normal population becomes
    Uniform(0, 1) by construction, which aligns the tails as well as the centres.

    It is deliberately a *rank* transform, so it is immune to the score's units
    -- and for the same reason it throws away the magnitude of the anomaly, which
    is what ``pixel-AP`` partly relies on.  Both consequences are measured.
    """
    scores = scores.view(1, -1) if scores.dim() == 1 else scores
    null = scores
    if u_field is not None:
        u = u_field.reshape(-1).to(scores.dtype)
        m = u <= torch.quantile(u, float(u_q))
        if int(m.sum()) >= 16:
            null = scores[:, m]
    ns, _ = torch.sort(null, dim=-1)
    n = int(ns.shape[-1])
    idx = torch.searchsorted(ns.contiguous(), scores.contiguous(), right=True)
    return (idx.to(scores.dtype) + 0.5) / float(n + 1)


def matched_null_offset(scores, u_field, u_q=0.5, red='median', red_q=0.9):
    """Per-image offset estimated from the patches that partial OT DID match.

    u_field : [L] unmatched mass in [0, 1].  A convincingly matched patch has
    u ~ 0, an unmatchable one u ~ 1.  The scores of the confidently matched
    patches are therefore a plan-derived estimate of *this image's* normal
    level -- exactly the quantity that has to be removed before a class-wide
    pooled AP is computed.  This is the mechanism that couples the two
    innovations: the transport plan of APTR supplies the null model that LONM's
    score is calibrated against.
    """
    scores = scores.view(1, -1) if scores.dim() == 1 else scores
    if u_field is None:
        return scores.median(dim=-1, keepdim=True).values
    u = u_field.reshape(-1).to(scores.dtype)
    thr = torch.quantile(u, float(u_q))
    m = u <= thr
    if int(m.sum()) < 16:
        return scores.median(dim=-1, keepdim=True).values
    sel = scores[:, m]
    if red == 'mean':
        return sel.mean(dim=-1, keepdim=True)
    if red == 'q':
        # align the normal TAIL rather than the normal centre: what competes with
        # another image's defects in the pooled ranking is this image's normal
        # tail, so the quantity to put on a common level is a high quantile of
        # the null, not its median.
        return torch.quantile(sel, float(red_q), dim=-1).unsqueeze(-1)
    return sel.median(dim=-1).values.unsqueeze(-1)


def matched_null_stats(scores, u_field, u_q=0.5, eps=1e-6):
    """Return (offset, scale) of the *confidently matched* patches.

    'matched'   uses only the offset  -> s - off
    'matched_z' also divides by the spread of the matched set, i.e. it asks
                "how many normal-sigma away from this image's own normal floor
                is this patch", with both sigma and the floor estimated from the
                patches partial transport could actually match.  This is the
                one form of normalisation that is NOT the failed pointwise one:
                it is a single scalar per image (so the within-image ordering is
                kept) and it is estimated only from patches the plan vouches for.
    """
    scores = scores.view(1, -1) if scores.dim() == 1 else scores
    if u_field is not None:
        u = u_field.reshape(-1).to(scores.dtype)
        m = u <= torch.quantile(u, float(u_q))
        if int(m.sum()) >= 16:
            sel = scores[:, m]
            off = sel.median(dim=-1).values.unsqueeze(-1)
            hi = torch.quantile(sel, 0.95, dim=-1).unsqueeze(-1)
            return off, (hi - off).clamp_min(eps)
    off = scores.median(dim=-1, keepdim=True).values
    hi = torch.quantile(scores, 0.95, dim=-1, keepdim=True)
    return off, (hi - off).clamp_min(eps)


def matched_null_scale(scores, u_field, u_q=0.5, hi_q=0.95, eps=1e-6):
    """Per-image WIDTH (not level) of the patches partial transport matched.

    The diagnosis that motivates every cross-image transform in this project is
    a *scale* mismatch: on the classes where the pooled pixel-AP is far below the
    pixel-AUROC (VisA pcb2/pcb3: AUROC 0.99, AP 0.26), the within-image ordering
    is nearly perfect and the loss is that the images do not share a common
    normal score REGIME.  The offset form of the correction removes the level.
    On MVTec metal_nut / transistor that is exactly what destroys the score,
    because there the defect itself lifts the whole image's level, so removing
    the level removes evidence.

    The width is a different quantity and it is the one the offset cannot fix:
    an image whose normal surface is merely noisier has a fatter null tail that
    reaches up into other images' defect scores, and subtracting a constant does
    nothing about that.  Dividing by this width puts every image's null on a
    common *scale* while leaving every level untouched, so it cannot delete
    evidence -- which is the property the offset version lacks.

    The width is estimated from the plan-vouched null (u <= u_q), i.e. patches
    partial transport could actually match, so it inherits the coupling to APTR,
    and as a single scalar per image it is still a per-image monotone transform
    (within-image ordering is untouched by construction).

    Returns tau [B, 1] > 0.
    """
    scores = scores.view(1, -1) if scores.dim() == 1 else scores
    sel = None
    if u_field is not None:
        u = u_field.reshape(-1).to(scores.dtype)
        m = u <= torch.quantile(u, float(u_q))
        if int(m.sum()) >= 16:
            sel = scores[:, m]
    if sel is None:
        sel = scores
    lo = sel.median(dim=-1).values.unsqueeze(-1)
    hi = torch.quantile(sel, float(hi_q), dim=-1).unsqueeze(-1)
    return (hi - lo).abs().clamp_min(eps)


def centroid_score(q_norm, bank, ctx, M=24, mode='raw', chunk=128, eps=1e-8):
    """0-th order geometry control: score against the *local centroid*.

    The M nearest normal candidates inside the window are averaged into one
    local reference ``z0``; the query is then scored either

        mode='cos'   1 - cos(q, z0 / ||z0||)   (the RAD-style angular statistic)
        mode='raw'   ||q - z0||                (the absolute distance)

    Same candidate pool, same aggregation, same window -- the pair isolates the
    score FORM with nothing else changed.  For a *single* candidate the two are
    strictly monotone equivalent (||q - z||^2 = 2(1 - cos) when both are unit
    norm), so this control is what decides whether the gain of the absolute
    distance comes from the score form itself or only from the fact that a
    *reconstruction* (centroid, subspace projection, curvature correction) is
    not a unit vector, in which case the angular form throws away how far the
    reconstruction sits from the query.
    """
    L, C = q_norm.shape
    out = torch.empty(L, device=q_norm.device, dtype=q_norm.dtype)
    for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
        q = q_norm[s0:s1]
        sim = torch.bmm(cand, q.unsqueeze(-1)).squeeze(-1)
        if ok is not None:
            sim = sim.masked_fill(~ok, float('-inf'))
        Msel = min(M, cand.shape[1])
        _, topi = sim.topk(Msel, dim=-1)
        Z = torch.gather(cand, 1, topi.unsqueeze(-1).expand(-1, -1, C))
        z0 = Z.mean(dim=1)                                        # [Pc, C]
        if mode == 'raw':
            out[s0:s1] = torch.linalg.norm(q - z0, dim=-1)
        else:
            out[s0:s1] = 1.0 - (q * F.normalize(z0, dim=-1, eps=eps)).sum(dim=-1)
    return out


# ============================================================================
# 6) DAD -- dimensionality-aware deviation (DAO / SLOF on the candidate pool)
# ============================================================================
#
# Why this family exists
# ----------------------
# RAD scores a patch with the 1-NN distance to the normal bank.  A *distance*
# is not a *density*: two regions of the same test image can have the same
# distance to the bank and yet be wildly different evidence, because the normal
# population itself is spread out very differently from place to place.  The
# measured consequence in this project (PROGRESS / INNOVATION section 1,
# diagnosis C) is that pooled pixel-AP is far below pixel-AUROC on the
# self-similar textures -- the *upper tail of the normal scores*, not the
# ordering, is what kills AP.  A few perfectly normal patches sit in places
# where the reference population is intrinsically sparse, so their distance is
# large in absolute terms and they out-rank real defects.
#
# Every fix attempted before this one was a per-image statistic (subtract the
# image's level, divide by the image's width, standardise the image's layer
# maps).  All of them are wrong for the same reason, and the failures are on
# record: subtracting a level deletes defect evidence when the defect lifts the
# whole image (metal_nut / transistor, -0.0493), and standardising deletes the
# between-image scale that IS the image-level signal (u_c1zf, -0.237).
#
# DAO (Anderberg et al., SIAM SDM 2024, arXiv:2401.05453) takes the ratio of a
# query's k-NN distance to the k-NN distances of its own neighbours:
#
#     DAO_k(q) = (1/k) * sum_{o in NN_k(q)} ( kdist(q) / kdist(o) )^{LID(o)}
#
# and the paper's own remark is the key to why it is the right object here:
# its predecessor SLOF is "nearly identical ... with the exception that the
# k-NN distance ratio of DAO has exponent equal to the LID of the neighbour",
# i.e. SLOF "makes the implicit (but theoretically unjustified) assumption that
# the underlying local intrinsic dimensionalities are equal to 1 at every
# neighbour".
#
# Three properties make it the right fix for THIS project:
#
#  1. The denominator kdist(o) is computed on NORMAL BANK patches only, never on
#     the test image.  So an image that is anomalous everywhere still has a
#     large DAO -- the absolute evidence is not destroyed, which is exactly what
#     killed every per-image transform.
#  2. It still removes the between-region scale mismatch, but the scale it
#     removes is the *reference population's own* local scale, which is a
#     property of the normal data (estimable once, offline) rather than a
#     statistic of one test image.
#  3. It is a strict generalisation: LID == 1 for every neighbour collapses DAO
#     to SLOF, and this code exposes that as `lid_mode='one'` so the ablation
#     isolates exactly one thing -- whether the LID = 1 assumption is harmful.
#
# The bank side is query-free: kdist(o) and LID(o) depend only on the bank, so
# they are precomputed once per class and cached.


def _lid_from_sorted_sim(sims, metric='l2', eps=1e-12):
    """Amsaleg/Houle MLE estimator of local intrinsic dimensionality.

    sims : [..., k] the k LARGEST cosine similarities to the neighbour pool, in
           descending order, all valid.

        LID = -( (1/(k-1)) * sum_{j=1..k-1} log(d_j / d_k) )^{-1}
            =  (k-1) / sum_{j<k} log(d_k / d_j)

    Unit-norm features give d = sqrt(2 - 2s), hence
    ``d_k/d_j = sqrt((1-s_k)/(1-s_j))``.  The square root does NOT cancel: it
    multiplies the whole sum, so ``metric='cos'`` (which treats ``1 - s`` itself
    as the distance) returns exactly HALF the LID of ``metric='l2'``.  The
    metric is therefore a real modelling choice, not a detail -- the same lesson
    as contribution (1), where ``1 - cos`` and the true distance stopped being
    equivalent as soon as the reference point was a reconstruction.  ``l2`` is
    the default and ``cos`` is the ablation; ``test_dad.py`` asserts the exact
    factor 2 between them.
    """
    one_m = (1.0 - sims).clamp_min(eps)
    ratio = (one_m[..., -1:] / one_m[..., :-1]).clamp_min(1.0)
    lg = torch.log(ratio)
    if metric == 'l2':
        lg = 0.5 * lg
    num = lg.shape[-1]
    if num <= 0:
        return torch.zeros(sims.shape[:-1], device=sims.device, dtype=sims.dtype)
    return num / lg.sum(dim=-1).clamp_min(eps)


@torch.no_grad()
def cls_block_purity(sizes, cls_banks, device='cuda', n_sub=1500, seed=0):
    """1-NN CLS block purity of a candidate class-block partition.

    The checkpoint stores ONE bank covering every class, laid out as contiguous
    row blocks, but it records no item order.  Any candidate partition is fully
    described by the block sizes (which come from disk) plus a permutation.  This
    scores a candidate: the fraction of (subsampled) rows whose nearest CLS
    neighbour in the bank lands inside the same block.  Same-class images have
    near-identical global embeddings, so the true layout scores ~1.0.

    Measured on the real MVTec bank: alphabetical disk order 0.7716, true build
    order 1.0000 -- decisive, which is what makes the recovery auditable instead
    of assumed.

    sizes : block sizes, in the candidate order (list/tensor of ints)
    cls_banks : list of [N, C] CLS banks; every layer agrees, so all layers that
                match ``sum(sizes)`` are averaged.  Pure function of its inputs,
                hence unit-testable without a GPU or a dataset.
    """
    sizes = torch.as_tensor(list(sizes), dtype=torch.long)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    n_tot = int(starts[-1].item())
    lab = torch.zeros(n_tot, dtype=torch.long)
    for i in range(int(sizes.numel())):
        lab[int(starts[i]):int(starts[i + 1])] = i
    usable = [cb for cb in cls_banks if int(cb.shape[0]) == n_tot]
    if not usable:
        return float('nan')
    dev = torch.device(device)
    g = torch.Generator().manual_seed(int(seed))
    sub = torch.randperm(n_tot, generator=g)[:min(int(n_sub), n_tot)]
    sub_d = sub.to(dev)
    hit = 0
    with torch.no_grad():
        for cb in usable:
            X = F.normalize(cb.detach().to(dev, torch.float32), dim=-1)
            S = X[sub_d] @ X.t()
            S[torch.arange(sub_d.numel(), device=dev), sub_d] = -2.0
            hit += int((lab[S.argmax(dim=1).cpu()] == lab[sub]).sum())
    return hit / (sub.numel() * len(usable))


def bank_self_knn_stats(rows, ctx, k=8, device='cuda', chunk=16, refn=0,
                        metric='l2', seed=0, eps=1e-12, row_sel=None,
                        normalize=True):
    """Query-free ``kdist(o)`` and ``LID(o)`` for every bank patch of a class.

    rows : [Ntot, L, C] the whole normal bank.  It is usually a memory-mapped
           tensor; ``row_sel`` picks the rows of the class being evaluated and
           only the columns the current position chunk needs are touched, so
           neither the 32 GiB bank nor the 1.8 GiB class slice is ever
           materialised.
    normalize : L2-normalise the patches before taking any inner product.

    ``normalize`` matters and is not a detail: the checkpoint holds RAW DINOv3
    features whose norms run 12..118, while every distance at test time is
    computed on the L2-normalised rows (the GPU bank cache normalises them).
    Feeding raw rows here makes ``A @ B`` a raw inner product with magnitudes up
    to 1e4, so ``2 - 2*top`` is strongly negative, ``clamp_min(0)`` zeroes the
    whole tensor, every row is judged invalid and ``rk_med`` silently falls back
    to its ``nan_to_num`` default of EXACTLY 1.0000 -- a statistic that looks
    plausible in the log and is pure noise.  It is the same lesson as the
    absolute-vs-cosine reference point: an unnormalised reference breaks the
    metric.  Set normalize=False only if the caller really passes unit rows.
    row_sel : [N] long, the class's rows inside ``rows`` (default: all of them)
    ctx  : build_cand_context(...) for this (H, W, r).

    The neighbour pool of bank patch (m, p) is the SAME position-window pool
    used at test time, MINUS every patch of image m itself.  At test time the
    query image is absent from the bank, so keeping image m on the bank side
    would compare "distance to 149 other images" against "distance to myself" --
    the same scope error that made ``img_cal_scope='batch'`` wrong.

    ``refn`` caps the number of ANCHOR images that may act as CANDIDATES (0 =
    all).  Every class row always gets a statistic -- the base axis is never
    subsampled -- but the neighbour pool becomes a random subset when the class
    bank is large, which is what keeps the O(N * refn) cost of the VisA classes
    (~740 rows each) bounded.  The statistic is a scale, so a subsampled pool is
    statistically sufficient; the default stays exact because the exact version
    is what the paper should report.

    Returns rk [N, L], lid [N, L] float32 on CPU, and nmiss [N] int64.
    """
    N = int(row_sel.numel()) if row_sel is not None else int(rows.shape[0])
    L, C = int(rows.shape[1]), int(rows.shape[2])
    Kp = int(ctx['Kp'])
    idx, valid = ctx['idx'], ctx['valid']
    ridx = (torch.as_tensor(row_sel, dtype=torch.long).reshape(-1)
            if row_sel is not None else torch.arange(int(rows.shape[0])))
    n_anchor = N if (int(refn) <= 0 or int(refn) >= N) else int(refn)
    if n_anchor >= N:
        anchor = torch.arange(N, dtype=torch.long)
    else:
        g = torch.Generator().manual_seed(int(seed))
        anchor = torch.randperm(N, generator=g)[:n_anchor].sort().values
    apos = torch.arange(n_anchor, dtype=torch.long)

    dev = torch.device(device)
    rk = torch.zeros(N, L, dtype=torch.float32)
    lid = torch.zeros(N, L, dtype=torch.float32)
    nmiss = torch.zeros(N, dtype=torch.int64)
    n_valid_anchor = int(n_anchor)

    for s0 in range(0, L, chunk):
        s1 = min(s0 + chunk, L)
        Pc = s1 - s0
        pl = torch.arange(s0, s1, dtype=torch.long)
        # only the columns this chunk can possibly touch are read off the bank,
        # and only the rows of the class currently being evaluated
        need = torch.unique(torch.cat([pl, idx[s0:s1].reshape(-1).cpu()]))
        loc = torch.full((L,), -1, dtype=torch.long)
        loc[need] = torch.arange(need.numel(), dtype=torch.long)
        sub = rows[ridx.view(-1, 1), need.view(1, -1)].to(dev).to(torch.float32)
        if normalize:
            # the checkpoint stores raw features (norms 12..118); every distance
            # in this module is a chord distance on the unit sphere
            sub = F.normalize(sub, dim=-1, eps=eps)
        A = sub[:, loc[pl].to(dev)]                                 # [N, Pc, C]
        B = sub[:, loc[idx[s0:s1].reshape(-1).cpu()].to(dev)]
        B = B.reshape(N, Pc, Kp, C)
        if n_anchor < N:
            B = B.index_select(0, anchor.to(dev))
        B = B.contiguous()                                          # [n_anchor, Pc, Kp, C]
        Sa = torch.einsum('npc,mpkc->nmpk', A, B)                   # [N, n_anchor, Pc, Kp]
        # Sa[n, j, p, kk] = <bank[n, p], bank[anchor[j], idx[p, kk]]>
        #   axis 0  = the BASE image n (whose statistic we want)
        #   axis 1  = the CANDIDATE neighbour (anchor index j)
        # so the reduction must run over the candidate axes (1, 3) and keep the
        # base axis.  Reducing over (0, 3) instead -- the mirror image, which is
        # what a first version of this function did -- yields a different,
        # asymmetric statistic, and test_dad.py catches it.
        Sa = Sa.masked_fill(~valid[s0:s1].to(dev).view(1, 1, Pc, Kp), -2.0)
        Sa[anchor.to(dev), apos.to(dev)] = -2.0                     # leave-one-image-out
        S6 = Sa.permute(2, 0, 1, 3).reshape(
            Pc, N, n_valid_anchor * Kp)                             # [(j, kk) flattened]
        top, _ = S6.topk(min(int(k), S6.shape[-1]), dim=-1)         # [Pc, N, k]
        top = top.permute(1, 0, 2)                                  # [N, Pc, k]
        rk[:, s0:s1] = (2.0 - 2.0 * top[..., -1]).clamp_min(0.0).sqrt()
        lid[:, s0:s1] = _lid_from_sorted_sim(top, metric=metric)
        nmiss += (top[..., -1] <= -2.0).sum(dim=1).to(torch.int64).cpu()
    return rk, lid, nmiss


@torch.no_grad()
def window_cfar_score(q_norm, bank, ctx, rk_ref, k=8, comb='logp', chunk=128,
                      eps=1e-12, stats=None):
    """Distribution-free POSITION-WISE calibration of the k-NN non-conformity.

    The k-NN distance is not comparable across positions.  Two patches can have
    the same distance to the bank and be wildly different evidence, because the
    normal population itself is spread out very differently from place to place:
    on a self-similar texture the reference patches at some positions sit in a
    tight cluster and at others in a loose one, and the loose ones push
    perfectly NORMAL patches into the upper tail of the pooled ranking.  That is
    the measured failure this whole block exists for (INNOVATION section 1,
    diagnosis C: VisA pcb2/pcb3 have pixel-AUROC 0.99 with pixel-AP 0.26).

    ``bank_self_knn_stats`` already gives the null sample for every position p:
    the k-NN distances ``rk_ref[:, p]`` of the CLASS's own normal patches at that
    same position, computed leave-one-image-out.  The query is scored by its
    empirical p-value inside that sample:

        pv(q) = (1 + #{n : rk_ref[n, p] >= kdist(q)}) / (N_class + 1)
        score = -log pv                                   (comb='logp')
                1 - pv                                    (comb='neg')

    which is the standard finite-sample p-value of a rank test (the +1 keeps it
    away from 0, so the score stays finite).  Three properties, all of them
    constructive rather than tuned:

      * the calibration sample is drawn from NORMAL BANK patches only, never
        from the test image, so an image that is anomalous everywhere has a
        small p-value everywhere -- the absolute evidence survives.  Every
        per-image normalisation this project tried (u_c1zf -0.237, img_cal
        -0.0493) destroyed exactly that;
      * it is invariant to any monotone transform of the score at a given
        position, so a position whose normal spread happens to be large stops
        dominating the pooled ranking.  This is the classic CFAR argument of
        radar detection, where the threshold must track the local clutter
        level, and the position-wise normal bank is the clutter estimate;
      * the score is bounded by log(N_class + 1), so the tail cannot blow up
        the way an exponentiated density ratio does.

    ``comb='neg'`` is only a display variant (identical ranking); it exists so
    ``mean_normal_score`` is interpretable.
    """
    Kr, L, C = bank.shape
    Kp = int(ctx['Kp'])
    dev = q_norm.device
    rk_ref = rk_ref.to(dev).to(q_norm.dtype)
    Nref = int(rk_ref.shape[0])
    out = torch.empty(L, device=dev, dtype=q_norm.dtype)
    kk_use = min(int(k), Kr * Kp)
    for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
        P = s1 - s0
        sim = torch.bmm(cand, q_norm[s0:s1].unsqueeze(-1)).squeeze(-1)
        if ok is not None:
            sim = sim.masked_fill(~ok, -2.0)
        top, _ = sim.topk(kk_use, dim=-1)
        rq = (2.0 - 2.0 * top[:, -1]).clamp_min(0.0).sqrt()       # [P]
        col = rk_ref[:, s0:s1]                                    # [Nref, P]
        cnt = (col >= rq.view(1, -1)).sum(dim=0).to(q_norm.dtype)  # [P]
        pv = (1.0 + cnt) / (Nref + 1.0)
        if comb == 'neg':
            out[s0:s1] = 1.0 - pv
        else:
            out[s0:s1] = -torch.log(pv.clamp_min(eps))
        if stats is not None:
            stats['n'] = stats.get('n', 0) + P
    return out


@torch.no_grad()
def window_dad_score(q_norm, bank, rows, ctx, rk_ref, lid_ref, k=8,
                     lid_mode='est', comb='dao', metric='l2', chunk=128,
                     rk_fallback=None, lid_fallback=1.0, stats=None, eps=1e-12,
                     row_base=0, rk_med=None, dad_floor=0.0):
    """DAO / SLOF non-conformity over the same candidate pool as RAD.

    q_norm  : [L, C]      query patch features (L2-normalised)
    bank    : [Kr, L, C]  the retrieved reference patches
    rows    : [Kr]        their row indices into the FULL bank, which is what
                          ``rk_ref`` / ``lid_ref`` are indexed by
    rk_ref  : [N, L]      ``kdist(o)``   from bank_self_knn_stats
    lid_ref : [N, L]      ``LID(o)``     from bank_self_knn_stats

    ``row_base`` is the global bank row index of ``rk_ref[0]``.  The checkpoint
    holds one bank covering every class, so ``rows`` carries GLOBAL indices
    while the bank-side tables were built from one contiguous class block; the
    two only coincide when the class is the whole bank.  Retrievals that land
    outside the class block are counted as misses and fall back, rather than
    silently reading another class's row.

        comb='r'     score = kdist(q)                     -- pure k-NN control
        comb='med'   score = kdist(q) / med_p             -- ② the class's
                      per-position MEDIAN kdist as the denominator
        comb='slof'  score = mean_o (ratio**1)            -- SLOF, exponent 1
        comb='dao'   score = mean_o (ratio**LID(o))       -- DAO, the paper
        comb='mul'   score = kdist(q) * DAO_k(q)

    ``rk_med`` [L] is the class's per-position median bank-side kdist, and
    ``dad_floor`` floors the per-neighbour denominator at ``dad_floor * rk_med``.

    Why ``comb='med'`` exists.  DAO raises ``kdist(q)/kdist(o)`` to the power
    ``LID(o)``.  ``LID(o)`` is ~20 on the VisA bank, and ``kdist(o)`` is a
    ONE-SAMPLE estimate of the local scale of the position, so where a position
    is nearly degenerate it collapses towards 0 and the ratio explodes: the first
    real-data run produced mean_normal_score 6.4e12 and P-AP 0.19 against RAD's
    0.77.  ``rk_med`` is the SAME quantity -- the typical k-th neighbour distance
    at that position of that class -- estimated from every bank patch at that
    position instead of from one, i.e. with ~N times less variance, and it is
    what the diagnosis (P-AUROC 0.99 / P-AP 0.26 on VisA pcb2) actually asks
    for: dividing each pixel by the normal scale of ITS OWN position.  ``comb='r'``
    then isolates the numerator and ``comb='med'`` the ratio.

    ``rk_fallback`` [L] is used when a retrieved row has no bank-side statistic;
    the number of such hits is accumulated into ``stats['miss']`` so a silently
    degraded row cannot pass as a clean one.
    """
    Kr, L, C = bank.shape
    Kp = int(ctx['Kp'])
    idx = ctx['idx']
    dev = q_norm.device
    rows_t = torch.as_tensor(rows, device=dev).long().reshape(-1)
    rk_ref = rk_ref.to(dev).to(q_norm.dtype)
    lid_ref = lid_ref.to(dev).to(q_norm.dtype)
    Nref = int(rk_ref.shape[0])
    fb = None
    if rk_fallback is not None:
        fb = rk_fallback.to(dev).to(q_norm.dtype).reshape(-1)
    out = torch.empty(L, device=dev, dtype=q_norm.dtype)
    kk_use = min(int(k), Kr * Kp)

    for s0, s1, cand, ok in collect_candidates(bank, ctx, chunk=chunk):
        P = s1 - s0
        sim = torch.bmm(cand, q_norm[s0:s1].unsqueeze(-1)).squeeze(-1)
        if ok is not None:
            sim = sim.masked_fill(~ok, -2.0)
        top, topi = sim.topk(kk_use, dim=-1)                    # descending sim
        if stats is not None:
            stats['n'] = stats.get('n', 0) + P * kk_use
        m = topi // Kp
        kk = topi % Kp
        pos = idx[s0:s1].gather(1, kk)                          # [P, k] bank position
        grow = rows_t[m]                                        # [P, k] GLOBAL bank row
        loc = grow - int(row_base)
        outside = (loc < 0) | (loc >= Nref)
        loc = loc.clamp(0, max(Nref - 1, 0))
        rk_o = rk_ref[loc, pos]                                 # [P, k]
        lid_o = lid_ref[loc, pos]
        zero = rk_o <= 0
        bad = outside | zero
        nbad = int(bad.sum())
        if nbad:
            if stats is not None:
                # kept apart on purpose: `out` means the class-block
                # reconstruction is wrong (every number would be garbage),
                # `zero` means the bank-side table itself is degenerate.
                stats['miss'] = stats.get('miss', 0) + nbad
                stats['miss_out'] = stats.get('miss_out', 0) + int(outside.sum())
                stats['miss_zero'] = stats.get('miss_zero', 0) + int(zero.sum())
            if fb is not None:
                rk_o = torch.where(bad, fb[s0:s1].unsqueeze(1), rk_o)
        rq_k = (2.0 - 2.0 * top[:, -1]).clamp_min(0.0).sqrt()   # [P] k-th NN dist
        if metric == 'cos':
            rq_k = 0.5 * rq_k.square()
            rk_o = 0.5 * rk_o.square()
        med_p = None
        if rk_med is not None:
            med_p = rk_med.to(dev).to(q_norm.dtype).reshape(-1)[s0:s1]
            if metric == 'cos':
                med_p = 0.5 * med_p.square()
        # --- floor the per-neighbour denominator --------------------------
        # kdist(o) of a SINGLE neighbour is a one-sample estimate of the local
        # normal scale.  Where a position is nearly degenerate -- every normal
        # patch of the class agrees there -- it collapses towards 0 and
        # ratio**LID explodes: the first real-data run reached
        # mean_normal_score 6.4e12 with LID ~ 20.  The class's per-POSITION
        # median kdist is the same quantity estimated from every bank patch at
        # that position instead of from one, so flooring at it costs nothing
        # statistically and bounds the ratio.  dad_floor=0 keeps the strict
        # paper formula.
        if med_p is not None and dad_floor > 0.0:
            rk_o = torch.maximum(rk_o, float(dad_floor) * med_p.unsqueeze(1))
        ratio = (rq_k.unsqueeze(1) / rk_o.clamp_min(eps)).clamp_min(eps)
        if comb == 'r':                       # pure k-NN control, no ratio
            out[s0:s1] = rq_k
        elif comb == 'med':                   # query distance / class scale
            if med_p is None:
                raise ValueError("comb='med' needs rk_med (the per-position "
                                 "class median kdist)")
            out[s0:s1] = rq_k / med_p.clamp_min(eps)
        else:
            e = (torch.where(lid_o > 0.0, lid_o,
                             torch.full_like(lid_o, float(lid_fallback)))
                 if (lid_mode == 'est' and comb != 'slof')
                 else torch.ones_like(ratio))
            # log-space mean: the paper's statistic is an ARITHMETIC mean of
            # (ratio)**LID, and computing it as exp(mean(log ...)) would
            # silently turn it into a geometric mean.  logsumexp keeps the
            # arithmetic mean exact while staying overflow-safe.
            lg = (e * torch.log(ratio)).clamp(-40.0, 40.0)
            dao = torch.exp(torch.logsumexp(lg, dim=1)
                            - math.log(float(lg.shape[1])))
            out[s0:s1] = dao * rq_k if comb == 'mul' else dao
    return out


def grid_band_pass(X, h, w, mode='swt', band_w=(1.0, 1.0, 1.0, 1.0), beta=0.0):
    """
    Frequency-domain filtering of a patch-GRID tensor.

    X    : [..., C, h, w]   (the two trailing axes are the patch grid)
    h, w : the grid the score map was built on.  RAD's patch grid is 28 x 28 at
           448 px input, and the evaluator upsamples it to 256 before scoring,
           so the caller must pass the SAME grid, not the mask resolution.

    mode 'off' : no-op.
    mode 'hp'  : residual high-pass   X - beta * blur3(X).   beta = 0 -> no-op.
    mode 'swt' : one-level UNDECIMATED ("a trous") Haar decomposition into four
                 subbands that EACH remain at h x w, recombined as

                     out = 0.5 * (wa*A + wh*H + wv*V + wd*D)

                 with band_w = (wa, wh, wv, wd).  Because A + H + V + D == 2*x00,

                     (1, 1, 1, 1)  ->  x00            == the input  (identity)
                     (1, 0, 0, 0) ->  0.5*A          == 2x2 box LOW-PASS, at the
                                                       same level as the input
                     (0, 1, 1, 1) ->  0.5*(H+V+D)    == x00 - A/2, i.e. a PURE
                                                       high-pass (the map minus
                                                       its own 2x2 local mean)
                     (t, 1, 1, 1) ->  x00 - 0.5*((1-t)/1)*A   a one-parameter
                                       family interpolating identity (t=1) to
                                       pure high-pass (t=0)

    NEUTRAL SETTINGS RETURN THE INPUT OBJECT ITSELF, so a caller configured with
    baseline flags is bit-identical to a caller that never invokes this at all.
    test_dad.py asserts that with `is`, not with a tolerance.

    That short-circuit is LOAD-BEARING, not an optimisation.  A + H + V + D is a
    cancelling sum: the four terms are each of the order of the map itself while
    their sum is 2*x00, so on entries where x00 is near zero the relative error
    of the algebraic identity reaches ~1e-4 in float32 (measured, test_dad.py
    section 11).  Returning `X` directly is what makes (1,1,1,1) exact rather
    than merely close, and "merely close" would be enough to make every u1/u3b
    row a cross-code-version comparison instead of a baseline.

    Why UNDECIMATED.  A decimated 2D DWT on RAD's 28 x 28 grid produces 14 x 14
    subbands, while pixel-AP / AUPRO are scored at 256 x 256 after bilinear
    interpolation.  A 14 x 14 band cannot carry the localisation, so any gain it
    appeared to show would be an artefact of the upsampling rather than of the
    filtering -- and localisation is exactly the metric this project has to
    move.  The a-trous form samples every subband at EVERY grid position, so the
    band weights act on frequency content alone and never on resolution.

    Why REFLECT padding.  The patch grid carries no periodic boundary semantics,
    so circular shifts would fold one row and one column onto each other and
    inject spurious high-frequency energy exactly at the border.  Reflect
    padding keeps the operator local and shift-stable there.
    """
    if mode == 'off':
        return X
    h = int(h)
    w = int(w)
    if h < 2 or w < 2:
        raise ValueError(f'grid_band_pass needs h,w >= 2 (got {h},{w})')
    if tuple(X.shape[-2:]) != (h, w):
        raise ValueError(f'grid_band_pass: trailing shape {tuple(X.shape[-2:])} '
                         f'is not the grid (h,w)=({h},{w})')

    if mode == 'hp':
        beta = float(beta)
        if beta == 0.0:
            return X
        Yp = F.pad(X, (1, 1, 1, 1), mode='reflect')
        acc = None
        for dy in range(3):
            for dx in range(3):
                sl = Yp[..., dy:dy + h, dx:dx + w]
                acc = sl if acc is None else acc + sl
        return X - beta * (acc / 9.0)

    if mode == 'swt':
        bw = tuple(float(v) for v in band_w)
        if len(bw) != 4:
            raise ValueError('grid_band_pass: band_w needs four entries '
                             '(a, h, v, d)')
        if bw == (1.0, 1.0, 1.0, 1.0):
            return X
        Yp = F.pad(X, (0, 1, 0, 1), mode='reflect')
        x00 = Yp[..., 0:h,     0:w]
        x01 = Yp[..., 0:h,     1:w + 1]
        x10 = Yp[..., 1:h + 1, 0:w]
        x11 = Yp[..., 1:h + 1, 1:w + 1]
        A = (x00 + x01 + x10 + x11) * 0.5
        Hh = (x00 - x01 + x10 - x11) * 0.5
        Vv = (x00 + x01 - x10 - x11) * 0.5
        Dd = (x00 - x01 - x10 + x11) * 0.5
        return (bw[0] * A + bw[1] * Hh + bw[2] * Vv + bw[3] * Dd) * 0.5

    raise ValueError(f'grid_band_pass: unknown mode {mode!r}')
