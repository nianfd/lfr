"""Per-class decomposition of the radius ablation requested by the reviewer.

The aggregate pairing says the finite radius moves MVTec-AD P-AP by -0.01 pp
(beta = 0.9 vs R = inf).  A category mean can conceal a real effect on a few
categories, so the aggregate alone does not answer the reviewer's question.
This script reports, for every class and every metric,

    span     R = inf   - RAD          (the local affine span)
    radius   b = 0.9   - R = inf      (the finite coordinate radius, at the
                                       paper's default trust level)
    tighten  b = 0.5   - b = 0.9      (a genuinely tighter radius)

with the sign convention "positive = the later arm is better", and then the
mechanism question actually being asked -- "does the finite radius buy
accuracy?" -- as the ACROSS-CLASS rank correlation between the radius effect
and the fraction of (patch, layer) coordinates at which the constraint
actually binds (``clip_frac``, measured on the very same run).

Why Spearman and not only Pearson: with 15 classes and a heavy right tail on
``clip_frac`` (bottle 5.17% at beta = 0.5 against ~0.05% elsewhere) a product
correlation is dominated by one or two points and its magnitude is not
trustworthy, while its sign still is.  Both are printed; the conclusion is
drawn from the rank version.

clip_frac is read from the run's own ``per_class`` entry, so the rate and the
score change come from the same evaluation, not from two different runs.

usage:  python exp/rd_radius_perclass.py            (MVTec-AD)

The radius axis is MVTec-AD only.  Its VisA arms (rd_b50_v, rd_inf_v) were
cancelled on 2026-10-07 with no metrics.json written, so RD_DS=visa now exits
with [CANCELLED] rather than a MISSING listing -- see REVIEWER_RESPONSE.md
section 2.4.1.  The span comparison, which carries the accuracy claim, is still
reported for VisA in the main table; it is this per-class mechanism analysis
that is not replicated there.
"""
import json
import os
import sys

RUNS_ROOT = os.environ.get('RAD_SAVED', './saved_curvot')
DS = os.environ.get('RD_DS', 'mvtec')

# Cancelled by decision, not left unfinished.  Reported as such so that nobody
# goes looking for a run that is never going to appear.
CANCELLED = {
    'visa': 'radius axis is MVTec-AD only (scope decision 2026-10-07)',
}

ARMS_BY_DS = {                           # role -> run name, per dataset
    # The v2 arms (knn_chunk_imgs = 128, --lonm_clipstat 1) are the same runs
    # the beta sweep uses, so the radius and the beta evidence come from one
    # zero-exemption family.  They reproduce the pre-v2 arms to 4 dp on all
    # seven metrics (five of them bit-exact), which is what retires the
    # knn_chunk_imgs exemption from "known inert" to "measured inert here".
    'mvtec': dict(RAD='mf_rad', INF='rd_infc_m', B90='rd_b90c_m',
                  B50='rd_b50c_m'),
    'visa': dict(RAD='v3_rad', INF='rd_inf_v', B90='v3_rawl4', B50='rd_b50_v'),
}
ARMS = ARMS_BY_DS[DS]

MET = [('I-AUROC', 'auroc_sp'), ('I-AP', 'ap_sp'), ('I-F1', 'f1_sp'),
       ('P-AUROC', 'auroc_px'), ('P-AP', 'ap_px'), ('P-F1', 'f1_px'),
       ('AUPRO', 'aupro_px')]

PAIRS = [('span', 'RAD', 'INF'), ('radius', 'B90', 'INF'),
         ('tighten', 'B50', 'B90')]


def load(name):
    p = os.path.join(RUNS_ROOT, name, 'metrics.json')
    if not os.path.isfile(p):
        raise SystemExit(f'missing run: {name} ({p})')
    with open(p) as f:
        d = json.load(f)
    return dict(args=d.get('args') or {}, mean=d.get('mean') or {},
                per_class=d.get('per_class') or {})


def rank(v):
    """Average ranks, ascending, 1-based (ties share the mean rank)."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if sx == 0 or sy == 0:
        return float('nan')
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    return pearson(rank(x), rank(y))


def perm_pvalue(x, y, n_perm=100000, seed=12345):
    """Two-sided permutation p-value for the rank correlation.

    Computed here rather than looked up, so the threshold does not depend on a
    table transcribed from elsewhere.  Under the null the clip rates and the
    score changes are independent, so the ranks of y may be permuted freely
    while the ranks of x stay fixed; Spearman(x, y) equals Pearson on the two
    rank vectors, which is what is being recomputed each draw.
    """
    import random
    rng = random.Random(seed)
    rx = rank(x)
    ry = rank(y)
    obs = pearson(rx, ry)
    perm = list(ry)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(perm)
        if abs(pearson(rx, perm)) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main():
    if DS in CANCELLED:
        print(f'[CANCELLED] dataset {DS!r}: {CANCELLED[DS]}')
        print(f'[CANCELLED] this is a scope decision, not a failed or unfinished '
              f'run -- no metrics.json is expected for {DS} radius arms.')
        print(f'[CANCELLED] available: ' + ', '.join(
            d for d in ARMS_BY_DS if d not in CANCELLED))
        return 0
    R = {k: load(v) for k, v in ARMS.items()}
    cls = sorted(R['INF']['per_class'])
    for k, r in R.items():
        n = len(r['per_class'])
        if n != len(cls):
            raise SystemExit(f'{ARMS[k]}: {n} classes vs {len(cls)}')

    # --- per-class effect table, ordered by the radius clipping rate ---------
    cf = {c: R['INF']['per_class'][c].get('clip_frac') for c in cls}
    cf5 = {c: R['B50']['per_class'][c].get('clip_frac') for c in cls}
    cf = {c: v for c, v in cf.items() if v is not None}
    cf5 = {c: v for c, v in cf5.items() if v is not None}
    if not cf or not cf5:
        raise SystemExit('clip_frac absent: run with --lonm_clipstat 1')

    print(f'# radius decomposition -- {DS}  ({len(cls)} classes)')
    print(f'#   RAD={ARMS["RAD"]}  R=inf={ARMS["INF"]}  '
          f'b=0.9={ARMS["B90"]}  b=0.5={ARMS["B50"]}')
    print()
    print('| class | clip% b0.9 | clip% b0.5 | d_clip | d P-AP radius | '
          'd P-AP tighten | d P-AP span |')
    print('|---|---|---|---|---|---|---|')
    for c in sorted(cls, key=lambda c: -cf[c]):
        dr = 100 * (R['B90']['per_class'][c]['ap_px']
                    - R['INF']['per_class'][c]['ap_px'])
        dt = 100 * (R['B50']['per_class'][c]['ap_px']
                    - R['B90']['per_class'][c]['ap_px'])
        dsp = 100 * (R['INF']['per_class'][c]['ap_px']
                     - R['RAD']['per_class'][c]['ap_px'])
        print(f'| {c} | {100 * cf[c]:.4f} | {100 * cf5[c]:.4f} | '
              f'{100 * (cf5[c] - cf[c]):+.4f} | {dr:+.3f} | {dt:+.3f} | '
              f'{dsp:+.3f} |')

    # --- mechanism check: does clipping buy accuracy across classes? ---------
    print()
    print('  across-class correlation of the clip rate with the score change')
    print('  (n = %d; Spearman is the trustworthy one, see module docstring)'
          % len(cls))
    for pname, lo, hi in PAIRS:
        rate = cf if pname == 'radius' else (
            {c: cf5[c] - cf[c] for c in cls} if pname == 'tighten' else None)
        if rate is None:
            continue
        for lab, kk in MET:
            d = [100 * (R[hi]['per_class'][c][kk] - R[lo]['per_class'][c][kk])
                 for c in cls]
            x = [rate[c] for c in cls]
            sp, pe = spearman(x, d), pearson(x, d)
            p = perm_pvalue(x, d)
            tag = ('n.s.' if p > 0.05 else 'SIGNIFICANT')
            print(f'    {pname:8s} {lab:8s} Spearman={sp:+.3f}  '
                  f'Pearson={pe:+.3f}  p_perm={p:.4f} ({tag})')
    print('  The clip effect on the score is bounded by the per-class ranges')
    print('  below; a rank correlation that is not distinguishable from zero,')
    print('  or whose direction does not survive increasing the clip rate,')
    print('  cannot be read as an accuracy contribution from the radius.')
    print()

    # --- how large is the biggest single-class radius effect? ---------------
    for pname, lo, hi in PAIRS:
        worst = []
        for lab, kk in MET:
            d = [(100 * (R[hi]['per_class'][c][kk] - R[lo]['per_class'][c][kk]), c)
                 for c in cls]
            mn = min(d)
            mx = max(d)
            worst.append((lab, mn, mx))
        print(f'  {pname}: per-class range over the 7 metrics '
              f'(most negative / most positive)')
        for lab, mn, mx in worst:
            print(f'    {lab:8s} min {mn[0]:+.3f} ({mn[1]})   '
                  f'max {mx[0]:+.3f} ({mx[1]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
