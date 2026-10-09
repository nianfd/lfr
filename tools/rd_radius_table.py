"""Radius-ablation table for the reviewer's second request.

Question: does the finite coordinate radius R (beta = 0.9) buy accuracy, or is
it what makes the reference bounded (hence the sensitivity bound finite) while
accuracy comes from the locally fitted span itself?

Axis (MVTec-AD ONLY -- the VisA arms of this axis were cancelled on 2026-10-07,
see REVIEWER_RESPONSE.md section 2.4.1; everything else fixed -- frozen DINOv3
features, full normal bank, native k_image / pos_radius, layer_fuse=mean,
img_cal=none, batch size 8, the historical gather window):

    RAD            use_lonm 0                       pure 1-NN cosine, no span
    LFR R=inf      lonm_unbounded 1                 fitted span, alpha == 1
    LFR beta=0.9   lonm_trust 0.9                   the paper's default
    LFR beta=0.5   lonm_trust 0.5                   tight radius

so the three paired differences decompose the axis:

    span        = (R=inf)      - (RAD)        what the fitted span adds
    radius      = (beta 0.9)   - (R=inf)      what bounding it adds
    tighten     = (beta 0.5)   - (beta 0.9)   what tightening it adds

Runs are enumerated BY CONTENT (each row's own ``args``) rather than by file
name, every stored per-class mean is re-checked against its class average, and
every delta is formed from the raw values and rounded only afterwards.
``--lonm_clipstat`` is proven numerically inert by ``test_unbounded_equiv.py``,
so its presence on the radius arms and absence on the anchors does not make the
pairing a two-knob comparison; it is nonetheless the only field allowed to
differ besides the intended knob.

Run on the server:  python exp/rd_radius_table.py
"""
import itertools
import json
import os
import sys

RUNS_ROOT = os.environ.get('RAD_SAVED', './saved_curvot')

MV = ('carpet grid leather tile wood bottle cable capsule hazelnut '
      'metal_nut pill screw toothbrush transistor zipper').split()
VI = ('candle capsules cashew chewinggum fryum macaroni1 macaroni2 '
      'pcb1 pcb2 pcb3 pcb4 pipe_fryum').split()

# dataset -> pipeline fields that must be identical in every arm
BASE = {
    'mvtec': dict(k_image=150, pos_radius=1, layer_fuse='mean',
                  img_cal='none', batch_size=8, use_aptr=0, use_centroid=0,
                  use_transport_window=0, use_positional_bank=1, limit_imgs=0,
                  shot_k=0, knn_chunk_imgs=128, gpu_bank_cache_mb=9000,
                  normalize_bank=1, image_size=512, crop_size=448,
                  encoder_name='dinov3_vitb16'),
    'visa': dict(k_image=900, pos_radius=2, layer_fuse='mean',
                 img_cal='none', batch_size=8, use_aptr=0, use_centroid=0,
                 use_transport_window=0, use_positional_bank=1, limit_imgs=0,
                 shot_k=0, knn_chunk_imgs=32, gpu_bank_cache_mb=9000,
                 normalize_bank=1, image_size=512, crop_size=448,
                 encoder_name='dinov3_vitb16'),
}
CLASSES = {'mvtec': MV, 'visa': VI}
# the paper's LFR production readout, in full
LFR_PROD = dict(use_lonm=1, lonm_norm=1, lonm_norm_mode='raw', lonm_dmax=4,
                lonm_quadratic=0, lonm_M=24, lonm_energy=0.9, lonm_ridge=1e-4,
                lonm_ridge_rel=0.0, lonm_res_rank=0, lonm_curv_gain=1.0,
                lonm_radial=0, lonm_scale_beta=0.0, lonm_scale_q=0.9,
                lonm_maha=0, lonm_maha_rank=0)
# defaults for keys absent from older run records (schema evolution)
DEFAULT = dict(use_lonm=0, lonm_unbounded=0, lonm_clipstat=0, lonm_maha=0,
               lonm_maha_rank=0, lonm_fast=0, use_centroid=0, use_aptr=0,
               use_positional_bank=1, lonm_radial=0, lonm_res_rank=0,
               lonm_quadratic=1, lonm_norm=0, lonm_norm_mode='ratio',
               img_cal='none', layer_fuse='mean', agg=None)

ARM_SPEC = {
    ('mvtec', 'RAD'): dict(use_lonm=0),
    ('mvtec', 'LFR R=inf'): dict(LFR_PROD, lonm_unbounded=1),
    ('mvtec', 'LFR b=0.9'): dict(LFR_PROD, lonm_trust=0.9),
    ('mvtec', 'LFR b=0.5'): dict(LFR_PROD, lonm_trust=0.5),
    ('visa', 'RAD'): dict(use_lonm=0),
    ('visa', 'LFR R=inf'): dict(LFR_PROD, lonm_unbounded=1),
    ('visa', 'LFR b=0.9'): dict(LFR_PROD, lonm_trust=0.9),
    ('visa', 'LFR b=0.5'): dict(LFR_PROD, lonm_trust=0.5),
}
ORDER = ['RAD', 'LFR R=inf', 'LFR b=0.9', 'LFR b=0.5']

# Datasets whose radius arms were cancelled BY DECISION rather than left
# unfinished.  The distinction is worth code, not just a comment: an unfinished
# arm is MISSING and will appear later, a cancelled one never will, and reading
# one as the other sends the next person hunting for a failed run that does not
# exist.  The VisA arms (rd_b50_v, rd_inf_v) were stopped mid-flight on
# 2026-10-07 with no metrics.json written, so no completed result was discarded.
# See REVIEWER_RESPONSE.md section 2.4.1.
CANCELLED = {
    'visa': 'radius axis is MVTec-AD only (scope decision 2026-10-07; the span '
            'comparison, which carries the accuracy claim, is still reported '
            'for VisA in the main table)',
}

# The four runs of the axis, pinned BY NAME and then verified by content.
# Pinning is necessary, not lazy: `s12_dp_v` carries the SAME args as the
# VisA anchor `v3_rawl4` but was produced by a different `curvot_utils.py`
# (the S12 d_p patch), so args alone cannot distinguish code versions.  The
# content check below still runs on every pinned run, so a rename or a
# regenerated run with different settings is caught.
NAMED = {
    # The v2 arms (knn_chunk_imgs = 128, --lonm_clipstat 1) are preferred over
    # the pre-v2 ones: they reproduce them to 4 dp on all seven metrics while
    # removing the knn_chunk_imgs exemption, and rd_b90c_m carries the clip
    # diagnostic for the default percentile, which the anchor mf_raw4 predates.
    'mvtec': {'RAD': 'mf_rad', 'LFR R=inf': 'rd_infc_m',
              'LFR b=0.9': 'rd_b90c_m', 'LFR b=0.5': 'rd_b50c_m'},
    'visa': {'RAD': 'v3_rad', 'LFR R=inf': 'rd_inf_v',
             'LFR b=0.9': 'v3_rawl4', 'LFR b=0.5': 'rd_b50_v'},
}

MET = [('I-AUROC', 'auroc_sp'), ('I-AP', 'ap_sp'), ('I-F1', 'f1_sp'),
       ('P-AUROC', 'auroc_px'), ('P-AP', 'ap_px'), ('P-F1', 'f1_px'),
       ('AUPRO', 'aupro_px')]
# differences allowed between arms of the same axis
KNOBS = ('use_lonm', 'lonm_unbounded', 'lonm_trust', 'lonm_clipstat')
IGNORE = ('save_name', 'save_dir', 'item_list', 'bank_path', 'data_path',
          'encoder_weight')
# Fields that differ between the anchors and the radius arms and are excused
# ONLY because they were measured to be numerically inert, with the measurement
# named.  Anything else that differs aborts the table.
INERT_MEASURED = {
    'knn_chunk_imgs':
        'measured inert: 32 vs 128 gave identical values on all seven metrics '
        '(the option only chunks the kNN gather), and the MVTec arms are being '
        're-run at 128 to re-confirm it in this exact configuration',
    'lonm_clipstat':
        'test_unbounded_equiv.py: clip_stat=dict() is bit-identical to no '
        'clip_stat in all five cases',
}


def g(a, k):
    v = a.get(k)
    return DEFAULT.get(k) if v is None else v


def num_eq(x, y):
    try:
        return abs(float(x) - float(y)) < 1e-9
    except (TypeError, ValueError):
        return str(x) == str(y)


def matches(run, ds, spec):
    a = run['args']
    if list(run['items']) != CLASSES[ds]:
        return False
    for k, v in BASE[ds].items():
        if k in INERT_MEASURED:
            continue
        if not num_eq(g(a, k), v):
            return False
    for k, v in spec.items():
        if not num_eq(g(a, k), v):
            return False
    if 'lonm_trust' not in spec and not num_eq(g(a, 'lonm_trust'), 0.9):
        return False
    return True


def load_runs():
    out = {}
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f'no such run root: {RUNS_ROOT}')
    for name in sorted(os.listdir(RUNS_ROOT)):
        p = os.path.join(RUNS_ROOT, name, 'metrics.json')
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            d = json.load(f)
        a = d.get('args') or {}
        items = a.get('item_list')
        if isinstance(items, str):
            items = items.split()
        out[name] = dict(args=a, mean=d.get('mean') or {},
                         per_class=d.get('per_class') or {},
                         items=list(items or []))
    return out


def check_internal(run):
    bad = []
    for lab, k in MET:
        vals = [c[k] for c in run['per_class'].values() if k in c]
        m = sum(vals) / len(vals) if vals else None
        s = run['mean'].get(k)
        if m is None or s is None or abs(m - s) > 5e-4:
            bad.append(f'{lab}: class-avg {m} vs stored {s}')
    return bad


def inert_pair(a, b, f):
    """A differing field that provably cannot move either arm's score.

    ``curvot_rad.py`` gates the whole LONM readout behind ``if use_lonm:``, so
    when one side of the pair is the RAD arm (``use_lonm == 0``) every
    ``lonm_*`` option is unreachable for it and its value is irrelevant to the
    pairing.  ``lonm_clipstat`` and ``knn_chunk_imgs`` are excused on measured
    grounds, named in INERT_MEASURED and printed with the pairing.
    """
    if f in INERT_MEASURED:
        return True
    if f.startswith('img_cal_') and str(g(a, 'img_cal')) == 'none' \
            and str(g(b, 'img_cal')) == 'none':
        return True
    if f.startswith('lonm_') and (int(g(a, 'use_lonm')) == 0
                                  or int(g(b, 'use_lonm')) == 0):
        return True
    return False


def pick(runs, ds):
    """Resolve the four named arms and verify them BY CONTENT.

    Every arm must (a) exist, (b) satisfy its own argument spec, and (c) be
    mutually identical on every field except the intended knob and the
    numerically inert --lonm_clipstat.  Any other run that would also satisfy
    an arm's spec is reported, so an ambiguous alias is visible rather than
    silently resolved.
    """
    chosen, alias = {}, {}
    ok = True
    for key in ORDER:
        name = NAMED[ds][key]
        if name not in runs:
            print(f'  [MISSING] {ds:6s} {key:11s} (expected run {name!r})')
            ok = False
            continue
        hits = sorted(n for n, r in runs.items()
                      if matches(r, ds, ARM_SPEC[(ds, key)]))
        if name not in hits:
            print(f'  [SPEC VIOLATION] {ds:6s} {key:11s}: {name!r} does not '
                  f'satisfy its own argument spec')
            ok = False
            continue
        chosen[key] = name
        other = [h for h in hits if h != name]
        if other:
            alias[key] = other
    # (c) mutual identity -- run on whatever is available, so a mismatch shows
    # up even while some arms are still training.  Only fields present in EVERY
    # arm are auto-audited: the anchors predate several argparse options, and a
    # key that exists in only one arm carries no comparison.
    names = sorted(chosen.values())
    keys = (set.intersection(*[set(runs[n]['args'].keys()) for n in names])
            if names else set())
    keys |= set(BASE[ds].keys()) | set(LFR_PROD.keys())
    for fn, fk in itertools.combinations(names, 2):
        a, b = runs[fn]['args'], runs[fk]['args']
        extra, excused = [], []
        for f in sorted(keys):
            if f in KNOBS or f in IGNORE or num_eq(g(a, f), g(b, f)):
                continue
            (excused if inert_pair(a, b, f) else extra).append(f)
        if excused:
            print(f'  [excused] {ds:6s} {fn} vs {fk}: '
                  + ', '.join(f'{f}={g(a, f)}/{g(b, f)}' for f in excused))
        if extra:
            print(f'  [NON-IDENTICAL] {ds:6s} {fn} vs {fk}: '
                  f'{[(f, g(a, f), g(b, f)) for f in extra]}')
            ok = False
    if not ok or len(chosen) != len(ORDER):
        return None, alias
    return chosen, alias


def main():
    runs = load_runs()
    ds_list = os.environ.get('RD_DS', 'mvtec').split()
    print(f'# radius ablation -- {len(runs)} runs under {RUNS_ROOT} '
          f'(datasets: {" ".join(ds_list)})\n')
    sel, aborted = {}, False
    for ds in ds_list:
        if ds in CANCELLED:
            print(f'  [CANCELLED] {ds:6s}: {CANCELLED[ds]}')
            print(f'  [CANCELLED] {ds:6s}: not a failure and not pending -- no '
                  f'run is expected to appear for this dataset.')
            continue
        chosen, alias = pick(runs, ds)
        for k, other in sorted(alias.items()):
            print(f'  [alias] {ds:6s} {k:11s}: also matched by '
                  f'{other[:4]} (same args, not used)')
        if chosen is None:
            aborted = True
        else:
            sel.update({(ds, k): v for k, v in chosen.items()})
    print()
    if not sel and not aborted:
        print('nothing to tabulate: every requested dataset is cancelled.')
        return 0
    if aborted:
        print('ABORT: could not form a clean single-knob pairing.')
        return 2

    bad = False
    for (ds, k), n in sel.items():
        e = check_internal(runs[n])
        if e:
            bad = True
            print(f'  [MEAN MISMATCH] {n}: {e}')
    if bad:
        print('\nABORT: a stored mean does not equal its class average.')
        return 2

    for ds in ds_list:
        if not all((ds, k) in sel for k in ORDER):
            continue
        ncls = len(runs[sel[(ds, ORDER[0])]]['per_class'])
        print(f'===== {ds.upper()}  ({ncls} classes) =====')
        for k in ORDER:
            print(f'    {k:11s} -> {sel[(ds, k)]}')
        print()
        print('| Reference / readout | ' + ' | '.join(l for l, _ in MET)
              + ' | clip% |')
        print('|' + '---|' * (len(MET) + 2))
        for k in ORDER:
            r = runs[sel[(ds, k)]]
            cells = [f'{r["mean"][kk] * 100:.2f}' for _, kk in MET]
            cf = [c['clip_frac'] for c in r['per_class'].values()
                  if 'clip_frac' in c]
            if k == 'LFR R=inf':
                # alpha == 1 by construction, so no coordinate is ever clipped.
                # The run's own clip_stat still measures the fraction that
                # WOULD be clipped at its --lonm_trust (the default 0.9); that
                # is the beta = 0.9 rate, reported separately below instead of
                # being placed on this row.
                clip = '0 (by construction)'
            else:
                clip = f'{100 * sum(cf) / len(cf):.3f}' if cf else '--'
            print(f'| {k} | ' + ' | '.join(cells) + f' | {clip} |')
        print()
        for k in ORDER:
            r = runs[sel[(ds, k)]]
            with_clip = [c for c in r['per_class'].values()
                         if 'clip_frac' in c]
            if not with_clip:
                continue
            unweighted = 100 * sum(c['clip_frac'] for c in with_clip) / len(
                with_clip)
            n_tot = sum(c['clip_n'] for c in with_clip)
            weighted = 100 * sum(c['clip_frac'] * c['clip_n']
                                 for c in with_clip) / n_tot
            trust = r['args'].get('lonm_trust')
            print(f'  [would-clip @ trust {trust}] {k}: unweighted '
                  f'{unweighted:.4f}%  coordinate-weighted {weighted:.4f}%  '
                  f'({n_tot} query coords)')
        print()
        for lab, ka, kb in [('span    R=inf  - RAD  ', 'RAD', 'LFR R=inf'),
                            ('radius  b=0.9  - R=inf', 'LFR R=inf', 'LFR b=0.9'),
                            ('tighten b=0.5  - b=0.9', 'LFR b=0.9', 'LFR b=0.5')]:
            ra, rb = runs[sel[(ds, ka)]], runs[sel[(ds, kb)]]
            cls = sorted(ra['per_class'])
            dd, sg = [], []
            for _, kk in MET:
                d = [rb['per_class'][c][kk] - ra['per_class'][c][kk]
                     for c in cls]
                dd.append(f'{100 * sum(d) / len(d):+.2f}')
                p = sum(1 for x in d if x > 0)
                q = sum(1 for x in d if x < 0)
                sg.append(f'{p}/{len(d) - p - q}/{q}')
            print(f'  {lab}  Δ(pp)        ' + ' '.join(
                f'{l_}={c}' for (l_, _), c in zip(MET, dd)))
            print(f'  {" " * len(lab)}  win/tie/loss ' + ' '.join(
                f'{l_}={s}' for (l_, _), s in zip(MET, sg)))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
