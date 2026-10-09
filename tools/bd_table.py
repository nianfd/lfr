"""Matched-feature baseline table for the reviewer's first request.

Under identical frozen DINOv3 features, normal memory, retrieval pool,
anomaly-map recipe, top-1% image pooling and evaluator, how do the
training-free reference constructions of Table I score?

    RAD         retrieved exemplar / cosine        the paired rerun anchor
    LFR         locally fitted reference / PNR     the paper's default
    SubspaceAD  category-level PCA / squared orthogonal residual
    ANoCo       anchored bipartite Laplacian / drift

PaDiM and PatchCore are implemented in exp/baseline_scores.py and wired to the
same driver, but they are OUT OF SCOPE for the reported panel on both datasets:
the request named SubspaceAD and ANoCo, and those two are the constructions that
instantiate the paper's claim (a category-level linear reference versus a
retrieval-anchored graph reference).  They are reported as "not run here" rather
than [MISSING] so that [MISSING] keeps meaning "will appear later".

Dinomaly is deliberately absent: its reference is a trained decoder, so a
matched-feature treatment would require its training rather than a swap of the
reference.  Its published row stays in Table II as protocol context.

Table structure
---------------
Run names are pinned (an alias with identical args but a different
``curvot_utils.py`` cannot be told apart from the args alone) and then verified
BY CONTENT: every arm must satisfy the dataset's pipeline spec, and all arms
must agree on every recorded field except the intended one.  A differing field
is excused only if it provably cannot move the score:

* ``baseline*``  -- the knob;
* ``use_lonm``   -- the knob for the RAD anchor;
* ``lonm_*``     -- unreachable unless ``use_lonm == 1 and baseline == 'none'``,
                    and only one such row exists per dataset, so a lonm_* difference
                    is meaningful only if the check says so.

Deltas are formed from the raw per-class values and rounded only afterwards,
and every row is reported with its per-class win/tie/loss against LFR, because
a category-mean difference can be produced by two or three outlying categories.

Dataset scope (2026-10-07, rev 2)
---------------------------------
Both datasets: SubspaceAD and ANoCo only.  PaDiM and PatchCore are implemented
and their hooks are exercised, but they are not part of the reported panel --
the request named SubspaceAD and ANoCo, and a four-row panel fits section IV
where a wider one would not.

The reduced panel is a scope decision, NOT an unfinished run, and this file
prints it as "not run here" rather than [MISSING]; the rule is that [MISSING]
must keep meaning "will appear later".

Run on the server:  python exp/bd_table.py
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
LFR_PROD = dict(use_lonm=1, lonm_norm=1, lonm_norm_mode='raw', lonm_dmax=4,
                lonm_quadratic=0, lonm_M=24, lonm_energy=0.9, lonm_ridge=1e-4,
                lonm_ridge_rel=0.0, lonm_res_rank=0, lonm_curv_gain=1.0,
                lonm_radial=0, lonm_scale_beta=0.0, lonm_scale_q=0.9,
                lonm_maha=0, lonm_maha_rank=0)
DEFAULT = dict(use_lonm=0, lonm_unbounded=0, lonm_clipstat=0, lonm_maha=0,
               lonm_maha_rank=0, lonm_fast=0, use_centroid=0, use_aptr=0,
               use_positional_bank=1, lonm_radial=0, lonm_res_rank=0,
               lonm_quadratic=1, lonm_norm=0, lonm_norm_mode='ratio',
               img_cal='none', layer_fuse='mean', agg=None, baseline='none',
               baseline_d=64, baseline_power=2.0, baseline_pc_ratio=0.01,
               baseline_pc_k=9, baseline_pc_rw=1.0, baseline_pc_cap=2000,
               baseline_pc_proj=64, baseline_pc_subcap=120000,
               baseline_padim_shrink='published', baseline_padim_ridge_rel=0.0,
               baseline_padim_ridge_abs=0.01, baseline_padim_sqrt=1,
               baseline_anoco_lambda=1.0, baseline_anoco_anchor=1)

# ---------------------------------------------------------------------------
# Flags introduced after the anchors were recorded (mf_rad / mf_raw4, 09-27).
#
# The anchors' args dict simply has no key for these, while a run recorded today
# (bd_sub_m, bd_anc_m) carries all 105 keys.  Reading an absent key as "unset,
# therefore different" would report 24 [NON-IDENTICAL] fields on a comparison
# that is in fact identical -- but reading it as "equals today's default" is an
# assumption about the code, so it is not taken on trust either:
#
#   * every value below is dumped mechanically from curvot_rad.py's argparse
#     (exp/dump_defaults.py) rather than transcribed by hand;
#   * filling happens ONLY when the other side's recorded value equals that
#     default.  If the newer run had actually passed a non-default value for
#     one of these flags, the fill would not match and the field would still be
#     reported as [NON-IDENTICAL];
#   * exp/vchk_rad.sh re-runs the anchor configuration with today's code on two
#     MVTec-AD categories and compares per class, which is what turns "the
#     default is inert along this path" from a belief into a measurement.
#
# The audit block prints every field that was filled, with the value on the
# other side, so the assumption is visible in the table output rather than
# buried in this dict.
# ---------------------------------------------------------------------------
LATE_FLAGS = {
    'agg': 'min',
    'anc_beta': 0.0,
    'band_beta': 0.0,
    'band_mode': 'off',
    'band_w': [1.0, 1.0, 1.0, 1.0],
    'bank_item_order': None,
    'cfar_comb': 'logp',
    'cfar_post': 0.0,
    'dad_comb': 'dao',
    'dad_dir': './cache_dad',
    'dad_floor': 0.0,
    'dad_k': 8,
    'dad_lid': 'est',
    'dad_metric': 'l2',
    'dad_post': 0.0,
    'dad_refn': 0,
    'ia_K': 0,
    'img_cal_alpha': 1.0,
    'img_cal_red': 'median',
    'img_cal_red_q': 0.9,
    'img_cal_ref': 'none',
    'img_cal_scale_q': 0.95,
    'img_cal_scope': 'batch',
    'img_cal_stats': 0,
}
DEFAULT.update(LATE_FLAGS)

# arm -> (spec that the run's args must satisfy, expected run name per dataset)
ARMS = [
    ('RAD', dict(use_lonm=0), {'mvtec': 'mf_rad', 'visa': 'v3_rad'}),
    ('LFR', dict(LFR_PROD), {'mvtec': 'mf_raw4', 'visa': 'v3_rawl4'}),
    ('SubspaceAD', dict(baseline='subspacead'),
     {'mvtec': 'bd_sub_m', 'visa': 'bd_sub_v'}),
    ('ANoCo', dict(baseline='anoco'),
     {'mvtec': 'bd_anc_m', 'visa': 'bd_anc_v'}),
    ('PaDiM', dict(baseline='padim'),
     {'mvtec': 'bd_pad_m', 'visa': 'bd_pad_v'}),
    ('PatchCore', dict(baseline='patchcore'),
     {'mvtec': 'bd_pc_m', 'visa': 'bd_pc_v'}),
]
# sensitivity rows, reported separately, never as the method's entry
SENS = [
    ('PaDiM [Ledoit-Wolf]', dict(baseline='padim', baseline_padim_shrink='lw'),
     {'mvtec': 'bd_padlw_m', 'visa': 'bd_padlw_v'}),
    ('PaDiM [rel. ridge 1e-3]',
     dict(baseline='padim', baseline_padim_shrink='ridge',
          baseline_padim_ridge_abs=0.0, baseline_padim_ridge_rel=1e-3),
     {'mvtec': 'bd_padsr_m', 'visa': 'bd_padsr_v'}),
]
ORDER = [a[0] for a in ARMS]

# ---------------------------------------------------------------------------
# Per-dataset arm scope (2026-10-07, rev 2).  The request named SubspaceAD and
# ANoCo for the matched comparison, and the panel is those two on BOTH
# datasets.  PaDiM and PatchCore have working implementations and matching run
# hooks (see exp/bd_baselines.sh, BD_INCLUDE_FULL_PANEL=1) but are out of scope
# for the reported table.
#
# This is a scope decision, NOT an unfinished run, so the reduced panel must be
# reported as "not run here" rather than [MISSING]: MISSING reads as "will
# appear later" and sends the reader looking for a run that is never coming.
# See REVIEWER_RESPONSE.md section 3.6.
# ---------------------------------------------------------------------------
NOT_RUN = {
    ('mvtec', 'PaDiM'),
    ('mvtec', 'PatchCore'),
    ('mvtec', 'PaDiM [Ledoit-Wolf]'),
    ('mvtec', 'PaDiM [rel. ridge 1e-3]'),
    ('visa', 'PaDiM'),
    ('visa', 'PatchCore'),
    ('visa', 'PaDiM [Ledoit-Wolf]'),
    ('visa', 'PaDiM [rel. ridge 1e-3]'),
}
NOT_RUN_WHY = ('out of scope for this dataset: the matched panel is SubspaceAD '
               'and ANoCo only, the two constructions the request named')
MET = [('I-AUROC', 'auroc_sp'), ('I-AP', 'ap_sp'), ('I-F1', 'f1_sp'),
       ('P-AUROC', 'auroc_px'), ('P-AP', 'ap_px'), ('P-F1', 'f1_px'),
       ('AUPRO', 'aupro_px')]
IGNORE = ('save_name', 'save_dir', 'item_list', 'bank_path', 'data_path',
          'encoder_weight')


def g(a, k):
    v = a.get(k)
    return DEFAULT.get(k) if v is None else v


def num_eq(x, y):
    try:
        return abs(float(x) - float(y)) < 1e-9
    except (TypeError, ValueError):
        return str(x) == str(y)


def is_lfr(a):
    return int(g(a, 'use_lonm')) == 1 and str(g(a, 'baseline')) == 'none'


def inert_pair(a, b, f):
    if f.startswith('baseline'):
        return True
    if f == 'use_lonm':
        return True
    if f.startswith('lonm_'):
        return not (is_lfr(a) and is_lfr(b))
    return False


def matches(run, ds, spec):
    a = run['args']
    if list(run['items']) != CLASSES[ds]:
        return False
    for k, v in BASE[ds].items():
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


def check_mean(run):
    bad = []
    for lab, k in MET:
        vals = [c[k] for c in run['per_class'].values() if k in c]
        m = sum(vals) / len(vals) if vals else None
        s = run['mean'].get(k)
        if m is None or s is None or abs(m - s) > 5e-4:
            bad.append(f'{lab}: class-avg {m} vs stored {s}')
    return bad


def resolve(runs, ds, arms, required=True):
    resolved, missing, alias = [], [], {}
    for label, spec, names in arms:
        name = names[ds]
        if name not in runs:
            missing.append(f'{ds} {label} (expected {name!r})')
            continue
        hits = sorted(n for n, r in runs.items() if matches(r, ds, spec))
        if name not in hits:
            missing.append(f'{ds} {label}: {name!r} violates its spec')
            continue
        resolved.append((label, name))
        if [h for h in hits if h != name]:
            alias[label] = [h for h in hits if h != name][:4]
    return resolved, missing, alias


def identity_report(runs, ds, chosen, audit, unexplained):
    """Every arm must agree on every recorded field except the intended knob.

    A field whose key is absent on exactly one side was not compared, it was
    filled from the parser default.  That is allowed only for the late flags
    listed in LATE_FLAGS and only when the other side really carries that
    default (num_eq below); anything else is schema drift the table cannot
    justify, so it is reported rather than absorbed.
    """
    keys = set(BASE[ds].keys()) | set(LFR_PROD.keys())
    for _, n in chosen:
        keys |= set(runs[n]['args'].keys())
    bad = False
    for (ln, fn), (lm, fm) in itertools.combinations(chosen, 2):
        a, b = runs[fn]['args'], runs[fm]['args']
        extra = []
        for f in sorted(keys):
            if f in IGNORE or inert_pair(a, b, f):
                continue
            if num_eq(g(a, f), g(b, f)):
                if (f not in a) != (f not in b):
                    val = g(a, f) if f in a else g(b, f)
                    if f in LATE_FLAGS:
                        audit.setdefault(f, val)
                    else:
                        unexplained[f] = (ln, fn, val)
                        bad = True
                continue
            extra.append((f, g(a, f), g(b, f)))
        if extra:
            print(f'  [NON-IDENTICAL] {ds}: {ln}({fn}) vs {lm}({fm}) -> '
                  f'{extra}')
            bad = True
    return bad


def check_late_flags():
    """Cross-check LATE_FLAGS against curvot_rad.py's argparse, if dumped.

    LATE_FLAGS decides which fields may be filled from a parser default, so if
    it drifts from the parser (a default changed, a flag renamed) the fill would
    compare against the wrong value and the resulting "identical" claim would be
    wrong in a way nothing else in this file can see.  exp/dump_defaults.py
    writes the parser's own view to exp/args_defaults.json; when that file is
    present it is the authority and any mismatch fails the run.
    """
    p = 'exp/args_defaults.json'
    if not os.path.isfile(p):
        print(f'  [check skipped] {p} absent: the {len(LATE_FLAGS)} LATE_FLAGS '
              f'values are NOT verified against the parser.  Regenerate with '
              f'"python exp/dump_defaults.py".')
        return 0
    with open(p) as f:
        dumped = json.load(f)
    bad = 0
    for f_, v in sorted(LATE_FLAGS.items()):
        rec = dumped.get('--' + f_)
        if rec is None:
            print(f'  [LATE_FLAGS DRIFT] {f_}: not a flag of curvot_rad.py')
            bad = 1
        elif rec.get('required'):
            print(f'  [LATE_FLAGS DRIFT] {f_}: the parser marks it required, so '
                  f'there is no default to fill from')
            bad = 1
        elif not num_eq(v, rec.get('default')):
            print(f'  [LATE_FLAGS DRIFT] {f_}: LATE_FLAGS says {v!r}, the parser '
                  f'default is {rec.get("default")!r}')
            bad = 1
    if not bad:
        print(f'  [check] all {len(LATE_FLAGS)} late flags match the curvot_rad.py '
              f'defaults in {p}')
    return bad


def main():
    runs = load_runs()
    ds_list = os.environ.get('BD_DS', 'mvtec visa').split()
    print(f'# matched-feature baselines -- {len(runs)} runs under {RUNS_ROOT}'
          f'   (datasets: {" ".join(ds_list)})')
    bad = bool(check_late_flags())
    print()
    audit, unexplained = {}, {}
    for ds in ds_list:
        arms = [a for a in ARMS if (ds, a[0]) not in NOT_RUN]
        for a in ARMS:
            if (ds, a[0]) in NOT_RUN:
                print(f'  [not run here] {ds} {a[0]}: {NOT_RUN_WHY}')
        resolved, missing, alias = resolve(runs, ds, arms)
        for m in missing:
            print(f'  [MISSING] {m}')
        for k, v in sorted(alias.items()):
            print(f'  [alias] {ds} {k}: also matched by {v} (not used)')
        if missing or not resolved:
            print(f'  [{ds.upper()} INCOMPLETE] no table printed for this '
                  f'dataset\n')
            bad = True
            continue
        if identity_report(runs, ds, resolved, audit, unexplained):
            bad = True

        chosen = resolved
        names = [n for _, n in chosen]
        mismatch = False
        for n in names:
            e = check_mean(runs[n])
            if e:
                print(f'  [MEAN MISMATCH] {n}: {e}')
                mismatch = True
        if mismatch:
            print(f'  [{ds.upper()} INCOMPLETE] class means do not reproduce '
                  f'the stored means\n')
            bad = True
            continue

        ncls = len(runs[names[0]]['per_class'])
        print(f'===== {ds.upper()}  ({ncls} classes) =====')
        for lab, n in chosen:
            print(f'    {lab:12s} -> {n}')
        for lab, spec, names_ in SENS:
            n = names_[ds]
            if (ds, lab) in NOT_RUN:
                print(f'    {lab:22s} -> [not run here] {NOT_RUN_WHY}')
            elif n in runs:
                hits = [k for k, r in runs.items() if matches(r, ds, spec)]
                print(f'    {lab:22s} -> {n}'
                      + ('' if n in hits else '   [SPEC?]'))
            else:
                print(f'    {lab:22s} -> [MISSING] expected run {n!r}')
        print()

        lfr = runs[dict(chosen)['LFR']]
        print('| Method | ' + ' | '.join(l for l, _ in MET) + ' |')
        print('|' + '---|' * (len(MET) + 1))
        for lab, n in chosen:
            r = runs[n]
            cells = []
            for l_, kk in MET:
                v = f'{r["mean"][kk] * 100:.2f}'
                cells.append(f'**{v}**' if lab == 'LFR' else v)
            print(f'| {lab} | ' + ' | '.join(cells) + ' |')
        print()

        print(f'  Δ vs LFR (pp), and per-class win/tie/loss:')
        for lab, n in chosen:
            if lab == 'LFR':
                continue
            r = runs[n]
            cls = sorted(set(r['per_class']) & set(lfr['per_class']))
            dd, sg = [], []
            for _, kk in MET:
                d = [lfr['per_class'][c][kk] - r['per_class'][c][kk]
                     for c in cls]
                dd.append(f'{100 * sum(d) / len(d):+.2f}')
                p = sum(1 for x in d if x > 0)
                q = sum(1 for x in d if x < 0)
                sg.append(f'{p}/{len(d) - p - q}/{q}')
            print(f'    {lab:12s} Δ ' + ' '.join(
                f'{l_}={c}' for (l_, _), c in zip(MET, dd)))
            print(f'    {" " * 12}   ' + ' '.join(
                f'{l_}={s}' for (l_, _), s in zip(MET, sg)))
        print()

        for lab, spec, names_ in SENS:
            n = names_[ds]
            if n not in runs:
                continue
            r = runs[n]
            cells = [f'{r["mean"][kk] * 100:.2f}' for _, kk in MET]
            print(f'  {lab}: ' + ' '.join(
                f'{l_}={c}' for (l_, _), c in zip(MET, cells)))
        print()

    if unexplained:
        print('  [UNEXPLAINED GAP] field absent on exactly one side of a pair '
              'and not a known late flag:')
        for f, (ln, fn, val) in sorted(unexplained.items()):
            print(f'      {f}: {ln} ({fn}) does record the key (value {val!r}) '
                  f'while the other side has no key for it')
        print()
    if audit:
        print('  [filled] flag(s) absent from the 09-27 anchor args and carrying')
        print('           the parser default on the other side, so these pairs are')
        print('           equal BY CONSTRUCTION, not by measurement:')
        for f in sorted(audit):
            print(f'             {f} = {audit[f]!r}')
        print('  The construction is auditable two ways: a fill only happens when')
        print('  the recorded value equals curvot_rad.py\'s argparse default, so a')
        print('  flag the newer run actually set to a non-default value would still')
        print('  be reported [NON-IDENTICAL]; and exp/vchk_rad.sh re-runs the anchor')
        print('  configuration with today\'s code on two MVTec-AD categories, whose')
        print('  per-class agreement with mf_rad is what makes the fill a')
        print('  measurement rather than a belief.  Check exp/logs/bd_equiv.txt.')
        print()

    if bad:
        print('PARTIAL: at least one dataset is incomplete, or its arms are '
              'not a clean single-knob comparison.')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
