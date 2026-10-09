"""Synthetic self-test for exp/rd_beta_table.py.

The beta sweep costs ~100 minutes of GPU time, so the analyser is exercised on a
fixture whose answer is known before that time is spent.  The fixture is not a
guess at the real numbers: values are planted so that every branch of the
verdict is reached at least once.

Planted facts (MVTec-AD, 15 categories, values in pp):

  * beta = 0.9 attains the maximum on 5 of the 7 metrics;
  * P-AP at beta = 0.6 is 0.18 pp ABOVE beta = 0.9 -> must be reported as a real
    exceedance of the seed-noise floor, not rounded away (the honest-reporting
    branch);
  * P-F1 at beta = 0.7 is 0.07 pp above beta = 0.9 -> must be reported as
    "within noise" (the other branch of the same test);
  * P-AP at beta = 0.8 loses on exactly 10 categories and ties on 5 -> the exact
    sign test must print p = 0.002 (2 * C(10,0) / 2^10);
  * clipping activation must rise monotonically as beta falls.

Four negative / degraded controls, each in its own copy of the fixture:

  F1  pre-v2 arm names (knn 32, no clipstat) -> fallbacks accepted, [excused]
      printed, rc = 0
  N1  a differing non-knob field             -> [NON-IDENTICAL] and rc = 2
  N2  a stored mean != its class average     -> [MEAN MISMATCH] and rc = 2
  N3  one arm missing                        -> INCOMPLETE and rc = 2

Run:  python test_beta_table.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, 'rd_beta_table.py')

MV = ('carpet grid leather tile wood bottle cable capsule hazelnut '
      'metal_nut pill screw toothbrush transistor zipper').split()
BETAS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
BETA_RUN = {0.9: 'rd_b90c_m', 0.8: 'rd_b80_m', 0.7: 'rd_b70_m',
            0.6: 'rd_b60_m', 0.5: 'rd_b50c_m', 0.4: 'rd_b40_m',
            0.3: 'rd_b30_m', 0.2: 'rd_b20_m', 0.1: 'rd_b10_m'}
FB_RUN = {0.9: 'mf_raw4', 0.5: 'rd_b50_m'}
FB_INF = 'rd_inf_m'
RAD = 'mf_rad'
INF = 'rd_infc_m'

MET = [('I-AUROC', 'auroc_sp'), ('I-AP', 'ap_sp'), ('I-F1', 'f1_sp'),
       ('P-AUROC', 'auroc_px'), ('P-AP', 'ap_px'), ('P-F1', 'f1_px'),
       ('AUPRO', 'aupro_px')]

BASE = dict(k_image=150, pos_radius=1, layer_fuse='mean', img_cal='none',
            batch_size=8, use_aptr=0, use_centroid=0, use_transport_window=0,
            use_positional_bank=1, limit_imgs=0, shot_k=0,
            gpu_bank_cache_mb=9000, normalize_bank=1, image_size=512,
            crop_size=448, encoder_name='dinov3_vitb16')
LFR_PROD = dict(use_lonm=1, lonm_norm=1, lonm_norm_mode='raw', lonm_dmax=4,
                lonm_quadratic=0, lonm_M=24, lonm_energy=0.9, lonm_ridge=1e-4,
                lonm_ridge_rel=0.0, lonm_res_rank=0, lonm_curv_gain=1.0,
                lonm_radial=0, lonm_scale_beta=0.0, lonm_scale_q=0.9,
                lonm_maha=0, lonm_maha_rank=0, knn_chunk_imgs=128)

# ------------------------------------------------------------- planted values
MV_MEAN = {
    'auroc_sp': {0.9: 99.57, 0.8: 99.57, 0.7: 99.56, 0.6: 99.56, 0.5: 99.55,
                 0.4: 99.54, 0.3: 99.52, 0.2: 99.48, 0.1: 99.40},
    'ap_sp':    {0.9: 99.87, 0.8: 99.87, 0.7: 99.86, 0.6: 99.86, 0.5: 99.85,
                 0.4: 99.85, 0.3: 99.84, 0.2: 99.82, 0.1: 99.78},
    'f1_sp':    {0.9: 99.05, 0.8: 99.05, 0.7: 99.04, 0.6: 99.04, 0.5: 98.99,
                 0.4: 98.97, 0.3: 98.95, 0.2: 98.90, 0.1: 98.80},
    'auroc_px': {0.9: 98.73, 0.8: 98.73, 0.7: 98.73, 0.6: 98.72, 0.5: 98.73,
                 0.4: 98.72, 0.3: 98.71, 0.2: 98.69, 0.1: 98.60},
    'ap_px':    {0.9: 77.44, 0.8: 77.43, 0.7: 77.43, 0.6: 77.62, 0.5: 77.41,
                 0.4: 77.38, 0.3: 77.33, 0.2: 77.22, 0.1: 76.90},
    # 0.7 is planted 0.07 pp ABOVE 0.9 -> the "within noise" branch
    'f1_px':    {0.9: 72.63, 0.8: 72.62, 0.7: 72.70, 0.6: 72.60, 0.5: 72.60,
                 0.4: 72.57, 0.3: 72.50, 0.2: 72.35, 0.1: 71.90},
    'aupro_px': {0.9: 96.29, 0.8: 96.29, 0.7: 96.28, 0.6: 96.28, 0.5: 96.30,
                 0.4: 96.27, 0.3: 96.25, 0.2: 96.18, 0.1: 95.90},
}
RAD_MEAN = {'auroc_sp': 99.51, 'ap_sp': 99.83, 'f1_sp': 98.87,
            'auroc_px': 98.47, 'ap_px': 75.57, 'f1_px': 71.29,
            'aupro_px': 95.80}
INF_MEAN = {'auroc_sp': 99.59, 'ap_sp': 99.87, 'f1_sp': 99.05,
            'auroc_px': 98.73, 'ap_px': 77.45, 'f1_px': 72.64,
            'aupro_px': 96.30}

PROFILE = [1.6, 1.4, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           0.8, 0.8, 0.8, 0.8, 0.6, 0.6]
SPREAD = [(i - 7) * 0.01 for i in range(15)]        # sums to zero


def clip_of(b):
    return min(0.5, 0.002154 * (0.9 / b) ** 2)


def per_class_for(metric, beta, kind):
    """15 per-category values as fractions for one arm and one metric."""
    if kind in ('rad', 'inf'):
        base = RAD_MEAN[metric] if kind == 'rad' else INF_MEAN[metric]
        return [(base + SPREAD[i]) / 100.0 for i in range(15)]
    base = MV_MEAN[metric][0.9]
    if metric == 'ap_px' and abs(beta - 0.8) < 1e-9:
        d = [-0.02] * 10 + [0.0] * 5          # planted lose-on-10 / tie-on-5
        return [(base + SPREAD[i] + d[i]) / 100.0 for i in range(15)]
    shift = MV_MEAN[metric][beta] - base
    return [(base + SPREAD[i] + shift * PROFILE[i]) / 100.0 for i in range(15)]


def build_run(name, kind, beta=None, old=False):
    args = dict(BASE)
    if kind == 'rad':
        args.update(use_lonm=0, lonm_norm=0, lonm_norm_mode='ratio',
                    lonm_dmax=4, lonm_quadratic=1, lonm_M=24, lonm_trust=0.9,
                    lonm_unbounded=0, lonm_clipstat=0, knn_chunk_imgs=128)
    else:
        args.update(LFR_PROD)
        args.update(lonm_clipstat=0 if old else 1, lonm_trust=0.9,
                    lonm_unbounded=0)
        if old:
            args['knn_chunk_imgs'] = 32
        if kind == 'inf':
            args['lonm_unbounded'] = 1
        else:
            args['lonm_trust'] = beta
    args['item_list'] = list(MV)
    args['save_name'] = name

    per_class, mean = {}, {}
    for _, k in MET:
        vals = per_class_for(k, beta, kind)
        mean[k] = sum(vals) / len(vals)
        for i, c in enumerate(MV):
            per_class.setdefault(c, {})[k] = vals[i]
    for i, c in enumerate(MV):
        cl = clip_of(beta) if kind == 'lfr' else 0.0
        per_class[c].update(clip_frac=cl, clip_n=784 * 117,
                            clip_ratio_mean=0.32, clip_ratio_max=3.35)
    per_class['bottle']['clip_frac'] = (3 * clip_of(beta) if kind == 'lfr'
                                        else 0.0)
    return dict(args=args, mean=mean, per_class=per_class)


def build_fixture(root, fallback=False):
    runs = {RAD: build_run(RAD, 'rad')}
    if fallback:
        runs[FB_INF] = build_run(FB_INF, 'inf', 0.9, old=True)
    else:
        runs[INF] = build_run(INF, 'inf', 0.9)
    for b in BETAS:
        name = FB_RUN.get(b, BETA_RUN[b]) if fallback else BETA_RUN[b]
        runs[name] = build_run(name, 'lfr', b, old=fallback and b in FB_RUN)
    for name, d in runs.items():
        p = os.path.join(root, name)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, 'metrics.json'), 'w') as f:
            json.dump(d, f)
    return runs


DIRS = []


def fresh(name, fallback=False):
    d = tempfile.mkdtemp(prefix=f'betatest_{name}_')
    DIRS.append(d)
    build_fixture(d, fallback=fallback)
    return d


def run_table(root, perm='2000'):
    env = dict(os.environ, RAD_SAVED=root, RD_PERM_N=perm)
    r = subprocess.run([sys.executable, TABLE], capture_output=True, text=True,
                       env=env, cwd=HERE)
    return r.returncode, r.stdout + r.stderr


FAILS = []


def check(label, cond, extra=''):
    print(f'  [{"OK" if cond else "FAIL"}] {label}'
          + ('' if cond else '   ' + str(extra)))
    if not cond:
        FAILS.append(label)


def main():
    # ------------------------------------------------- syntax gate (py 3.11)
    # The local interpreter is 3.13 and the server is 3.11.7, and 3.12 relaxed
    # the f-string grammar (PEP 701).  A multi-line f-string expression that
    # parses here raises SyntaxError there, and ast.parse(feature_version=...)
    # does NOT catch it -- verified.  So lint the target before exercising it,
    # otherwise every check below fails on the server for one unhelpful reason.
    lint = os.path.join(HERE, 'lint_py311.py')
    lint_problems = []
    if os.path.isfile(lint):
        for target in (TABLE,):
            r = subprocess.run([sys.executable, lint, target],
                               capture_output=True, text=True, cwd=HERE)
            if r.returncode != 0:
                lint_problems.append(r.stdout.strip())
        check('py3.11 syntax gate (PEP 701 f-strings)', not lint_problems,
              lint_problems)

    # ---------------------------------------------------------- happy path
    root = fresh('happy')
    rc, out = run_table(root)
    print('===== fixture: happy path (rc=%d) =====' % rc)
    print(out)

    check('rc == 0', rc == 0, f'rc={rc}')
    check('audit line present', 'audit: the 10 LFR arms' in out)
    check('clip% monotone True',
          'monotone non-decreasing as beta falls: True' in out)
    check('P-AP argmax is beta = 0.6', '| P-AP | 0.6 |' in out)
    check('real exceedance flagged', '** BEATS 0.9 **' in out)
    check('within-noise branch used', 'within noise' in out)
    check('P-F1 argmax is beta = 0.7', '| P-F1 | 0.7 |' in out)
    check('sign test p=0.002 (beta 0.8 / P-AP)', 'p=0.002' in out)
    check('sign test p=0.000 (beta 0.6 / P-AP)', 'p=0.000' in out)
    # 4/7, not 5/7: AUPRO's planted maximum sits at beta = 0.5 (96.30 vs
    # 96.29), which is exactly the "within noise" case the check below wants.
    check('verdict counts 4/7',
          'beta = 0.9 attains the maximum on 4/7 metrics' in out,
          [l for l in out.splitlines() if 'attains the maximum' in l])
    check('AUPRO argmax is beta = 0.5', '| AUPRO | 0.5 |' in out)
    check('verdict reports +0.18 pp worst-case gap',
          'worst-case gap from beta = 0.9 to the best beta: +0.18 pp' in out)
    # The multiplicity context must be printed, and the direction claim must be
    # derived, not asserted: this fixture deliberately plants a tighter-is-
    # better effect (P-AP at 0.6, +0.18 pp), so the verdict has to name it.
    check('verdict reports multiplicity, not a bare "significant"',
          'pairings are nominally significant at 0.05' in out
          and 'expected by chance alone:' in out,
          [l for l in out.splitlines() if 'nominally significant' in l])
    check('verdict derives the direction of the significant pairings',
          'favour the tighter radius' in out, 
          [l for l in out.splitlines() if 'favour the tighter' in l])
    check('verdict reports the axis range against the noise floor',
          'the whole axis spans less than that on every metric' in out)
    check('verdict names the boundary, not an interior optimum',
          'attains its maximum at the unbounded end' in out
          and 'a genuine accuracy optimum would be interior' in out,
          [l for l in out.splitlines()
           if 'unbounded end' in l or 'strictly better' in l])
    check('RAD row values correct', '| RAD | 99.51 | 99.83 | 98.87 |' in out)
    check('rank of beta 0.9 is 22', '| 0.9 | 22 |' in out)
    check('rank of beta 0.1 is 3', '| 0.1 | 3 |' in out)
    check('default marker on the 0.9 row', '**<- default**' in out)
    check('paired table shows win/tie/loss',
          '(0/5/10)' in out)

    # --------------------------------------- F1 pre-v2 fallback arm names
    d = fresh('fallback', fallback=True)
    rc, out = run_table(d)
    print('===== F1 pre-v2 fallback names (rc=%d) =====' % rc)
    print(out[:1400])
    check('F1 rc == 0', rc == 0, f'rc={rc}')
    check('F1 fallbacks announced', '[fallback]' in out)
    check('F1 excuse printed with its measured reason',
          '[excused]' in out and 'knn_chunk_imgs 32 instead of 128' in out)
    check('F1 still produces the verdict', 'verdict' in out.lower())

    # ------------------------------------------------ N1 non-knob field
    d = fresh('n1')
    p = os.path.join(d, BETA_RUN[0.7], 'metrics.json')
    j = json.load(open(p))
    j['args']['lonm_ridge'] = 1e-3                     # not a knob
    json.dump(j, open(p, 'w'))
    rc, out = run_table(d)
    print('===== N1 non-identical non-knob field (rc=%d) =====' % rc)
    print(out[-700:])
    check('N1 rc == 2', rc == 2, f'rc={rc}')
    check('N1 NON-IDENTICAL reported',
          'NON-IDENTICAL' in out and 'lonm_ridge' in out)

    # ---------------------------------------------------- N2 mean mismatch
    d = fresh('n2')
    p = os.path.join(d, BETA_RUN[0.6], 'metrics.json')
    j = json.load(open(p))
    j['mean']['ap_px'] = 0.8010                        # 80.10 pp, not the avg
    json.dump(j, open(p, 'w'))
    rc, out = run_table(d)
    print('===== N2 mean mismatch (rc=%d) =====' % rc)
    print(out[-420:])
    check('N2 rc == 2', rc == 2, f'rc={rc}')
    check('N2 MEAN MISMATCH reported', 'MEAN MISMATCH' in out)

    # ------------------------------------------------------ N3 missing arm
    d = fresh('n3')
    shutil.rmtree(os.path.join(d, BETA_RUN[0.3]))
    rc, out = run_table(d)
    print('===== N3 missing arm (rc=%d) =====' % rc)
    print(out)
    check('N3 rc == 2', rc == 2, f'rc={rc}')
    check('N3 INCOMPLETE reported', 'INCOMPLETE' in out)
    check('N3 names the missing run', BETA_RUN[0.3] in out)
    check('N3 did not crash', 'Traceback' not in out)

    for x in DIRS:
        shutil.rmtree(x, ignore_errors=True)

    print()
    if FAILS:
        print(f'RESULT: {len(FAILS)} FAILED -> ' + '; '.join(FAILS))
        return 1
    print('RESULT: ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
