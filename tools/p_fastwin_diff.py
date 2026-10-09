#!/usr/bin/env python3
"""
p_fastwin_diff.py -- field-by-field comparison of two metrics.json files.

WHY THIS EXISTS
---------------
--lonm_fast changes HOW the LONM window similarities are formed: one bmm per
(layer, chunk) instead of one `bank[:, idx[p,kk], :]` gather per (position,
offset).  The Real-IAD arm b row is being re-run under it, and the MVTec / VisA
main-table rows have to be re-run too (exp/p_fastwin_recheck.sh) so that every
number in the paper comes from one code version.  This tool is the evidence
that the published numbers do not move.

WHAT IT CHECKS, IN THIS ORDER
-----------------------------
  1. args       -- the two runs must differ ONLY by the keys we intend.
                   If a flag drifted, every later comparison is meaningless;
                   so this is checked first and reported loudly.
  2. mean       -- per metric: abs delta, rel delta, bit-equality (== 0.0, not
                   `< 1e-9` -- project rule) and 4-decimal equality.
                   4 dp is what the paper prints; that is the bar that counts.
  3. per_class  -- per class, per metric: how many cells are bit-equal and the
                   worst absolute delta, with the offending class named.

USAGE
-----
  python p_fastwin_diff.py OLD.json NEW.json [label]
  python p_fastwin_diff.py --pairs mf_raw4 v3_rawl4
        shorthand for the two MVTec/VisA main-table rows:
            saved_curvot/<run>/metrics.json   (OLD)
            saved_recheck/<run>/metrics.json  (NEW)
  python p_fastwin_diff.py --ri
        shorthand for the Real-IAD arm-b single-code re-run: compares
            saved_ri_rho4_pathv1/<run>/metrics.json  (OLD, pre-patch path)
            saved_ri_rho4/<run>/metrics.json         (NEW, --lonm_fast 1)
        over every run that exists in BOTH directories.

Exit code is 0 when no published (4 dp) value moved, 1 otherwise -- so this can
be used as a gate.
"""

import json
import os
import sys

# Keys that are expected to differ and carry no algorithmic meaning.
IGNORE_ARGS = {'save_dir', 'save_name'}

# Keys that only exist in the NEW file because the argparse grew a flag after
# the OLD run was produced.  Reported separately, never counted as a drift.
KNOWN_NEW_ARG_KEYS = {
    'lonm_fast',                       # this patch
    'lonm_maha', 'lonm_maha_ridge', 'lonm_maha_rank', 'anc_beta',  # earlier line
}

# Keys that ARE allowed to differ, because one of them is the variable under
# test.  When the OLD run was produced by a build that already had the flag (the
# zipper probe pair) it shows up on both sides and would otherwise be counted as
# a drift, which reads as "something else changed too" -- the opposite of the
# point.  Everything else that differs is still a drift.
EXPECTED_DIFF_ARGS = {'lonm_fast'}

# The seven published metrics, in paper order.  mean_normal_score is reported
# last as a diagnostic; it is not a published headline number.
PUBLISHED = ['auroc_sp', 'ap_sp', 'f1_sp', 'auroc_px', 'ap_px', 'f1_px', 'aupro_px']
EXTRA_MEAN = ['mean_normal_score']


def load(path):
    with open(path) as f:
        return json.load(f)


def bit_equal(a, b):
    """Project rule: bit-equality is `== 0.0`, never `< 1e-9`."""
    return (a - b) == 0.0


def rel(a, b):
    d = max(abs(a), abs(b))
    return 0.0 if d == 0.0 else abs(a - b) / d


def fmt(x, w=12, p=6):
    return f'{x:>{w}.{p}f}'


def diff_args(old, new, label):
    print(f'--- 1. args  ({label}) ---')
    for k in sorted(IGNORE_ARGS):
        if old.get(k) != new.get(k):
            print(f'  [ok-ignored] {k}: {old.get(k)!r} -> {new.get(k)!r}')
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    drift = []
    expected = []
    for k in sorted(set(old) & set(new)):
        if k in IGNORE_ARGS:
            continue
        if old[k] != new[k]:
            (expected if k in EXPECTED_DIFF_ARGS else drift).append(k)
    unexplained_new = [k for k in only_new if k not in KNOWN_NEW_ARG_KEYS]
    if only_old:
        print(f'  [WARN] args only in OLD: {only_old}')
    if only_new:
        known = [k for k in only_new if k in KNOWN_NEW_ARG_KEYS]
        if known:
            print(f'  [ok]    args only in NEW (argparse grew after OLD was made): {known}')
    if unexplained_new:
        print(f'  [WARN] args only in NEW, not a known added flag: {unexplained_new}')
    if expected:
        for k in expected:
            print(f'  [ok]    variable under test: {k}: {old[k]!r} -> {new[k]!r}')
    if drift:
        print(f'  [DRIFT] value differs on shared keys: {drift}')
        for k in drift:
            print(f'           {k}: {old[k]!r} -> {new[k]!r}')
    else:
        print('  [ok]    no value drift on shared keys')
    return drift


def diff_mean(old, new, label):
    om, nm = old.get('mean', {}), new.get('mean', {})
    keys = [k for k in PUBLISHED if k in om or k in nm] + \
           [k for k in EXTRA_MEAN if k in om or k in nm]
    keys += [k for k in sorted(set(om) | set(nm)) if k not in keys]
    print(f'--- 2. mean  ({label}) ---')
    hdr = (f'  {"metric":<18}{"OLD":>14}{"NEW":>14}{"abs":>12}{"rel":>11}'
           f'{"bit":>6}{"4dp":>6}')
    print(hdr)
    moved4 = []
    for k in keys:
        if k not in om or k not in nm:
            print(f'  {k:<18} [missing on one side]')
            continue
        a, b = om[k], nm[k]
        be = bit_equal(a, b)
        eq4 = round(a, 4) == round(b, 4)
        if not eq4:
            moved4.append(k)
        print(f'  {k:<18}{a:>14.8f}{b:>14.8f}{abs(a - b):>12.3e}'
              f'{rel(a, b):>11.3e}{("Y" if be else "n"):>6}{("Y" if eq4 else "N"):>6}')
    return moved4


def diff_perclass(old, new, label):
    op, np_ = old.get('per_class', {}), new.get('per_class', {})
    shared = [c for c in op if c in np_]
    only_o = [c for c in op if c not in np_]
    only_n = [c for c in np_ if c not in op]
    print(f'--- 3. per_class  ({label}) ---')
    if only_o or only_n:
        print(f'  [WARN] class sets differ: only-OLD={only_o} only-NEW={only_n}')
    if not shared:
        print('  (no shared classes)')
        return {}
    # metric axis: union of metric keys across shared classes
    mkeys = []
    for c in shared:
        for k in op[c]:
            if k not in mkeys:
                mkeys.append(k)
    worst = {}
    nbit = {}
    ncell = 0
    for k in mkeys:
        if k in ('n_images',):
            continue
        mx = 0.0
        who = None
        nb = 0
        n = 0
        for c in shared:
            if k not in op[c] or k not in np_[c]:
                continue
            a, b = op[c][k], np_[c][k]
            n += 1
            ncell += 1
            if bit_equal(a, b):
                nb += 1
            d = abs(a - b)
            if d > mx:
                mx, who = d, c
        if n:
            worst[k] = (mx, who, nb, n)
    print(f'  classes compared: {len(shared)}   cells compared: {ncell}')
    print(f'  {"metric":<18}{"max|d|":>12}   {"bit-equal":>12}   worst class')
    for k in PUBLISHED + [x for x in mkeys if x not in PUBLISHED and x != 'n_images']:
        if k not in worst:
            continue
        mx, who, nb, n = worst[k]
        print(f'  {k:<18}{mx:>12.3e}   {f"{nb}/{n}":>12}   {who if mx else "-"}')
    return worst


def one(old_path, new_path, label):
    if not os.path.exists(old_path):
        print(f'!! missing OLD: {old_path}')
        return None
    if not os.path.exists(new_path):
        print(f'!! missing NEW: {new_path}')
        return None
    old, new = load(old_path), load(new_path)
    print('=' * 78)
    print(f'PAIR: {label}')
    print(f'  OLD = {old_path}')
    print(f'  NEW = {new_path}')
    print('=' * 78)
    drift = diff_args(old.get('args', {}), new.get('args', {}), label)
    moved4 = diff_mean(old, new, label)
    diff_perclass(old, new, label)
    if moved4:
        print(f'  VERDICT [{label}]: {len(moved4)} metric(s) MOVED at 4 dp: {moved4}')
    else:
        print(f'  VERDICT [{label}]: all published metrics identical at 4 dp')
    if drift:
        print(f'  !! arguments drifted ({drift}) -- read the numbers with care')
    print()
    return moved4


RI_OLD = 'saved_ri_rho4_pathv1'
RI_NEW = 'saved_ri_rho4'


def main(argv):
    if argv and argv[0] == '--pairs':
        rows = argv[1:]
        if not rows:
            rows = ['mf_raw4', 'v3_rawl4']
        bad = 0
        for r in rows:
            m = one(f'saved_curvot/{r}/metrics.json',
                    f'saved_recheck/{r}/metrics.json', r)
            if m is None or m:
                bad += 1
        return 1 if bad else 0

    if argv and argv[0] == '--ri':
        if not os.path.isdir(RI_OLD):
            print(f'!! no {RI_OLD}/ -- the pre-patch evidence is gone')
            return 1
        runs = sorted(d for d in os.listdir(RI_OLD)
                      if os.path.exists(os.path.join(RI_OLD, d, 'metrics.json')))
        if not runs:
            print(f'!! {RI_OLD}/ has no metrics.json')
            return 1
        bad = 0
        for r in runs:
            np_ = os.path.join(RI_NEW, r, 'metrics.json')
            if not os.path.exists(np_):
                print(f'[pending] {r}: NEW not written yet')
                continue
            m = one(os.path.join(RI_OLD, r, 'metrics.json'), np_, r)
            if m is None or m:
                bad += 1
        return 1 if bad else 0

    if len(argv) >= 2:
        label = argv[2] if len(argv) > 2 else os.path.basename(os.path.dirname(argv[0]))
        m = one(argv[0], argv[1], label)
        return 1 if (m is None or m) else 0

    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
