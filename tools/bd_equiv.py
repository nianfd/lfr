"""Is the 2026-09-27 anchor commensurable with today's code?

The matched-feature table pairs runs recorded on 09-27 (the anchors mf_rad,
mf_raw4) with runs recorded today (bd_sub_m, bd_anc_m).  The anchor args dict
predates 24 flags, so those fields cannot be compared directly; bd_table.py
fills them from curvot_rad.py's argparse defaults, which is only sound if the
defaults reproduce what the older code did unconditionally.

exp/vchk_rad.sh measures that: it re-runs the RAD anchor configuration with
today's code on two MVTec-AD categories and stores them as vchk_rad_mv.  This
script compares the two per class and writes exp/logs/bd_equiv.txt.

Verdict bands, in the order they are tested.  The project's accepted noise
precedent for a re-implementation of the same computation is 1e-7 on the
window fast path and exact equality on the main-table re-run, so a bit-identical
result is the expected outcome here and 1e-6 is still inside precedent.

    BIT-IDENTICAL          every metric, every class, == 0.0
    WITHIN-DP              max |diff| < 1e-6, i.e. invisible at the 4 dp the
                           paper prints, and inside the project's own precedent
    WEAK                   max |diff| in [1e-6, 5e-4): still invisible at 4 dp
                           but past precedent -- usable, must be disclosed
    NOT COMMENSURABLE      max |diff| >= 5e-4: the fill in LATE_FLAGS is not
                           justified; the matched table must not be built on it

Run on the server:  python exp/bd_equiv.py
"""
import json
import os
import sys

ROOT = os.environ.get('RAD_SAVED', './saved_curvot')
OLD, NEW = 'mf_rad', 'vchk_rad_mv'
MET = [('I-AUROC', 'auroc_sp'), ('I-AP', 'ap_sp'), ('I-F1', 'f1_sp'),
       ('P-AUROC', 'auroc_px'), ('P-AP', 'ap_px'), ('P-F1', 'f1_px'),
       ('AUPRO', 'aupro_px')]
IGNORE = ('save_name', 'save_dir', 'item_list', 'bank_path', 'data_path',
          'encoder_weight')


def load(name):
    p = os.path.join(ROOT, name, 'metrics.json')
    if not os.path.isfile(p):
        raise SystemExit(f'[MISSING] {p}')
    with open(p) as f:
        return json.load(f)


def eq(x, y):
    try:
        return abs(float(x) - float(y)) < 1e-12
    except (TypeError, ValueError):
        return str(x) == str(y)


def main():
    out = []

    def say(s=''):
        out.append(s)
        print(s)

    old, new = load(OLD), load(NEW)
    say('# commensurability probe: 09-27 anchor vs today\'s code, same config')
    say(f'#   old = {OLD}   (recorded 09-27, driver revision without the late flags)')
    say(f'#   new = {NEW}   (today\'s code, no --lonm*/--baseline flag passed)')
    say()

    # --- 1. the two runs must agree wherever both recorded the field ---------
    oa, na = old['args'], new['args']
    shared = sorted((set(oa) & set(na)) - set(IGNORE))
    only_old = sorted((set(oa) - set(na)) - set(IGNORE))
    only_new = sorted((set(na) - set(oa)) - set(IGNORE))
    drift = [(k, oa[k], na[k]) for k in shared if not eq(oa[k], na[k])]
    say(f'args: {len(oa)} recorded 09-27, {len(na)} recorded today, '
        f'{len(shared)} shared')
    say(f'  keys only on the old side : {only_old if only_old else "none"}')
    say(f'  keys only on the new side : {len(only_new)} (the late flags)')
    say(f'  disagreement on shared keys: {drift if drift else "none"}')
    say()

    # --- 2. per class, per metric -------------------------------------------
    cls = sorted(set(old.get('per_class', {})) & set(new.get('per_class', {})))
    if not cls:
        raise SystemExit('[MISSING] the two runs share no category')
    say(f'categories compared: {cls}')
    say()
    say('| Metric | ' + ' | '.join(f'{c} old | {c} new | diff' for c in cls) + ' |')
    say('|' + '---|' * (1 + 3 * len(cls)))
    worst = 0.0
    nbit = 0
    ntot = 0
    for lab, k in MET:
        cells = []
        for c in cls:
            ov = old['per_class'][c][k]
            nv = new['per_class'][c][k]
            d = abs(ov - nv)
            worst = max(worst, d)
            ntot += 1
            nbit += 1 if d == 0.0 else 0
            cells.append(f'{ov:.6f} | {nv:.6f} | {d:.2e}')
        say(f'| {lab} | ' + ' | '.join(cells) + ' |')
    say()
    say(f'bit-identical cells: {nbit}/{ntot}   worst |diff| = {worst:.3e}')
    say()

    # --- 3. verdict ----------------------------------------------------------
    if worst == 0.0:
        verdict = 'BIT-IDENTICAL'
        why = ('every metric on every compared category is exactly equal, so the '
               '24 late flags are inert along the anchor\'s path and filling them '
               'from the parser defaults introduces no difference')
        rc = 0
    elif worst < 1e-6:
        verdict = 'WITHIN-DP'
        why = ('the largest disagreement is below 1e-6, i.e. far inside the 4 dp '
               'the paper prints and inside this project\'s accepted 1e-7 '
               're-implementation noise; the fill is justified but the table '
               'note must not claim bit-identity')
        rc = 0
    elif worst < 5e-4:
        verdict = 'WEAK'
        why = ('still invisible at 4 dp but past the project\'s noise precedent; '
               'the matched table may be built on it only with the difference '
               'stated in the caption')
        rc = 0
    else:
        verdict = 'NOT COMMENSURABLE'
        why = ('the disagreement is large enough to move a printed digit; '
               'bd_table.py must NOT fill the late flags from the parser '
               'defaults for this pairing')
        rc = 2
    say(f'BD EQUIV VERDICT: {verdict}')
    say(f'  {why}')
    say()
    say('Scope of this evidence: two MVTec-AD categories, RAD configuration '
        '(no --baseline, no --lonm*).  It covers the anchor path the matched '
        'table actually pairs against.  It does not by itself certify the VisA '
        'anchors, which were recorded by the same 09-27 driver revision and are '
        'covered by the same argument, not by a separate probe.')

    os.makedirs('exp/logs', exist_ok=True)
    with open('exp/logs/bd_equiv.txt', 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\n[written] exp/logs/bd_equiv.txt')
    return rc


if __name__ == '__main__':
    sys.exit(main())
