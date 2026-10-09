"""Dry-run the cross-version identity audit for VisA before the last arm lands.

bd_table.py skips the audit entirely while an arm is missing, so a VisA schema
gap would only surface after ~80 min of ANoCo.  This calls the same
identity_report() on the arms that already exist, so the whitelist can be
verified (or extended) immediately.

Run:  python exp/bd_gapchk.py [dataset]
"""
import sys

import bd_table as B

ds = sys.argv[1] if len(sys.argv) > 1 else 'visa'
runs = B.load_runs()
want = [('RAD', B.BASE[ds], {'mvtec': 'mf_rad', 'visa': 'v3_rad'}),
        ('LFR', B.BASE[ds], {'mvtec': 'mf_raw4', 'visa': 'v3_rawl4'}),
        ('SubspaceAD', B.BASE[ds], {'mvtec': 'bd_sub_m', 'visa': 'bd_sub_v'}),
        ('ANoCo', B.BASE[ds], {'mvtec': 'bd_anc_m', 'visa': 'bd_anc_v'})]
chosen = [(lab, names[ds]) for lab, _s, names in want if names[ds] in runs]

print(f'dataset={ds}  arms present={[n for _, n in chosen]}')
for n in [n for _, n in chosen]:
    print(f'  {n:10s} args keys={len(runs[n]["args"]):3d}')
audit, unexplained = {}, {}
bad = B.identity_report(runs, ds, chosen, audit, unexplained)
print()
print('filled from LATE_FLAGS (both sides carry the parser default):')
if audit:
    for f, v in sorted(audit.items()):
        print(f'    {f}: {v!r}')
else:
    print('    (none)')
print('unexplained gaps (present on exactly one side, not a known late flag):')
if unexplained:
    for f, (ln, fn, val) in sorted(unexplained.items()):
        print(f'    {f}: carried by {ln}({fn}) as {val!r}')
else:
    print('    (none)')
print()
print('VERDICT:', 'GAPS PRESENT' if bad else 'CLEAN')
sys.exit(1 if bad else 0)
