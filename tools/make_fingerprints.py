#!/usr/bin/env python3
"""Regenerate docs/CODE_FINGERPRINTS.txt and docs/md5sums.txt.

Why this exists
---------------
The release claims "the shipped runtime code is the code that produced the
reported numbers".  A claim like that is worth exactly its verifiability, so:
  * HISTORICAL below holds the md5 each file had ON THE TRAINING BOX, recorded
    before the release was assembled.  Any FROZEN file that no longer matches
    is a hard error -- this script exits non-zero rather than print a table
    that would silently overstate the result.
  * docs/md5sums.txt is emitted in `md5sum -c` format, so a reader can verify
    the whole tree with one command.

This script is not part of the running pipeline and contains no arithmetic that
affects any reported number.

Usage
-----
    python tools/make_fingerprints.py            # from the repository root
    python tools/make_fingerprints.py --check    # verify only, write nothing
"""
import argparse
import hashlib
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

OUT_DOC = 'docs/CODE_FINGERPRINTS.txt'
OUT_SUMS = 'docs/md5sums.txt'

# ---------------------------------------------------------------- provenance
# md5 on the training box (RAD-main) for the code that produced the paper's
# numbers.  Keyed by path relative to the release root.
HISTORICAL = {
    # runtime modules
    'curvot_rad.py':                    'a6db3855c2070f107ef6985434029e34',
    'curvot_utils.py':                  'ca16e42a25754a6d7b342fbc9d137290',
    'cu_ref.py':                        '054601b61cad9d3004b93f2f162e2bf8',
    'baseline_scores.py':               'f5e494f11c94c1daac42354f79bba73b',
    'dataset.py':                       'bd901630c758d0c627747a693ef1b988',
    'utils.py':                         '9854e9fe31b459b5bc6ec8ec6678e3a3',
    'rad_mvtec_visa_3dadam.py':         '53ce51f7a627395875bc9d32e0b0c368',
    'build_bank_multilayer.py':         '987c3dbd9722eba052a659917a459210',
    'bank_builder_disk.py':             '268235ac4dc287a94bfe55009b1d1942',
    'build_realiad_bank_multilayer.py': '3f52ff668ba54a98729e07f591006df4',
    'ri_launch_curvot_rad.py':          '40f5230ba350cea4d79a5fd1c6bf3a45',
    'ri_sharedbank.py':                 '35bf9d12a239a407981e6c1fbb941df5',
    # harnesses that were on the training box
    'test_baselines.py':                '0688dd846a46ef6056c4661caf99efe3',
    'test_unbounded_equiv.py':          'b3b6e8e32e004aff7bc1c36e178e8bf2',
    # tools
    'tools/bd_table.py':                '74513d0d5bedb43d8e5e8cc04221845a',
    'tools/bd_gapchk.py':               'f3e35178aecf574d0fb53dfd4c7d392e',
    'tools/bd_equiv.py':                '8440412dd951df3b751555a68b07dd47',
    'tools/rd_beta_table.py':           '0cfd49d21198974fd5d31ef58381a00c',
    'tools/rd_radius_table.py':         '9adb25c91057bfd4562c59236ad9d31d',
    'tools/rd_radius_perclass.py':      'cc7a284ace9deb862cfeb22fc7236fc0',
    'tools/ri_table_rho4.py':           'd1883f3a2e1ca227f13be97c82a66a21',
    'tools/lint_py311.py':              '858228ff246fdbefcdda3b38ec5f01d9',
    'tools/test_beta_table.py':         '5d304fa901c06e6e91bb8a2dcc878e8f',
    'tools/p_fastwin_diff.py':          '5c51c7d32508d2671a20ee6aad982932',
}

# the one file that carries a packaging edit, plus its pre-edit md5
EDIT_FILE = 'tools/dump_defaults.py'
EDIT_PRE_MD5 = 'db4e23aa2e1d5a9633a35e2a8a294744'

# harnesses that existed only on the development machine: no historical hash
HARNESS = {'test_window_fast.py', 'bench_window_fast.py'}

ASSET = {'docs/main_results.tex', 'figures/framework.png'}

SKIP_NAMES = {'.gitkeep'}
SKIP_DIRS = {'.git', '__pycache__', '.idea', '.vscode'}
SKIP_FILES = {OUT_DOC, OUT_SUMS}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def walk():
    """Every tracked-looking file, as repo-relative posix paths, sorted."""
    out = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name in SKIP_NAMES:
                continue
            rel = os.path.relpath(os.path.join(root, name), REPO)
            rel = rel.replace(os.sep, '/')
            if rel in SKIP_FILES:
                continue
            out.append(rel)
    return sorted(out)


def status(rel):
    if rel in HISTORICAL:
        return 'FROZEN'
    if rel == EDIT_FILE:
        return 'EDIT'
    if rel in HARNESS:
        return 'HARNESS'
    if rel.startswith('scripts/'):
        return 'DRIVER'
    if rel.startswith('refs/'):
        return 'DATA'
    if rel in ASSET:
        return 'ASSET'
    return 'GLUE'


LEGEND = """\
Status key
----------
FROZEN
    byte-identical to the file on the training box that produced the
    numbers in the paper.  The md5 below is simultaneously the shipped
    file and the historical file, so the identity is checkable with
    `md5sum -c docs/md5sums.txt`.
DRIVER
    the portable shell drivers written for this release.  They are new
    code: their job is to issue the same command lines the paper ran,
    with the machine-specific paths lifted into environment variables.
    They contain no arithmetic of their own -- every number in the paper
    comes from curvot_rad.py.
HARNESS
    a standalone test harness that existed only on the development
    machine.  It imports curvot_utils.py, so it exercises the FROZEN
    module shipped here; there is no historical md5 to compare against.
EDIT
    a packaging edit, and the only one.  tools/dump_defaults.py carried the
    training box's absolute path; the edit is marked in the source with a
    "PACKAGING EDIT" banner.  It changes where the tool reads and writes,
    nothing about what it computes.
DATA
    reference data rather than code: the published RAD reference column of
    the Real-IAD comparison, with its provenance recorded inside the file.
    It is not measured here and is never mixed into our own columns.
ASSET
    figure and table sources for this README and for the manuscript.
GLUE
    packaging files: requirements, ignore rules, the README, these notes,
    and this generator.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='verify only; write nothing')
    args = ap.parse_args()

    rels = walk()
    rows, bad = [], []
    for rel in rels:
        got = md5(os.path.join(REPO, rel.replace('/', os.sep)))
        st = status(rel)
        if st == 'FROZEN' and got != HISTORICAL[rel]:
            bad.append((rel, got, HISTORICAL[rel]))
        rows.append((got, st, rel))

    if bad:
        print('FROZEN MISMATCH -- the release no longer matches what ran:',
              file=sys.stderr)
        for rel, got, want in bad:
            print('  %-42s got %s want %s' % (rel, got, want), file=sys.stderr)
        return 2

    counts = Counter(st for _, st, _ in rows)
    n_frozen = counts['FROZEN']

    head = [
        'CODE FINGERPRINTS',
        '=================',
        '',
        'MD5 of every shipped file.  The claim "the released code is the code',
        'that ran" is worth exactly as much as your ability to check it, so',
        'every runtime module is listed with the md5 it had on the training',
        'box.  Verify the whole tree with:',
        '',
        '    md5sum -c docs/md5sums.txt',
        '',
        LEGEND.rstrip('\n'),
        '',
        'md5                                status   path',
        '---------------------------------- -------- ----',
    ]
    body = ['%s   %-8s %s' % (g, st, rel) for g, st, rel in rows]
    tail = [
        '',
        'FROZEN files: %d of %d listed' % (n_frozen, len(rows)),
        'md5 of %s before the packaging edit: %s' % (EDIT_FILE, EDIT_PRE_MD5),
        '',
    ]

    doc = '\n'.join(head + body + tail) + '\n'
    sums = ''.join('%s  %s\n' % (g, rel) for g, _, rel in rows)

    if args.check:
        print('OK: %d files, %d FROZEN, no mismatch' % (len(rows), n_frozen))
        return 0

    with open(os.path.join(REPO, OUT_DOC), 'w', encoding='utf-8', newline='') as f:
        f.write(doc)
    with open(os.path.join(REPO, OUT_SUMS), 'w', encoding='utf-8', newline='') as f:
        f.write(sums)

    print('%s: %d files, %s' % (OUT_DOC, len(rows), dict(counts)))
    print('%s: %d lines' % (OUT_SUMS, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
