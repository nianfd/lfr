"""Dump every argparse default in curvot_rad.py, as a check and as data.

Two consumers:

* ``bd_table.py`` may fill a field that is absent from a run's args (the field
  did not exist when that run was recorded) with the parser default, but only
  when the other side's recorded value equals that default.  ``check_late_flags``
  in bd_table.py cross-checks its LATE_FLAGS dict against this dump, so the
  values are verified mechanically instead of being trusted as transcription.
* ``bd_equiv.py`` is the measurement that makes such a fill sound in the first
  place; this file only establishes what the defaults ARE.

Writes exp/args_defaults.json (cwd-relative, i.e. the repository root) and prints
the JSON.

Run from the repository root:  python tools/dump_defaults.py

PACKAGING EDIT (the only change made to this file for release)
-------------------------------------------------------------
Upstream, SRC was the absolute path of the training box.  It is now resolved
relative to this file, and both the source and the output path can be given on
the command line or through the environment, so the tool is usable anywhere:

    python tools/dump_defaults.py [--src PATH] [--out PATH]

The default source is <repo root>/curvot_rad.py, which is exactly the file
upstream pointed at.
"""
import ast
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _opt(flag, env, default):
    if flag in sys.argv[1:]:
        return sys.argv[sys.argv.index(flag) + 1]
    return os.environ.get(env, default)


SRC = _opt('--src', 'LFR_SRC', os.path.join(_REPO, 'curvot_rad.py'))
OUT = _opt('--out', 'LFR_DEFAULTS_OUT', os.path.join('exp', 'args_defaults.json'))

tree = ast.parse(open(SRC, encoding='utf-8').read())
out = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'add_argument':
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            continue
        dflt, found = None, False
        for kw in node.keywords:
            if kw.arg == 'default':
                found = True
                try:
                    dflt = ast.literal_eval(kw.value)
                except Exception:
                    dflt = '<expr:' + ast.unparse(kw.value) + '>'
        # required=True means argparse enforces a value, so there is no default
        # to fill from; record it as such rather than as None.
        req = any(kw.arg == 'required' and getattr(kw.value, 'value', False)
                  for kw in node.keywords)
        out[flags[0]] = {'default': None if req else dflt, 'explicit': found,
                         'required': req}

text = json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True)
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
with open(OUT, 'w') as f:
    f.write(text + '\n')
print(f'[written] {OUT}  ({len(out)} flags)  from {SRC}')
print(text)
