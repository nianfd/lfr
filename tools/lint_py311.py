"""Pre-upload syntax gate: catch grammar this project's server cannot parse.

Why this exists
---------------
The local interpreter is Python 3.13; the experiment server runs 3.11.7.
``ast.parse(..., feature_version=(3, 11))`` does **not** help -- the parser
used is still the 3.13 one, and it accepts the 3.12-only f-string grammar
regardless of ``feature_version`` (verified: a multi-line f-string expression
is accepted at ``feature_version=(3, 11)``).

That gap already cost one round trip: an f-string whose expression spanned two
source lines parsed locally and raised SyntaxError on the server.  PEP 701
relaxed three things in 3.12, and this script rejects all three:

  1. an f-string *expression* spanning source lines;
  2. a backslash inside an f-string expression;
  3. reusing the enclosing quote character inside an f-string
     (``f'{d['k']}'``; the different-quote form is fine in 3.11).

Usage:  python lint_py311.py file.py [file.py ...]
Exit code 0 = clean, 1 = at least one problem.
"""
import io
import sys
import tokenize

FSTRING_START = getattr(tokenize, 'FSTRING_START', None)
FSTRING_END = getattr(tokenize, 'FSTRING_END', None)

# Prefix letters that may precede the opening quote of a string literal.
_PREFIX = set('fFrRbBuU')


def _quote_of(text):
    """Return the quote character of a string token, or None."""
    i = 0
    while i < len(text) and text[i] in _PREFIX:
        i += 1
    return text[i] if i < len(text) and text[i] in '\'"' else None


def check(path):
    """Return a list of human-readable problems for one file."""
    with open(path, 'rb') as fh:
        raw = fh.read()
    problems = []

    try:
        toks = list(tokenize.tokenize(io.BytesIO(raw).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as e:
        return [f'tokenize failed (may itself be a 3.12-only form): {e}']

    if FSTRING_START is None:
        return ['interpreter exposes no FSTRING_START (needs CPython >= 3.12); '
                'cannot lint']

    # Single pass, one stack of enclosing f-string quote characters.  A plain
    # (non-f) string token inside an f-string is what same-quote nesting looks
    # like to the 3.12+ tokenizer -- it does NOT emit a nested FSTRING_START.
    stack = []          # (quote_char, start_token)
    seen_start = []     # (start_line, end_line, start_token) per finished f-string
    for tok in toks:
        t = tok.type
        if t == FSTRING_START:
            q = _quote_of(tok.string)
            if stack and q == stack[-1][0]:
                problems.append(
                    f'line {tok.start[0]}: nested f-string reuses the '
                    f'enclosing quote {q!r} (PEP 701, 3.12+)')
            stack.append((q, tok))
        elif t == FSTRING_END:
            if stack:
                q, start = stack.pop()
                if tok.end[0] != start.start[0]:
                    problems.append(
                        f'line {start.start[0]}: f-string expression spans '
                        f'lines {start.start[0]}-{tok.end[0]}; Python 3.11 '
                        f'rejects a multi-line f-string expression '
                        f'(PEP 701, 3.12+)')
                seen_start.append(start)
        elif stack and t == tokenize.STRING:
            q = _quote_of(tok.string)
            if q is not None and q == stack[-1][0]:
                problems.append(
                    f'line {tok.start[0]}: string literal {tok.string!r} '
                    f'reuses the enclosing f-string quote {q!r} '
                    f'(PEP 701, 3.12+)')
        elif stack and t == tokenize.OP and tok.string == '\\':
            problems.append(f'line {tok.start[0]}: backslash inside an '
                            f'f-string expression (PEP 701, 3.12+)')
    if stack:
        problems.append('unbalanced FSTRING_START/FSTRING_END (interpreter '
                        'mismatch?); lint result may be incomplete')
    return problems


def main(argv):
    if FSTRING_START is None:
        print(f'lint_py311: running on {sys.version_info.major}.'
              f'{sys.version_info.minor}; no FSTRING tokens, cannot lint')
        return 0
    bad = 0
    for path in argv:
        problems = check(path)
        if problems:
            bad = 1
            print(f'[FAIL] {path}')
            for p in problems:
                print(f'    {p}')
        else:
            print(f'[OK]   {path}')
    print('RESULT: ' + ('PROBLEMS FOUND' if bad else 'PY311-CLEAN'))
    return bad


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
