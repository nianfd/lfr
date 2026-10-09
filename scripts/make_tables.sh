#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/make_tables.sh -- regenerate every table in the paper from runs.
#
# Nothing here measures anything: each generator reads the per-run
# saved_*/<name>/metrics.json (and, for Real-IAD, the append-only
# exp/results_ri_{a,b}_rho4.txt) and prints a table.  They exit non-zero rather
# than print a table when a run is missing a partner, when a run's recorded
# arguments contradict the dataset spec, or when a per-class mean disagrees with
# the stored mean -- a table that is quietly wrong is worse than no table.
#
# Usage:  bash scripts/make_tables.sh [all|main|baseline|radius|beta|realiad]
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WHAT="${1:-all}"
mkdir -p exp/tables
PY="$PYTHON"

# The cross-version field audit bd_table.py performs needs the parser's own view
# of every default, not a hand-copied list.  Regenerate it from the entry point.
gen_defaults() {
  if [ ! -s exp/args_defaults.json ] || [ curvot_rad.py -nt exp/args_defaults.json ]; then
    "$PY" tools/dump_defaults.py --src curvot_rad.py > exp/args_defaults.json
    echo "[defaults] exp/args_defaults.json regenerated from curvot_rad.py"
  fi
}

run() {
  local label=$1; shift
  echo "=================== $label ==================="
  "$@" 2>&1 | tee "exp/tables/${label}.txt"
  echo
}

main_table() {
  # The MVTec/VisA rows live in the paper's main table; the printed per-run
  # lines are the raw material for it.
  for n in mf_rad mf_raw4 v3_rad v3_rawl4; do
    if [ -s "$OUT_DIR/$n/metrics.json" ]; then
      echo "--- $n"
      "$PY" - "$OUT_DIR/$n/metrics.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))['mean']
keys = [('I-AUROC','auroc_sp'),('I-AP','ap_sp'),('I-F1','f1_sp'),
        ('P-AUROC','auroc_px'),('P-AP','ap_px'),('P-F1','f1_px'),('AUPRO','aupro_px')]
print('  ' + '  '.join('%s=%.2f' % (l, 100*m[k]) for l, k in keys))
PYEOF
    else
      echo "--- $n : [not run]"
    fi
  done
}

all_or() {
  case "$WHAT" in
    all) return 0 ;;
    "$1") return 0 ;;
    *) return 1 ;;
  esac
}

if all_or main; then run main_table main_table; fi

if all_or baseline; then
  gen_defaults
  run matched_baselines "$PY" tools/bd_table.py
fi

if all_or radius; then
  run radius_ablation   "$PY" tools/rd_radius_table.py
  run radius_perclass   "$PY" tools/rd_radius_perclass.py
fi

if all_or beta; then
  run beta_sweep "$PY" tools/rd_beta_table.py
fi

if all_or realiad; then
  run realiad "$PY" tools/ri_table_rho4.py --repo . \
      --paper_json refs/rad_realiad_ab.json \
      --out exp/tables/RAD_REALIAD_TABLE.md
fi

echo "tables written under exp/tables/"
