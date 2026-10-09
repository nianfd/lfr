#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/run_main_table.sh -- the two rows of Table II on MVTec-AD and VisA.
#
#   usage:  bash scripts/run_main_table.sh mvtec rad     # RAD  (1-NN + cosine)
#           bash scripts/run_main_table.sh mvtec lfr     # LFR  (this paper)
#           bash scripts/run_main_table.sh visa  rad
#           bash scripts/run_main_table.sh visa  lfr
#
# The two arms differ ONLY in --use_lonm / --lonm_norm / --lonm_norm_mode /
# --lonm_dmax / --lonm_quadratic (see RAD_ARM / LFR_ARM in scripts/common.sh).
# Everything else -- frozen features, normal memory, retrieval pool, anomaly-map
# recipe, top-1% image pooling, evaluator -- is shared by construction, so the
# paired difference isolates the reference construction and nothing else.
#
# Run names are the ones recorded for the paper, so that the table generators in
# tools/ pair the arms automatically:
#
#     mvtec rad -> mf_rad      mvtec lfr -> mf_raw4
#     visa  rad -> v3_rad      visa  lfr -> v3_rawl4
#
# Cost (single RTX 4090, all classes of one dataset in one process):
#   MVTec-AD ~25-35 min per row      VisA ~1.5-2 h per row
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DS="${1:-}"
ARM="${2:-}"

case "$DS/$ARM" in
  mvtec/rad) NAME=mf_rad;   A="$RAD_ARM"; D="$MVTEC_DIR"; B="$MVTEC_BANK";;
  mvtec/lfr) NAME=mf_raw4;  A="$LFR_ARM"; D="$MVTEC_DIR"; B="$MVTEC_BANK";;
  visa/rad)  NAME=v3_rad;   A="$RAD_ARM"; D="$VISA_DIR";  B="$VISA_BANK";;
  visa/lfr)  NAME=v3_rawl4; A="$LFR_ARM"; D="$VISA_DIR";  B="$VISA_BANK";;
  *) echo "usage: bash scripts/run_main_table.sh <mvtec|visa> <rad|lfr>" >&2; exit 2;;
esac

need_file "$D" "$B" "$ENC_W"

go_unless_done "$NAME" "$DS" $A

echo
echo "metrics: $OUT_DIR/$NAME/metrics.json"
echo "next:   python tools/bd_table.py          # matched-feature panel"
