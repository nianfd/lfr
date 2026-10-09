#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/run_radius_ablation.sh -- the support-radius axis, MVTec-AD only.
#
# The reference is fitted locally and then clipped to a radius R, the
# beta-quantile of the retained normal coordinates.  This script produces the
# three arms that make the axis legible:
#
#   rd_b90c_m   beta = 0.9   (the paper's default; the reference is bounded)
#   rd_b50c_m   beta = 0.5
#   rd_infc_m   R = infinity (unbounded affine span = pure orthogonal residual)
#
# With M = 24 candidates kept per query, the radius is the
# (1 + round(beta*(M-1)))-th smallest retained coordinate, so beta = 0.9 keeps
# the 22nd of 24 and beta = 0.5 the 13th.  The measured finding is that the axis
# is a *sensitivity bound*, not an accuracy knob: see README section 4.
#
# --lonm_clipstat 1 only records the fraction of coordinates whose retained
# coordinate exceeded R.  It is numerically inert (verified bit-identical by
# test_unbounded_equiv.py), so its presence does not break the pairing; it is
# what lets the clipping activation rate be reported alongside the scores.
#
# The radius axis is an MVTec-AD-only study by design: MVTec-AD plus VisA would
# double the stage's cost for a second copy of the same negative result, and the
# paper reports the axis where the reference geometry is easiest to read.
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

need_file "$MVTEC_DIR" "$MVTEC_BANK" "$ENC_W"

echo "########## radius ablation (MVTec-AD)  $(date) ##########"

go_unless_done rd_b90c_m mvtec $LFR_ARM --lonm_clipstat 1 --lonm_trust 0.9
go_unless_done rd_b50c_m mvtec $LFR_ARM --lonm_clipstat 1 --lonm_trust 0.5
go_unless_done rd_infc_m mvtec $LFR_ARM --lonm_clipstat 1 --lonm_unbounded 1

echo "radius stage done $(date)"
echo
echo "next:  python tools/rd_radius_table.py"
echo "       python tools/rd_radius_perclass.py     # per-class mechanism evidence"
