#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/run_beta_sweep.sh -- fill in the radius axis, MVTec-AD only.
#
#   beta = 0.8 0.7 0.6 0.4 0.3 0.2 0.1
#
# beta = 0.9 comes from rd_b90c_m (scripts/run_radius_ablation.sh) and R = inf
# from rd_infc_m; together they give the nine-point family reported in the
# paper.  Every arm here is the SAME command line as the LFR arms of the radius
# ablation, character for character, except the value of --lonm_trust -- the
# family is single-knob by construction, which is what makes the "0.9 is not a
# tuned sweet spot" argument auditable rather than rhetorical.
#
#   beta   0.9  0.8  0.7  0.6  0.5  0.4  0.3  0.2  0.1
#   rank    22   19   17   15   13   10    8    6    3      (of M = 24)
#
# The sweep therefore spans the 3rd to the 22nd of 24 retained coordinates: at
# the low end the reference degenerates towards the local normal centroid, at
# the high end towards the unbounded span of the R = inf arm.  Measured result:
# nine points, no interior optimum, all ranges inside the seed noise floor.
#
# Cost: ~25-35 min per arm on one RTX 4090, so ~3-4 h for the seven.
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

need_file "$MVTEC_DIR" "$MVTEC_BANK" "$ENC_W"

echo "########## beta sweep (MVTec-AD)  $(date) ##########"

for B in 0.8 0.7 0.6 0.4 0.3 0.2 0.1; do
  TAG="${B#0.}0"        # 0.8 -> 80, 0.1 -> 10  (matches the recorded run names)
  go_unless_done "rd_b${TAG}_m" mvtec $LFR_ARM --lonm_clipstat 1 --lonm_trust "$B"
done

echo "beta sweep done $(date)"
echo
echo "next:  python tools/rd_beta_table.py"
