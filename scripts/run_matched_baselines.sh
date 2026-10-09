#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/run_matched_baselines.sh -- the matched-feature baseline panel.
#
# Under the SAME frozen features, the SAME normal memory, the SAME retrieval
# pool, the SAME anomaly-map recipe, the SAME top-1% image pooling and the SAME
# evaluator, how do the two closest training-free reference constructions score?
#
#   subspacead  category-level PCA; score = squared orthogonal projection
#               residual, d = 64 retained components per layer
#   anoco       anchored bipartite Laplacian, closed form; reference = the
#               retrieved-image x spatial-neighbourhood pool (NOT a bipartite
#               graph over the full normal bank -- every other row sees the same
#               pool, so matching it is the point)
#
# Dinomaly is deliberately absent: its reference is a *trained* decoder, so a
# matched-feature treatment would require its training, not a swap of the
# reference.  Its published numbers stay in the paper as protocol context.
#
# Four runs, both datasets:
#   MVTec-AD  bd_sub_m, bd_anc_m       (~10 min each)
#   VisA      bd_sub_v, bd_anc_v       (~1.5-2 h each; ANoCo's closed-form
#                                       Laplacian solve is CPU-side, so expect
#                                       it several times slower than SubspaceAD)
#
# PaDiM and PatchCore are implemented in baseline_scores.py and reachable
# through the same interface (see --baseline in `python curvot_rad.py --help`),
# but are outside this panel: the claim under test is about the reference
# construction being locally fitted, which is what SubspaceAD and ANoCo
# instantiate.  If you do enable PaDiM, report its regularisation variants
# together -- its score is very sensitive to how the position-specific
# covariance is regularised at this feature scale (22 pp spread on a single
# MVTec-AD category between the published constant and the alternatives).
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

need_file "$MVTEC_DIR" "$VISA_DIR" "$MVTEC_BANK" "$VISA_BANK" "$ENC_W"

echo "########## matched-feature baselines  $(date) ##########"

go_unless_done bd_sub_m mvtec --baseline subspacead
go_unless_done bd_anc_m mvtec --baseline anoco
go_unless_done bd_sub_v visa  --baseline subspacead
go_unless_done bd_anc_v visa  --baseline anoco

echo "baseline stage done $(date)"
echo
echo "next:  python tools/bd_table.py"
