#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/run_realiad_rho4.sh -- Real-IAD, all 30 classes, one arm.
#
#   usage:  bash scripts/run_realiad_rho4.sh a     # RAD, as published
#           bash scripts/run_realiad_rho4.sh b     # LFR
#
# WHY rho = 4
# -----------
# The RAD paper's README prints rho = 0 for Real-IAD.  That value reproduces
# nothing: measured on audiojack with everything else held fixed,
#
#     rho          0      1      2      3      4      5
#     I-AUROC   .7531  .8511  .8765  .8777  .8777  .8777
#     P-AP       6.39  36.35  54.80  55.24  55.45  55.57
#
# against the paper's 87.9 / 57.6.  At rho = 4 six of the seven metrics land
# within +-0.2 of the paper's, and why the seventh (P-AP) cannot is discussed in
# the paper: this box's copy of the dataset holds 512x512 JPEGs where the
# official release is 1024, which blurs the mask boundary.  MVTec-AD (rho = 1)
# and VisA (rho = 2) reproduce to the decimal at the printed values, so this one
# cell is a typo in the upstream README, not a tuning choice.
#
# ONE INVOCATION PER CLASS.  Two reasons, both load-bearing:
#   1. Resumability.  114,585 test images per arm is a long job; the full-test
#      row of every class is appended the moment that class finishes, so the run
#      can be stopped at a class boundary and still hand over a complete,
#      honest partial table.  A class already recorded is skipped.
#   2. The GPU row cache is scoped to one class.  A test image of class X
#      retrieves mostly X's own bank rows, so one class's working set is at most
#      900 rows x 4 layers x 2.297 MB ~ 8.3 GB, which fits the cache budget set
#      by RI_CACHE_MB (14000 MB -> 1523 rows/layer, above the measured 1260-row
#      maximum class).  The per-image gather is then served from VRAM and the
#      cache refills once per class instead of thrashing.
#
# Measured cost, single RTX 4090: arm a ~31.9 h, arm b ~19.9 h with --lonm_fast.
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RI_K=900            # K = 900 global [CLS] neighbours, shared by every layer
RI_PR=4             # rho = 4, see the note above
RI_CACHE_MB=14000   # 14000 MB -> cap 1523 rows/layer > the largest class (1260)

ARM="${1:-}"
case "$ARM" in
  a) RSLT=exp/results_ri_a_rho4.txt; PFX=ri_rad4; EXTRA="$RAD_ARM";;
  b) RSLT=exp/results_ri_b_rho4.txt; PFX=ri_c14;  EXTRA="$LFR_ARM --lonm_fast 1";;
  *) echo "usage: bash scripts/run_realiad_rho4.sh a|b" >&2; exit 2;;
esac

need_file "$REALIAD_DIR" "$REALIAD_BANK" "$ENC_W" ri_launch_curvot_rad.py

# The allocator fix must be exported before the first CUDA call.  Without it the
# 2.02 GiB neighbour gather (900 x 784 x 768 x 4 B = one image's worth) can fail
# even though the total would fit: the failure is fragmentation, and this is the
# setting the allocator's own message asks for.  It changes allocation
# granularity only -- tensors and results are unaffected.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SDIR=./saved_ri_rho4
mkdir -p "$SDIR" exp/logs
[ -f "$RSLT" ] || echo "# Real-IAD arm $ARM at rho=$RI_PR -- protocol in scripts/common.sh, rationale in this script's header" > "$RSLT"

OK=0
for IT in $RI_CLASSES; do
  NAME=${PFX}_${IT}
  if grep -q "^${IT}: I-AUROC:" "$RSLT" 2>/dev/null; then
    echo "[skip] $IT already recorded"
    continue
  fi
  echo "=========== $NAME (realiad-$ARM rho=$RI_PR)  $(date +%H:%M:%S) ===========" >> "$RSLT"
  echo "# flags: --layout realiad --k_image $RI_K --pos_radius $RI_PR --resize_mask 256 --max_ratio 0.01 --gpu_bank_cache_mb $RI_CACHE_MB $EXTRA" >> "$RSLT"
  T0=$(date +%s)
  # ri_launch_curvot_rad.py, not curvot_rad.py: the 352 GB bank cannot be loaded
  # by the plain path on a box with vm.overcommit_memory=0 (one private mapping
  # of 352 GB is refused, and the fallback eager load is OOM-killed at ~124 GB
  # RSS).  The launcher swaps in a MAP_SHARED read-only loader over the *same*
  # bytes; its filename still ends in curvot_rad.py on purpose.  For MVTec-AD
  # and VisA the plain entry point is correct and is what the drivers use.
  "$PYTHON" ri_launch_curvot_rad.py \
      --data_path "$REALIAD_DIR" --layout realiad --bank_path "$REALIAD_BANK" \
      --item_list "$IT" --bank_item_order $RI_CLASSES \
      --save_dir "$SDIR" --save_name "$NAME" \
      --k_image $RI_K --pos_radius $RI_PR --resize_mask 256 --max_ratio 0.01 \
      --encoder_weight "$ENC_W" \
      --skip_bank_check 1 --gpu_bank_cache_mb $RI_CACHE_MB --num_workers 8 \
      $EXTRA > exp/logs/$NAME.log 2>&1
  RC=$?
  T1=$(date +%s)
  LINE=$(grep -h "^${IT}: I-AUROC:" exp/logs/$NAME.log | tail -1)
  PEAK=$(grep -h 'peak GPU mem' exp/logs/$NAME.log | tail -1)
  HIT=$(grep -h 'bank-cache rows:' exp/logs/$NAME.log | tail -1)
  if [ -n "$LINE" ]; then
    echo "$LINE" >> "$RSLT"
    echo "# wall=$((T1-T0))s | ${PEAK:-no-peak-line} | ${HIT:-NO-CACHE-REPORT}" >> "$RSLT"
    echo "[ok] $IT  $((T1-T0))s  $LINE"
    OK=$((OK + 1))
  else
    echo "# FAILED rc=$RC wall=$((T1-T0))s -- no per-class line; see exp/logs/$NAME.log" >> "$RSLT"
    echo "[FAIL] $IT rc=$RC -- last 5 lines of its log:"
    tail -5 exp/logs/$NAME.log
  fi
done

# A stage that produced NO rows must not certify itself as done: every other
# guard keys off a row appearing in the result file, so an all-fail arm would
# otherwise be indistinguishable from a complete one.  Partial success still
# counts as done -- an honest partial table is the design.
COVER=$(grep -c "^.*: I-AUROC:" "$RSLT" 2>/dev/null || echo 0)
if [ "${OK:-0}" -eq 0 ]; then
  echo "Real-IAD arm ${ARM} FAILED -- 0 of 30 classes produced a row" >&2
  exit 1
fi
echo "# coverage: $COVER / 30 classes recorded for arm $ARM at rho=$RI_PR" >> "$RSLT"
echo "Real-IAD arm ${ARM} done -- $COVER/30 classes"
echo
echo "next:  python tools/ri_table_rho4.py --repo . --out RAD_REALIAD_TABLE.md"
