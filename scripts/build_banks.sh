#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/build_banks.sh -- build the normal-memory banks, once per dataset.
#
# The bank is the frozen patch-descriptor memory of the anomaly-free training
# split, one .pth per dataset, reused by every run and every method comparison.
# It is derived data: it can always be rebuilt from the dataset + checkpoint,
# and it is what makes the whole benchmark training-free (no gradient ever
# touches these features).
#
# Protocol (identical for all three datasets, and identical to the paper):
#   encoder      DINOv3 ViT-B/16, weights frozen
#   layers       block indices 3, 6, 9, 11   (0-based; the paper counts blocks
#                from 1, so this is "layers 4/7/10/12")
#   input        resize to 512, then 448 centre crop  (448/16 = 28 -> 28x28 grid)
#   storage      float32 patch descriptors, normalised
#
# Disk: MVTec-AD is a few GB, VisA ~77 GB, Real-IAD ~352 GB (36,465 training
# images across 30 classes).  Build Real-IAD on a volume with that much room.
#
# Usage:  bash scripts/build_banks.sh [mvtec|visa|realiad|all]
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WHICH="${1:-all}"
mkdir -p "$BANK_DIR" "$LFR_ROOT/exp/logs"

build() {
  local name=$1 data=$2 classes=$3 bank=$4
  if [ -s "$bank" ]; then
    echo "[skip] $name bank already present: $bank"
    return 0
  fi
  need_file "$data"
  echo "=================== $name bank start $(date) ==================="
  # `python` (loader=standard) is correct here: the builder streams images and
  # writes the checkpoint, it never maps a finished bank back in.
  "$PYTHON" build_bank_multilayer.py \
    --data_path "$data" \
    --item_list $classes \
    --encoder_name dinov3_vitb16 \
    --encoder_weight "$ENC_W" \
    --image_size 512 --crop_size 448 --batch_size 32 --num_workers 4 \
    --layer_idx_list 3 6 9 11 \
    --bank_path "$bank" 2>&1 | tee "exp/logs/bank_$(basename "$bank" .pth).log"
  echo "=================== $name bank done rc=${PIPESTATUS[0]} $(date) ==================="
}

case "$WHICH" in
  mvtec)
    build mvtec   "$MVTEC_DIR"   "$MV_CLASSES" "$MVTEC_BANK"
    ;;
  visa)
    build visa    "$VISA_DIR"    "$VI_CLASSES" "$VISA_BANK"
    ;;
  realiad)
    # Real-IAD needs its own builder (the 30-class json layout, and a disk-backed
    # path because the result -- ~352 GB of float32 descriptors -- exceeds RAM
    # while being written).  The builder materialises a full-size temp copy
    # first, so point --tmp_dir at a volume with room; otherwise the peak on the
    # output volume is twice the bank.
    need_file "$REALIAD_DIR"
    mkdir -p "$BANK_TMP"
    echo "=================== realiad bank start $(date) ==================="
    "$PYTHON" build_realiad_bank_multilayer.py \
      --data_path "$REALIAD_DIR" \
      --item_list $RI_CLASSES \
      --encoder_name dinov3_vitb16 \
      --encoder_weight "$ENC_W" \
      --image_size 512 --crop_size 448 --batch_size 32 --num_workers 4 \
      --layer_idx_list 3 6 9 11 \
      --tmp_dir "$BANK_TMP" \
      --bank_path "$REALIAD_BANK" 2>&1 | tee exp/logs/bank_realiad.log
    echo "=================== realiad bank done rc=${PIPESTATUS[0]} $(date) ==================="
    ;;
  all)
    "$0" mvtec; "$0" visa; "$0" realiad
    ;;
  *)
    echo "usage: bash scripts/build_banks.sh [mvtec|visa|realiad|all]" >&2
    exit 2
    ;;
esac

echo "bank directory:"
ls -la "$BANK_DIR"
