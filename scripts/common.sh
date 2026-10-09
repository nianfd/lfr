#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/common.sh -- shared configuration and run helpers.
#
# Source this from every other script in scripts/.  Nothing here is specific to
# the machine the reported numbers were produced on: point the variables below
# at your own copies of the datasets and the checkpoint (see README section 3),
# then every driver in this directory becomes runnable.
#
# Override anything by exporting it before calling a driver, e.g.
#
#   DATA_ROOT=/mnt/data bash scripts/run_main_table.sh mvtec lfr
#
# The *protocol* constants (k_image, pos_radius, layer indices, image score)
# are NOT configurable here on purpose: they are what the paper fixes, and
# changing them silently would make the released commands stop reproducing the
# released table.
# ---------------------------------------------------------------------------
set -u

LFR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LFR_ROOT"

: "${PYTHON:=python}"                        # interpreter that has torch + deps
: "${DATA_ROOT:=$LFR_ROOT/data}"             # datasets        (README section 3.1)
: "${BANK_DIR:=$LFR_ROOT/bank}"              # normal-memory banks (section 3.3)
: "${MODEL_DIR:=$LFR_ROOT/models}"           # DINOv3 checkpoint   (section 3.2)
: "${ENC_W:=$MODEL_DIR/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
: "${OUT_DIR:=./saved_curvot}"               # per-run metrics.json lands here
: "${CACHE_MB:=9000}"                        # GPU bank-cache budget (MVTec/VisA)
: "${BANK_TMP:=$LFR_ROOT/.bank_tmp}"         # scratch for the Real-IAD builder

MVTEC_DIR="$DATA_ROOT/mvtec-ad"
VISA_DIR="$DATA_ROOT/VisA_pytorch/1cls"
REALIAD_DIR="$DATA_ROOT/Real-IAD"

MVTEC_BANK="$BANK_DIR/mvtec_dinov3_vitb16_multilayer36911_448_bank.pth"
VISA_BANK="$BANK_DIR/visa_dinov3_vitb16_multilayer36911_448_bank.pth"
REALIAD_BANK="$BANK_DIR/realiad_dinov3_vitb16_multilayer36911_448_bank.pth"

# Class lists, in the paper's order.  `--bank_item_order` must be given the
# Real-IAD list verbatim: the bank stores no class order, and the on-disk probe
# yields the ALPHABETICAL listing, which differs from this one at
# usb / usb_adaptor / u_block.  Mixing the two silently mis-aligns the row
# blocks, so both the builder and every evaluation get the same string.
MV_CLASSES="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
VI_CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
RI_CLASSES="audiojack bottle_cap button_battery end_cap eraser fire_hood mint mounts pcb phone_battery plastic_nut plastic_plug porcelain_doll regulator rolled_strip_base sim_card_set switch tape terminalblock toothbrush toy toy_brick transistor1 u_block usb usb_adaptor vcpill wooden_beads woodstick zipper"

# ---------------------------------------------------------------------------
# The two arms of the main experiment.  These are the only difference between
# the RAD row and the LFR row of Table II, on every dataset.
# ---------------------------------------------------------------------------
# RAD as published: 1-NN + cosine over the same retrieval pool.
RAD_ARM="--use_lonm 0"
# LFR: locally fitted reference with the PNR readout, spelled exactly as in the
# paper's default configuration.
LFR_ARM="--use_lonm 1 --lonm_norm 1 --lonm_norm_mode raw --lonm_dmax 4 --lonm_quadratic 0"

mkdir -p "$LFR_ROOT/exp/logs" "$BANK_DIR" "$OUT_DIR"

# ---------------------------------------------------------------------------
# go <run_name> <mvtec|visa> [extra flags...]
#
# One evaluation run: every class of one dataset, in a single process, writing
# saved_curvot/<run_name>/metrics.json.  Mirrors the driver used for the paper
# so the recorded argument dictionaries stay field-for-field comparable.
# ---------------------------------------------------------------------------
go() {
  local name=$1; shift
  local ds=$1; shift
  if [ "$ds" = "mvtec" ]; then
    D=$MVTEC_DIR; B=$MVTEC_BANK; CL="$MV_CLASSES"; KI=150; PR=1; EXTRA="--knn_chunk_imgs 128"
  elif [ "$ds" = "visa" ]; then
    D=$VISA_DIR;  B=$VISA_BANK;  CL="$VI_CLASSES"; KI=900; PR=2; EXTRA=""
  else
    echo "unknown dataset: $ds" >&2; return 2
  fi
  echo "=========== $name ($ds)  $(date +%H:%M:%S) ===========" >> exp/results.txt
  "$PYTHON" curvot_rad.py \
      --data_path "$D" --bank_path "$B" --item_list $CL \
      --save_dir "$OUT_DIR" --save_name "$name" \
      --k_image $KI --pos_radius $PR \
      --encoder_weight "$ENC_W" \
      --skip_bank_check 1 --gpu_bank_cache_mb $CACHE_MB \
      $EXTRA "$@" > exp/logs/$name.log 2>&1
  grep -h "^Mean:" exp/logs/$name.log | tail -1 >> exp/results.txt
  tail -1 exp/results.txt
}

# Skip a run whose metrics.json is already on disk, so that a driver can be
# re-invoked after an interruption without re-paying for finished work.
go_unless_done() {
  local name=$1
  if [ -s "$OUT_DIR/$name/metrics.json" ]; then
    echo "[skip] $name : metrics.json already on disk, not re-run"
    return 0
  fi
  go "$@"
}

# Cheap precondition check: fail loudly with an actionable message instead of
# a stack trace inside the first data-loading loop.
need_file() {
  for f in "$@"; do
    if [ ! -e "$f" ]; then
      echo "!! missing: $f" >&2
      echo "   see README section 3 for how to obtain and place it." >&2
      exit 1
    fi
  done
}
