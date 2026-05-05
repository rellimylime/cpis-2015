#!/bin/bash -l
#SBATCH --job-name=cpis_smoke_stats_v1
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/train_smoke_stats_v1_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/train_smoke_stats_v1_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# 5-epoch smoke train on the 20-tile stats_v1 subset to validate the 11-band
# pipeline end-to-end. Goal: loss decreases, band_stats.json has 11 entries,
# checkpoint loads in infer.py. NOT a modeling experiment — gates the full
# stats_v1 re-export.
#
# Splits:
#   train: 17 tiles (smoke tiles 000000-000019 minus the 3 in prod val)
#   val:   3 tiles (000003, 000016, 000017 — held out from 2021_v1 training)
#
# Archive this script after the smoke succeeds.

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
TILE_DIR="$REPO/data/imagery/2021_ssa_stats_v1"
RUN_DIR="$REPO/runs/semseg/smoke_stats_v1"
MASK_DIR="$RUN_DIR/masks"
TRAIN_TILES="$REPO/data/splits/smoke_stats_v1/train.txt"
VAL_TILES="$REPO/data/splits/smoke_stats_v1/val.txt"
ANCHORS="$REPO/data/anchors_prepared/anchor_truth/anchors/anchor_2021_normalized.gpkg"

mkdir -p "$RUN_DIR" "$MASK_DIR" "$REPO/outputs/semseg"

echo "Host: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# --- Step 1: rasterize masks for the 20 smoke tiles (idempotent) ---
N_MASKS=$(find "$MASK_DIR" -name "*_mask.tif" 2>/dev/null | wc -l)
if [ "$N_MASKS" -lt 20 ]; then
  echo ""
  echo "=== Rasterizing 2021 anchor masks for smoke tiles ($N_MASKS/20) ==="
  "$PYTHON_BIN" -m cpis.data.rasterize_labels \
    --tile-dir "$TILE_DIR" \
    --labels "$ANCHORS" \
    --output-dir "$MASK_DIR"
  echo "Rasterization done: $(date)"
else
  echo "=== Skipping rasterization ($N_MASKS masks already in $MASK_DIR) ==="
fi

# --- Step 2: 5-epoch smoke train (auto-detects 11 channels from tile) ---
echo ""
echo "=== 5-epoch smoke train (in_channels auto-detected) ==="
"$PYTHON_BIN" -m cpis.semseg.train \
  --tile-dir "$TILE_DIR" \
  --mask-dir "$MASK_DIR" \
  --train-tiles "$TRAIN_TILES" \
  --val-tiles "$VAL_TILES" \
  --run-dir "$RUN_DIR" \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-3 \
  --patch-size 256 \
  --base-filters 32 \
  --pos-weight 10

echo ""
echo "End: $(date)"
echo "Checkpoint: $RUN_DIR/best.pth"
echo "Band stats: $RUN_DIR/band_stats.json (should have 11 entries)"
echo "History: $RUN_DIR/history.json"
