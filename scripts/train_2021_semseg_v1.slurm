#!/bin/bash -l
#SBATCH --job-name=cpis_semseg_2021
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/train_2021_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/train_2021_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Train U-Net on 2021 Landsat tiles with 2021 anchor labels.
# Motivation: stable_pivots (2015 training data) only covered 44 tiles in
# north/east Africa. 2021 anchors have 29K pivots across 202 tiles including
# 7,185 pivots in the South Africa gold holdout region — fixes the geographic gap.
# Same sensor (Landsat 30m), temporal shift only.
# pos_weight=10 (down from 50) since 2021 anchors are much denser labels.

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
RUN_DIR="$REPO/runs/semseg/2021_v1"
MASK_DIR="$RUN_DIR/masks"

mkdir -p "$RUN_DIR" "$MASK_DIR" "$REPO/outputs/semseg"

echo "Host: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# --- Step 1: Rasterize 2021 anchor labels (skip if masks already exist) ---
N_MASKS=$(find "$MASK_DIR" -name "*_mask.tif" 2>/dev/null | wc -l)
if [ "$N_MASKS" -lt 202 ]; then
  echo ""
  echo "=== Rasterizing 2021 anchor labels ($N_MASKS/202 masks found) ==="
  "$PYTHON_BIN" -m cpis.data.rasterize_labels \
    --tile-dir data/imagery/2021_ssa \
    --labels data/anchors_prepared/anchor_truth/anchors/anchor_2021_normalized.gpkg \
    --output-dir "$MASK_DIR"
  echo "Rasterization done: $(date)"
else
  echo "=== Skipping rasterization ($N_MASKS masks already in $MASK_DIR) ==="
fi

# --- Step 2: Train U-Net ---
echo ""
echo "=== Training U-Net on 2021 Landsat ==="
"$PYTHON_BIN" -m cpis.semseg.train \
  --tile-dir data/imagery/2021_ssa \
  --mask-dir "$MASK_DIR" \
  --train-tiles data/splits/train_tiles_2021.txt \
  --val-tiles data/splits/2021_val_tiles_landsat.txt \
  --run-dir "$RUN_DIR" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --patch-size 256 \
  --base-filters 32 \
  --pos-weight 10

echo ""
echo "End: $(date)"
echo "Best checkpoint: $RUN_DIR/best.pth"
echo "Next: sbatch scripts/infer_semseg_2015.slurm (update CHECKPOINT to $RUN_DIR/best.pth)"
