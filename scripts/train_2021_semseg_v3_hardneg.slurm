#!/bin/bash -l
#SBATCH --job-name=cpis_semseg_2021v3
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/train_2021v3_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/train_2021v3_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
RUN_DIR="$REPO/runs/semseg/2021_v3_hardneg_stats_v1"
TILE_DIR="$REPO/data/imagery/2021_ssa_stats_v1"
MASK_DIR="$REPO/runs/semseg/stats_v1_masks"
NEGATIVE_CENTERS="$REPO/outputs/semseg/hard_negatives_2021_v1.gpkg"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
PATCH_SIZE="${PATCH_SIZE:-256}"
PATCHES_PER_TILE="${PATCHES_PER_TILE:-24}"
PIVOT_SAMPLE_PROB="${PIVOT_SAMPLE_PROB:-0.50}"
NEGATIVE_SAMPLE_PROB="${NEGATIVE_SAMPLE_PROB:-0.20}"
IN_CHANNELS="${IN_CHANNELS:-11}"
BASE_FILTERS="${BASE_FILTERS:-32}"
POS_WEIGHT="${POS_WEIGHT:-10}"

mkdir -p "$RUN_DIR" "$REPO/outputs/semseg"

echo "Host: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true
echo "Params: epochs=$EPOCHS batch=$BATCH_SIZE patch=$PATCH_SIZE patches_per_tile=$PATCHES_PER_TILE pivot_sample_prob=$PIVOT_SAMPLE_PROB negative_sample_prob=$NEGATIVE_SAMPLE_PROB pos_weight=$POS_WEIGHT augment=true"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f "$NEGATIVE_CENTERS" ]; then
  echo "Missing hard negatives: $NEGATIVE_CENTERS" >&2
  exit 1
fi

"$PYTHON_BIN" -m cpis.semseg.train \
  --tile-dir "$TILE_DIR" \
  --mask-dir "$MASK_DIR" \
  --train-tiles data/splits/train_tiles_2021.txt \
  --val-tiles data/splits/2021_val_tiles_landsat.txt \
  --run-dir "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch-size "$PATCH_SIZE" \
  --patches-per-tile "$PATCHES_PER_TILE" \
  --pivot-sample-prob "$PIVOT_SAMPLE_PROB" \
  --negative-sample-prob "$NEGATIVE_SAMPLE_PROB" \
  --negative-centers "$NEGATIVE_CENTERS" \
  --base-filters "$BASE_FILTERS" \
  --pos-weight "$POS_WEIGHT" \
  --in-channels "$IN_CHANNELS" \
  --augment

echo "End: $(date)"
echo "Best checkpoint: $RUN_DIR/best.pth"
