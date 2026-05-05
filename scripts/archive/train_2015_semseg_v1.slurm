#!/bin/bash -l
#SBATCH --job-name=cpis_semseg_2015_v2
#SBATCH --output=/home/ermiller/cpis-2015/runs/semseg/2015_v2/slurm_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/runs/semseg/2015_v2/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Train U-Net semseg on 2015 Landsat tiles with stable_pivots weak labels.
# 44 train tiles, 13 val tiles. Val keyed to BCE+Dice loss (best.pth saved).

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
WORK_DIR="$REPO/runs/semseg/2015_v2"
mkdir -p "$WORK_DIR"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m cpis.semseg.train \
  --tile-dir data/imagery/2015_ssa \
  --mask-dir runs/semseg/2015_masks \
  --train-tiles data/splits/train_tiles_2015.txt \
  --val-tiles data/splits/2015_val_tiles.txt \
  --run-dir "$WORK_DIR" \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-3 \
  --patch-size 256 \
  --base-filters 32 \
  --pos-weight 50.0

echo "End: $(date)"
