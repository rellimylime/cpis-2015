#!/bin/bash -l
#SBATCH --job-name=cpis_2021_landsat_v1
#SBATCH --output=/home/ermiller/cpis-2015/runs/instseg/2021_landsat_v1/slurm_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/runs/instseg/2021_landsat_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Train Cascade Mask R-CNN on 2021 Landsat dataset.
# Fine-tunes from Chen et al. checkpoint.
# --total-epochs 24: old S2 model peaked at epoch 19; 24 gives headroom
# without burning 140-epoch GPU time on a diagnostic run.
# Best checkpoint saved automatically by --validate (keyed to segm_mAP).

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
export PROJ_LIB="/home/ermiller/.conda/envs/cpi_fix/share/proj"
export PROJ_DATA="$PROJ_LIB"

WORK_DIR="/home/ermiller/cpis-2015/runs/instseg/2021_landsat_v1"
mkdir -p "$WORK_DIR"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $PYTHON_BIN"
nvidia-smi || true

cd /home/ermiller/global_cpis_codes

"$PYTHON_BIN" tools/new_method/train_paper_model.py \
  --dataset-root /home/ermiller/cpis-2015/runs/instseg/2021_dataset \
  --config model/cascade_mask_rcnn_pointrend_cbam.py \
  --checkpoint model/cascade_mask_rcnn_pointrend_cbam.pth \
  --work-dir "$WORK_DIR" \
  --log-file "$WORK_DIR/train_paper_model.log" \
  --samples-per-gpu 2 \
  --workers-per-gpu 4 \
  --total-epochs 24 \
  --validate \
  --compute-stats \
  --device cuda

echo "End: $(date)"
