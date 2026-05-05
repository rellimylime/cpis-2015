#!/bin/bash -l
#SBATCH --job-name=cpis_semseg_infer_2021v1
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/infer_2021v1_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/infer_2021v1_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
CHECKPOINT="$REPO/runs/semseg/2021_v1/best.pth"
BAND_STATS="$REPO/runs/semseg/2021_v1/band_stats.json"
PROB_DIR="$REPO/outputs/semseg/probs_2021_v1"

mkdir -p "$REPO/outputs/semseg" "$PROB_DIR"

echo "Host: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

echo ""
echo "=== Full inference (saving prob maps to $PROB_DIR) ==="
"$PYTHON_BIN" -m cpis.semseg.infer \
  --tile-dir data/imagery/2015_ssa \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$PROB_DIR" \
  --band-stats "$BAND_STATS" \
  --patch-size 256 \
  --overlap 32 \
  --threshold 0.5 \
  --save-probs

echo ""
echo "End: $(date)"
echo "Prob maps: $PROB_DIR"
echo "Next: sbatch scripts/threshold_sweep_2021_v1.slurm"
