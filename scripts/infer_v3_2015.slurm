#!/bin/bash -l
#SBATCH --job-name=infer_v3_2015
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/infer_v3_2015_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/infer_v3_2015_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Infer v3 (11-band + hard-neg trained) on 2015 stats_v1 tiles.
# Output: probability maps for threshold sweep.

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
TILE_DIR="$REPO/data/imagery/2015_ssa_stats_v1"
CHECKPOINT="$REPO/runs/semseg/2021_v3_hardneg_stats_v1/best.pth"
BAND_STATS="$REPO/runs/semseg/2021_v3_hardneg_stats_v1/band_stats.json"
INFER_DIR="$REPO/outputs/semseg/probs_2021_v3_2015"

mkdir -p "$INFER_DIR" "$REPO/outputs/semseg"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Band stats: $BAND_STATS"
echo "Tile dir: $TILE_DIR ($(find "$TILE_DIR" -name "*.tif" 2>/dev/null | wc -l) tiles)"
echo "Output: $INFER_DIR"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi || true

"$PYTHON_BIN" -m cpis.semseg.infer \
  --tile-dir "$TILE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --band-stats "$BAND_STATS" \
  --output-dir "$INFER_DIR" \
  --save-probs

N_PROBS=$(find "$INFER_DIR" -name "*.npy" 2>/dev/null | wc -l)
echo "End: $(date)"
echo "Prob maps saved: $N_PROBS"

if [ "$N_PROBS" -ge 100 ]; then
  echo "SUCCESS: Inference complete"
  exit 0
else
  echo "WARNING: Expected ~107 prob maps, found $N_PROBS"
  exit 0
fi
