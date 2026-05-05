#!/bin/bash -l
#SBATCH --job-name=cpis_sweep_2021v2
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/sweep_2021v2_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/sweep_2021v2_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
PROB_DIR="$REPO/outputs/semseg/probs_2021_v2"
GOLD="$REPO/data/gold/2015_gold_val_v2_holdout.gpkg"
SWEEP_OUT="$REPO/outputs/semseg/sweep_2021_v2"

mkdir -p "$SWEEP_OUT"

echo "Start: $(date)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m cpis.semseg.threshold_sweep \
  --prob-dir "$PROB_DIR" \
  --labels "$GOLD" \
  --output-dir "$SWEEP_OUT" \
  --thresholds 0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --min-area-px 30 \
  --max-area-px 2000 \
  --min-circularity 0.4 \
  --iou-threshold 0.5 \
  --clip-label-bounds \
  --label "2021_v2_gold"

echo "End: $(date)"
echo "Results: $SWEEP_OUT/sweep_results.json"
