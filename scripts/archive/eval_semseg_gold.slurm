#!/bin/bash -l
#SBATCH --job-name=cpis_semseg_eval
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/eval_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/eval_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

PRED="$REPO/outputs/semseg/pivots_2015_v2.gpkg"
GOLD="$REPO/data/gold/2015_gold_val_v2_holdout.gpkg"
OUT="$REPO/outputs/semseg/eval_v2.json"

echo "Start: $(date)"

"$PYTHON_BIN" -m cpis.eval.eval_semseg \
  --pred "$PRED" \
  --gold "$GOLD" \
  --output "$OUT" \
  --iou-threshold 0.5 \
  --label "semseg_v2_best"

echo "End: $(date)"
echo "Results: $OUT"
