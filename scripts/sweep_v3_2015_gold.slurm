#!/bin/bash -l
#SBATCH --job-name=sweep_v3_2015_gold
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/sweep_v3_2015_gold_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/sweep_v3_2015_gold_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Run threshold sweep of v3 2015 inference against gold holdout.
# Compare F1 vs the 2021_v1 baseline (0.1448 @ thresh 0.8).

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
PROBS_DIR="$REPO/outputs/semseg/probs_2021_v3_2015"
GOLD="$REPO/data/gold/2015_gold_val_v2_holdout.gpkg"
OUT_DIR="$REPO/outputs/semseg/sweep_2021_v3_2015"

mkdir -p "$OUT_DIR" "$REPO/outputs/semseg"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Probs: $PROBS_DIR"
echo "Gold: $GOLD"
echo "Output: $OUT_DIR"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m cpis.semseg.threshold_sweep \
  --probs-dir "$PROBS_DIR" \
  --gold "$GOLD" \
  --output-dir "$OUT_DIR"

echo "End: $(date)"
echo "Results: $OUT_DIR"
