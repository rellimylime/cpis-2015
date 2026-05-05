#!/bin/bash -l
#SBATCH --job-name=cpis_hardneg_v1
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/hardneg_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/hardneg_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
PROB_DIR="$REPO/outputs/semseg/probs_2021_v1_train"
OUT="$REPO/outputs/semseg/hard_negatives_2021_v1.gpkg"

THRESHOLD="${THRESHOLD:-0.8}"
BUFFER_M="${BUFFER_M:-600}"
MIN_AREA_PX="${MIN_AREA_PX:-20}"
MAX_AREA_PX="${MAX_AREA_PX:-2000}"
MIN_CIRCULARITY="${MIN_CIRCULARITY:-0.35}"

echo "Start: $(date)"
echo "Params: threshold=$THRESHOLD buffer_m=$BUFFER_M min_area=$MIN_AREA_PX max_area=$MAX_AREA_PX min_circularity=$MIN_CIRCULARITY"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m cpis.semseg.mine_hard_negatives \
  --prob-dir "$PROB_DIR" \
  --tile-dir data/imagery/2021_ssa \
  --tile-list data/splits/train_tiles_2021.txt \
  --labels data/anchors_prepared/anchor_truth/anchors/anchor_2021_normalized.gpkg \
  --output "$OUT" \
  --threshold "$THRESHOLD" \
  --buffer-m "$BUFFER_M" \
  --min-area-px "$MIN_AREA_PX" \
  --max-area-px "$MAX_AREA_PX" \
  --min-circularity "$MIN_CIRCULARITY"

echo "End: $(date)"
echo "Hard negatives: $OUT"
