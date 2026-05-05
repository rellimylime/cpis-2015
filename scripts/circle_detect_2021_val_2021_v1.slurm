#!/bin/bash -l
#SBATCH --job-name=cpis_circle_2021val
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/circle_2021val_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/circle_2021val_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
PROB_DIR="$REPO/outputs/semseg/probs_2021_v1_val"
OUT="$REPO/outputs/semseg/circles_2021_v1_val.gpkg"
EVAL_OUT="$REPO/outputs/semseg/circles_2021_v1_val_eval.json"

RADII_PX="${RADII_PX:-7-18}"
SEED_THRESHOLD="${SEED_THRESHOLD:-0.20}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
MIN_DISK_MEAN="${MIN_DISK_MEAN:-0.20}"
MIN_DISTANCE_PX="${MIN_DISTANCE_PX:-10}"
SEED_SIGMA="${SEED_SIGMA:-2.5}"
MAX_PEAKS_PER_TILE="${MAX_PEAKS_PER_TILE:-3000}"
SEED_MODE="${SEED_MODE:-both}"
MIN_COMPONENT_AREA_PX="${MIN_COMPONENT_AREA_PX:-12}"
MAX_COMPONENT_AREA_PX="${MAX_COMPONENT_AREA_PX:-5000}"
RING_WEIGHT="${RING_WEIGHT:-0.55}"
CENTER_WEIGHT="${CENTER_WEIGHT:-0.0}"
RADIUS_SCORE_POWER="${RADIUS_SCORE_POWER:-0.5}"
NMS_OVERLAP="${NMS_OVERLAP:-0.25}"

mkdir -p "$REPO/outputs/semseg"

echo "Start: $(date)"
echo "Params: radii=$RADII_PX seed=$SEED_THRESHOLD score=$SCORE_THRESHOLD disk=$MIN_DISK_MEAN max_peaks=$MAX_PEAKS_PER_TILE seed_mode=$SEED_MODE"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m cpis.semseg.circle_detect \
  --prob-dir "$PROB_DIR" \
  --output "$OUT" \
  --radii-px "$RADII_PX" \
  --seed-threshold "$SEED_THRESHOLD" \
  --score-threshold "$SCORE_THRESHOLD" \
  --min-disk-mean "$MIN_DISK_MEAN" \
  --min-distance-px "$MIN_DISTANCE_PX" \
  --seed-sigma "$SEED_SIGMA" \
  --max-peaks-per-tile "$MAX_PEAKS_PER_TILE" \
  --seed-mode "$SEED_MODE" \
  --min-component-area-px "$MIN_COMPONENT_AREA_PX" \
  --max-component-area-px "$MAX_COMPONENT_AREA_PX" \
  --ring-weight "$RING_WEIGHT" \
  --center-weight "$CENTER_WEIGHT" \
  --radius-score-power "$RADIUS_SCORE_POWER" \
  --nms-overlap "$NMS_OVERLAP"

"$PYTHON_BIN" -m cpis.eval.eval_polygons \
  --pred "$OUT" \
  --labels data/anchors_prepared/anchor_truth/anchors/anchor_2021_normalized.gpkg \
  --output "$EVAL_OUT" \
  --tile-dir data/imagery/2021_ssa \
  --tile-list data/splits/2021_val_tiles_landsat.txt \
  --score-thresholds 0.05:0.80:0.05 \
  --iou-threshold 0.5 \
  --label "circle_2021_val"

echo "End: $(date)"
echo "Predictions: $OUT"
echo "Eval: $EVAL_OUT"
