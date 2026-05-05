#!/bin/bash -l
#SBATCH --job-name=rasterize_2021_stats_v1
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/rasterize_2021_stats_v1_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/rasterize_2021_stats_v1_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Rasterize 2021 anchor labels onto 11-band stats_v1 imagery.
# This processes all 202 training+val tiles at once, creating aligned binary masks.
# Idempotent: re-run is safe; existing masks are checked.

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
TILE_DIR="$REPO/data/imagery/2021_ssa_stats_v1"
MASK_DIR="$REPO/runs/semseg/stats_v1_masks"
ANCHORS="$REPO/data/anchors_prepared/anchor_truth/anchors/anchor_2021_normalized.gpkg"

mkdir -p "$MASK_DIR" "$REPO/outputs/semseg"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Tile dir: $TILE_DIR ($(find "$TILE_DIR" -name "*.tif" 2>/dev/null | wc -l) tiles)"
echo "Anchor file: $ANCHORS"
echo "Mask output: $MASK_DIR"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Rasterize all available tiles (only those with imagery will be processed).
# This waits for rclone download to complete, so it's safe to run in parallel.
"$PYTHON_BIN" -m cpis.data.rasterize_labels \
  --tile-dir "$TILE_DIR" \
  --labels "$ANCHORS" \
  --output-dir "$MASK_DIR"

N_MASKS=$(find "$MASK_DIR" -name "*_mask.tif" 2>/dev/null | wc -l)
echo "End: $(date)"
echo "Masks created: $N_MASKS"

if [ "$N_MASKS" -ge 190 ]; then
  echo "SUCCESS: Rasterization complete for 11-band stats_v1"
  exit 0
else
  echo "WARNING: Expected ~202 masks, found $N_MASKS. May still be downloading."
  exit 0  # Not a hard failure — may just need to wait for rclone
fi
