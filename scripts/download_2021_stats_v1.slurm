#!/bin/bash -l
#SBATCH --job-name=rclone_2021_stats_v1
#SBATCH --output=/home/ermiller/cpis-2015/outputs/gee/rclone_2021_stats_v1_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/gee/rclone_2021_stats_v1_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Download 2021 stats_v1 GeoTIFFs from CPIS_2021_LANDSAT_STATS_V1 Google Drive folder.
# Rclone will skip files that already exist locally, so this is safe to re-run.

set -euo pipefail

DEST="/home/ermiller/cpis-2015/data/imagery/2021_ssa_stats_v1"
mkdir -p "$DEST"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Destination: $DEST"
echo "Pre-download count: $(find "$DEST" -name "*.tif" 2>/dev/null | wc -l) tiles"

# Use conda environment with rclone
source /home/ermiller/.conda/etc/profile.d/conda.sh
conda activate cpi_fix

rclone copy \
  gdrive:CPIS_2021_LANDSAT_STATS_V1 \
  "$DEST" \
  --include="*.tif" \
  --exclude="*_metadata.json" \
  -P --stats 30s --log-file=/home/ermiller/cpis-2015/outputs/gee/rclone_2021_stats_v1_detail.log

POST_COUNT=$(find "$DEST" -name "*.tif" 2>/dev/null | wc -l)
echo "Post-download count: $POST_COUNT tiles"
echo "End: $(date)"

if [ "$POST_COUNT" -ge 202 ]; then
  echo "SUCCESS: All 202 needed 2021 tiles downloaded"
  exit 0
else
  echo "WARNING: Expected 202 tiles, found $POST_COUNT. Check rclone log."
  exit 1
fi
