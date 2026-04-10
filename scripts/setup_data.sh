#!/usr/bin/env bash
# Download anchor shapefiles and set up data directories.
# GEE imagery must be exported separately via src/cpis/gee/export_year.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_DIR/data"

echo "=== cpis-2015 data setup ==="

# --- Anchor shapefiles ---
# These should be copied from the lab's shared storage or the old repo.
# Update these paths to match your environment.
ANCHOR_SOURCE="${ANCHOR_SOURCE:-/path/to/shared/Africa_CPIS-shp}"

if [ -d "$ANCHOR_SOURCE" ]; then
    echo "Copying anchor shapefiles from $ANCHOR_SOURCE"
    cp "$ANCHOR_SOURCE"/Africa_CPIS_2000.* "$DATA_DIR/anchors/"
    cp "$ANCHOR_SOURCE"/Africa_CPIS_2021.* "$DATA_DIR/anchors/"
else
    echo "WARNING: Anchor source not found at $ANCHOR_SOURCE"
    echo "  Set ANCHOR_SOURCE=/path/to/Africa_CPIS-shp and re-run, or copy manually:"
    echo "    cp /path/to/Africa_CPIS_2000.* data/anchors/"
    echo "    cp /path/to/Africa_CPIS_2021.* data/anchors/"
fi

# --- Region boundary ---
REGION_SOURCE="${REGION_SOURCE:-/path/to/shared/SSA_Arid_by_Country-shp}"

if [ -d "$REGION_SOURCE" ]; then
    echo "Copying region boundary from $REGION_SOURCE"
    cp "$REGION_SOURCE"/SSA_Arid_by_Country.* "$DATA_DIR/regions/"
else
    echo "WARNING: Region source not found at $REGION_SOURCE"
    echo "  Set REGION_SOURCE=/path/to/SSA_Arid_by_Country-shp and re-run"
fi

# --- GEE exports ---
echo ""
echo "Imagery must be exported from GEE. Run:"
echo "  python -m cpis.gee.export_year --year 2015 --output data/imagery/2015_ssa/"
echo "  python -m cpis.gee.export_year --year 2021 --output data/imagery/2021_ssa/"
echo ""

# --- Gold holdout ---
GOLD_SOURCE="${GOLD_SOURCE:-}"
if [ -n "$GOLD_SOURCE" ] && [ -d "$GOLD_SOURCE" ]; then
    echo "Copying gold holdout from $GOLD_SOURCE"
    cp -r "$GOLD_SOURCE"/* "$DATA_DIR/gold/"
else
    echo "NOTE: Gold holdout labels should be placed in data/gold/"
    echo "  These are manually reviewed 2015 tiles withheld from training."
fi

echo ""
echo "Setup complete. Check data/ for missing files before training."
ls -la "$DATA_DIR"/anchors/ "$DATA_DIR"/regions/ 2>/dev/null || true
