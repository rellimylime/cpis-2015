#!/usr/bin/env bash
# Run from inside cpis-2015/
# Assumes global_cpis_codes is a sibling directory.
set -euo pipefail

OLD="../global_cpis_codes"

# --- mm_overrides (MMDetection customizations) ---
cp -r "$OLD"/mm_scripts/* src/cpis/instseg/mm_overrides/

# --- Anchor shapefiles ---
cp "$OLD"/Africa_CPIS-shp/Africa_CPIS_2000.* data/anchors/ 2>/dev/null || \
  echo "WARNING: Africa_CPIS-shp not found in old repo — may be gitignored. Copy manually."
cp "$OLD"/Africa_CPIS-shp/Africa_CPIS_2021.* data/anchors/ 2>/dev/null || true

# --- Region boundary ---
cp "$OLD"/SSA_Arid_by_Country-shp/SSA_Arid_by_Country.* data/regions/

# --- Gold holdout splits ---
SPLITS="$OLD/runs/new_method/rse2023_2015_v1/splits"
if [ -d "$SPLITS" ]; then
  cp -r "$SPLITS"/* data/gold/
else
  echo "WARNING: splits dir not found at $SPLITS — may be gitignored/local-only"
fi

echo "Done. Check data/ for completeness:"
ls data/anchors/ data/regions/ data/gold/ 2>/dev/null
