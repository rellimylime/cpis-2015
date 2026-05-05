#!/bin/bash -l
#SBATCH --job-name=cpis_gee_export
#SBATCH --output=/home/ermiller/cpis-2015/outputs/gee/export_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/gee/export_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00

# Submit GEE export tasks for one year of Landsat composites with the 11-band
# stats_v1 contract (B/G/R/NIR/SWIR1/SWIR2 medians + NDVI p10/p50/p90/amp + NDWI p50).
#
# This script ONLY submits tasks to Google's servers. Files do NOT appear locally.
# Full workflow (orchestrated manually by the user):
#
#   1. YEAR=2021 MAX_TILES=20 sbatch scripts/export_landsat_stats_v1.slurm   # smoke
#   2. (wait ~30-60min for GEE) check Drive folder, then:
#      python -m cpis.cli.main gee export-year --year 2021 --refresh-status \
#        --run-dir ~/cpis-2015/runs/export/2021_stats_v1
#      (run from ~/global_cpis_codes — that's where the cpis.cli.main CLI lives)
#   3. rclone copy gdrive:CPIS_2021_LANDSAT_STATS_V1 ~/cpis-2015/data/imagery/2021_ssa_stats_v1/
#   4. Validate with rasterio (count == 11, descriptions match stats_v1).
#   5. Once smoke validated: re-run with MAX_TILES=0 for the full re-export.
#
# Env vars:
#   YEAR          required; 2021 or 2015
#   MAX_TILES     default 0 (no cap). Set to 20 for smoke.
#   TILE_ID_LIST  optional path to a text file of tile IDs to export. Format:
#                 one ID per line, either bare ('tile_000003') or full
#                 ('cpis_landsat_2021_tile_000003'). When set, only those tiles
#                 are submitted to GEE — the manifest still contains the full
#                 grid, but submissions are filtered. Use this for the full
#                 re-export to avoid wasting GEE quota on tiles we don't use.
#   REBUILD_TILES  set to 1 to regenerate the full tile grid in the manifest.
#                 Required when expanding from a smoke manifest (e.g., 20 tiles)
#                 to the full grid (~509 tiles). Combine with SKIP_LOCAL_DIR so
#                 already-downloaded tiles aren't re-submitted.
#   SKIP_LOCAL_DIR  optional path to a local directory of already-exported TIFFs.
#                 Tiles whose IDs match files in this dir are marked
#                 'local_exists=True' and skipped during submission.
#   RUN_SUFFIX    default "stats_v1". Used in run-dir and Drive folder name.
#   DRIVE_FOLDER  default "CPIS_${YEAR}_LANDSAT_${RUN_SUFFIX^^}"
#
# Why a separate run-dir: the existing 4-band manifest at
# runs/export/{year}/export_manifest.json must NOT be overwritten — we'd lose
# state on the existing tiles. New manifest goes to runs/export/{year}_stats_v1/.

set -euo pipefail

if [ -z "${YEAR:-}" ]; then
  echo "ERROR: YEAR env var required (e.g., YEAR=2021 sbatch ...)" >&2
  exit 1
fi

MAX_TILES="${MAX_TILES:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-stats_v1}"
DRIVE_FOLDER="${DRIVE_FOLDER:-CPIS_${YEAR}_LANDSAT_${RUN_SUFFIX^^}}"
PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
OLD_REPO="/home/ermiller/global_cpis_codes"
NEW_REPO="/home/ermiller/cpis-2015"
RUN_DIR="${NEW_REPO}/runs/export/${YEAR}_${RUN_SUFFIX}"

mkdir -p "${NEW_REPO}/outputs/gee" "${RUN_DIR}"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "YEAR=${YEAR} MAX_TILES=${MAX_TILES} DRIVE_FOLDER=${DRIVE_FOLDER}"
echo "RUN_DIR=${RUN_DIR}"

# IMPORTANT: do NOT cd into ${OLD_REPO}. That directory contains a top-level
# cpis.py wrapper that shadows the cpis package when CWD is auto-prepended to
# sys.path, breaking `python -m cpis.cli.main`. Run from a neutral cwd and rely
# on PYTHONPATH to locate the package under ${OLD_REPO}/src.
cd "${NEW_REPO}"
export PYTHONPATH="${OLD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

# Region file: same one used for the original 4-band 2021 export (per the
# existing manifest at runs/export/2021/export_manifest.json). The CLI default
# (arid_ssa.geojson) does not exist in this checkout. The shapefile + sidecars
# now live in this repo at data/regions/ — no longer depends on the old repo.
REGION="${NEW_REPO}/data/regions/SSA_Arid_by_Country.shp"

EXTRA_ARGS=()
if [ -n "${TILE_ID_LIST:-}" ]; then
  EXTRA_ARGS+=(--tile-id-list "${TILE_ID_LIST}")
fi
if [ "${REBUILD_TILES:-0}" = "1" ]; then
  EXTRA_ARGS+=(--rebuild-tiles)
fi
if [ -n "${SKIP_LOCAL_DIR:-}" ]; then
  EXTRA_ARGS+=(--skip-local-dir "${SKIP_LOCAL_DIR}")
fi

"${PYTHON_BIN}" -m cpis.cli.main gee export-year \
  --year "${YEAR}" \
  --region "${REGION}" \
  --feature-contract stats_v1 \
  --tile-size-km 200.0 \
  --cloud-max 40.0 \
  --drive-folder "${DRIVE_FOLDER}" \
  --project africa-irrigation-mine \
  --run-dir "${RUN_DIR}" \
  --max-tiles "${MAX_TILES}" \
  --start-tasks \
  --resume \
  "${EXTRA_ARGS[@]}"

echo "End: $(date)"
echo "Manifest: ${RUN_DIR}/export_manifest.json"
echo "Drive folder: ${DRIVE_FOLDER}"
echo "Next: wait for GEE to complete (~hours), refresh status, then rclone."
