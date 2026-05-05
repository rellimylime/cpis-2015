#!/bin/bash -l
#SBATCH --job-name=refresh_2015_gee
#SBATCH --output=/home/ermiller/cpis-2015/outputs/gee/refresh_2015_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/gee/refresh_2015_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00

# Refresh 2015 GEE export manifest status from the batch API.

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
OLD_REPO="/home/ermiller/global_cpis_codes"
NEW_REPO="/home/ermiller/cpis-2015"
RUN_DIR="${NEW_REPO}/runs/export/2015_stats_v1"

echo "Host: $(hostname)"
echo "Start: $(date)"

cd "${NEW_REPO}"
export PYTHONPATH="${OLD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON_BIN}" -m cpis.cli.main gee export-year \
  --year 2015 \
  --region "${NEW_REPO}/data/regions/SSA_Arid_by_Country.shp" \
  --feature-contract stats_v1 \
  --run-dir "${RUN_DIR}" \
  --refresh-status

echo "End: $(date)"
echo "Manifest: ${RUN_DIR}/export_manifest.json"
