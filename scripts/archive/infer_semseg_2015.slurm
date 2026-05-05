#!/bin/bash -l
#SBATCH --job-name=cpis_semseg_infer
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/infer_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/infer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
CHECKPOINT="$REPO/runs/semseg/2015_v2/best.pth"
BAND_STATS="$REPO/runs/semseg/2015_v2/band_stats.json"
PROB_DIR="$REPO/outputs/semseg/probs_v2"

mkdir -p "$REPO/outputs/semseg" "$PROB_DIR"

echo "Host: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# --- Sanity check: run 3 tiles at threshold=0.5, verify at least one has >0 pivot pixels ---
echo ""
echo "=== Sanity check (3 tiles) ==="
SANITY_TILES=(
  data/imagery/2015_ssa/cpis_landsat_2015_tile_000023.tif
  data/imagery/2015_ssa/cpis_landsat_2015_tile_000024.tif
  data/imagery/2015_ssa/cpis_landsat_2015_tile_000034.tif
)
SANITY_TOTAL=0
for TILE in "${SANITY_TILES[@]}"; do
  N=$("$PYTHON_BIN" - "$TILE" "$CHECKPOINT" "$BAND_STATS" << 'PYEOF'
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/home/ermiller/cpis-2015/src")
from cpis.semseg.infer import load_model, infer_tile
tile, ckpt, stats = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
device = torch.device("cuda")
model = load_model(ckpt, 32, device)
with open(stats) as f:
    s = json.load(f)
mean = np.array(s["mean"], dtype=np.float32)
std  = np.array(s["std"],  dtype=np.float32)
n = infer_tile(model, tile, Path("/tmp/_sanity.tif"), device, mean=mean, std=std, threshold=0.5)
print(n)
PYEOF
  )
  echo "  ${TILE##*/}: $N pivot pixels"
  SANITY_TOTAL=$((SANITY_TOTAL + N))
done

if [ "$SANITY_TOTAL" -eq 0 ]; then
  echo "ERROR: all 3 sanity tiles have 0 pivot pixels — model may be collapsed. Aborting."
  exit 1
fi
echo "Sanity check passed (total=$SANITY_TOTAL). Proceeding to full inference."

# --- Full inference: save float32 prob maps for offline threshold sweep ---
echo ""
echo "=== Full inference (saving prob maps to $PROB_DIR) ==="
"$PYTHON_BIN" -m cpis.semseg.infer \
  --tile-dir data/imagery/2015_ssa \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$PROB_DIR" \
  --band-stats "$BAND_STATS" \
  --patch-size 256 \
  --overlap 32 \
  --threshold 0.5 \
  --save-probs

echo ""
echo "End: $(date)"
echo "Prob maps: $PROB_DIR"
echo "Next: run scripts/threshold_sweep.slurm to test thresholds and eval against gold"
