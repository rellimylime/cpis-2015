#!/bin/bash -l
#SBATCH --job-name=cpis_sweep_2021v1
#SBATCH --output=/home/ermiller/cpis-2015/outputs/semseg/sweep_2021v1_%j.log
#SBATCH --error=/home/ermiller/cpis-2015/outputs/semseg/sweep_2021v1_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

PYTHON_BIN="/home/ermiller/.conda/envs/cpi_fix/bin/python"
REPO="/home/ermiller/cpis-2015"
PROB_DIR="$REPO/outputs/semseg/probs_2021_v1"
GOLD="$REPO/data/gold/2015_gold_val_v2_holdout.gpkg"
SWEEP_OUT="$REPO/outputs/semseg/sweep_2021_v1"

mkdir -p "$SWEEP_OUT"

echo "Start: $(date)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - "$PROB_DIR" "$GOLD" "$SWEEP_OUT" << 'PYEOF'
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROB_DIR = Path(sys.argv[1])
GOLD_PATH = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_AREA_PX = 30
MAX_AREA_PX = 2000
MIN_CIRC = 0.4


def prob_to_pivots(prob_path: Path, threshold: float) -> gpd.GeoDataFrame:
    with rasterio.open(prob_path) as src:
        prob = src.read(1)
        transform = src.transform
        crs = src.crs
        res = src.res[0]

    binary = (prob >= threshold).astype(np.uint8)
    records = []
    for geom_dict, val in rio_shapes(binary, mask=binary, transform=transform):
        if val != 1:
            continue
        poly = shape(geom_dict)
        area_px = poly.area / (res ** 2)
        if area_px < MIN_AREA_PX or area_px > MAX_AREA_PX:
            continue
        circ = (4 * np.pi * poly.area) / (poly.length ** 2 + 1e-12)
        if circ < MIN_CIRC:
            continue
        records.append({"geometry": poly, "tile": prob_path.stem})

    if not records:
        return gpd.GeoDataFrame(columns=["geometry", "tile"], crs=crs)
    return gpd.GeoDataFrame(records, crs=crs)


def iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def eval_against_gold(pred: gpd.GeoDataFrame, gold: gpd.GeoDataFrame, iou_thr=0.5):
    if len(pred) == 0:
        return {"tp": 0, "fp": 0, "fn": len(gold), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if pred.crs != gold.crs:
        pred = pred.to_crs(gold.crs)

    gold_bounds = gold.total_bounds
    pred = pred.cx[gold_bounds[0]:gold_bounds[2], gold_bounds[1]:gold_bounds[3]]
    sindex = gold.sindex
    matched_pred, matched_gold, ious = set(), set(), []
    for pi, geom in enumerate(pred.geometry):
        cands = list(sindex.intersection(geom.bounds))
        best_iou, best_gi = 0.0, -1
        for gi in cands:
            if gi in matched_gold:
                continue
            v = iou(geom, gold.geometry.iloc[gi])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_thr and best_gi >= 0:
            matched_pred.add(pi)
            matched_gold.add(best_gi)
            ious.append(best_iou)

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gold) - len(matched_gold)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
    }


gold = gpd.read_file(GOLD_PATH)
logger.info(f"Gold: {len(gold)} pivots")

prob_files = sorted(PROB_DIR.glob("*.tif"))
logger.info(f"Prob maps: {len(prob_files)} tiles")

results = []
for thr in THRESHOLDS:
    logger.info(f"\n=== Threshold {thr} ===")
    all_gdfs = []
    for pf in tqdm(prob_files, desc=f"thr={thr}"):
        gdf = prob_to_pivots(pf, thr)
        if len(gdf) > 0:
            all_gdfs.append(gdf)
    if all_gdfs:
        pred = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
    else:
        pred = gpd.GeoDataFrame(columns=["geometry", "tile"], crs=gold.crs)
    pred.to_file(OUT_DIR / f"pivots_thr{int(thr * 10):02d}.gpkg", driver="GPKG")
    metrics = eval_against_gold(pred, gold)
    row = {"threshold": thr, "n_pred_total": len(pred), **metrics}
    results.append(row)
    logger.info(
        f"  Pred={len(pred):,}  TP={metrics['tp']}  FP={metrics['fp']}  "
        f"FN={metrics['fn']}  P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
    )

with open(OUT_DIR / "sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== THRESHOLD SWEEP SUMMARY ===")
print(f"{'thr':>5} {'n_pred':>8} {'TP':>6} {'FP':>7} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7}")
for r in results:
    print(
        f"{r['threshold']:>5.1f} {r['n_pred_total']:>8,} {r['tp']:>6} {r['fp']:>7} "
        f"{r['fn']:>6} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}"
    )
PYEOF

echo ""
echo "End: $(date)"
echo "Results: $SWEEP_OUT/sweep_results.json"
