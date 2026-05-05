"""Threshold probability maps, polygonize candidates, and evaluate them."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from tqdm import tqdm

from cpis.eval.eval_semseg import match_predictions

logger = logging.getLogger(__name__)


def parse_thresholds(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, stop, step = [float(v) for v in part.split(":")]
            x = start
            while x <= stop + (step / 2):
                values.append(round(float(x), 6))
                x += step
        else:
            values.append(float(part))
    if not values:
        raise ValueError("at least one threshold is required")
    return sorted(set(values))


def prob_to_pivots(
    prob_path: Path,
    threshold: float,
    min_area_px: int,
    max_area_px: int,
    min_circularity: float,
) -> gpd.GeoDataFrame:
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
        if area_px < min_area_px or area_px > max_area_px:
            continue
        circularity = (4 * np.pi * poly.area) / (poly.length ** 2 + 1e-12)
        if circularity < min_circularity:
            continue
        records.append({
            "geometry": poly,
            "threshold": float(threshold),
            "area_px": round(float(area_px), 1),
            "circularity": round(float(circularity), 3),
            "tile": prob_path.stem,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=["geometry", "threshold", "area_px", "circularity", "tile"],
            crs=crs,
        )
    return gpd.GeoDataFrame(records, crs=crs)


def evaluate_predictions(
    pred: gpd.GeoDataFrame,
    labels: gpd.GeoDataFrame,
    iou_threshold: float,
    clip_label_bounds: bool,
) -> dict:
    if pred.crs != labels.crs:
        pred = pred.to_crs(labels.crs)
    if clip_label_bounds and len(pred):
        bounds = labels.total_bounds
        pred = pred.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()

    matched_pred, matched_labels, ious = match_predictions(pred, labels, iou_threshold)
    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(labels) - len(matched_labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "num_predictions": int(len(pred)),
        "num_labels": int(len(labels)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
    }


def run_sweep(args) -> list[dict]:
    prob_dir = Path(args.prob_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = gpd.read_file(args.labels)
    prob_files = sorted(prob_dir.glob("*.tif"))
    thresholds = parse_thresholds(args.thresholds)

    logger.info(f"Labels: {len(labels)}")
    logger.info(f"Prob maps: {len(prob_files)}")
    logger.info(f"Thresholds: {thresholds}")

    results = []
    for threshold in thresholds:
        all_gdfs = []
        for prob_path in tqdm(prob_files, desc=f"thr={threshold:g}"):
            gdf = prob_to_pivots(
                prob_path,
                threshold,
                min_area_px=args.min_area_px,
                max_area_px=args.max_area_px,
                min_circularity=args.min_circularity,
            )
            if len(gdf):
                all_gdfs.append(gdf)

        if all_gdfs:
            pred = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
        else:
            pred = gpd.GeoDataFrame(
                columns=["geometry", "threshold", "area_px", "circularity", "tile"],
                crs=labels.crs,
            )

        pred_path = output_dir / f"pivots_thr{int(round(threshold * 100)):03d}.gpkg"
        pred.to_file(pred_path, driver="GPKG")
        metrics = evaluate_predictions(pred, labels, args.iou_threshold, args.clip_label_bounds)
        row = {
            "label": args.label,
            "threshold": float(threshold),
            "pred_path": str(pred_path),
            **metrics,
        }
        results.append(row)
        logger.info(
            f"thr={threshold:g}: pred={metrics['num_predictions']} TP={metrics['tp']} "
            f"FP={metrics['fp']} FN={metrics['fn']} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'thr':>6} {'n_pred':>8} {'TP':>6} {'FP':>7} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7}")
    for row in results:
        print(
            f"{row['threshold']:>6.2f} {row['num_predictions']:>8,} {row['tp']:>6} "
            f"{row['fp']:>7} {row['fn']:>6} {row['precision']:>7.4f} "
            f"{row['recall']:>7.4f} {row['f1']:>7.4f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep prob-map thresholds and evaluate polygons")
    parser.add_argument("--prob-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--min-area-px", type=int, default=30)
    parser.add_argument("--max-area-px", type=int, default=2000)
    parser.add_argument("--min-circularity", type=float, default=0.4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--clip-label-bounds", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_sweep(args)


if __name__ == "__main__":
    main()
