"""Evaluate predicted pivot polygons against label polygons."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from cpis.eval.eval_semseg import match_predictions

logger = logging.getLogger(__name__)


def parse_thresholds(text: str | None) -> list[float | None]:
    if not text:
        return [None]
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [float(v) for v in part.split(":")]
            if len(bits) != 3:
                raise ValueError("range thresholds must be start:stop:step")
            start, stop, step = bits
            x = start
            while x <= stop + (step / 2):
                values.append(round(float(x), 6))
                x += step
        else:
            values.append(float(part))
    return sorted(set(values)) if values else [None]


def load_tile_extent(tile_dir: Path, tile_list: Path | None, crs) -> gpd.GeoDataFrame | None:
    if tile_list is None:
        return None
    with open(tile_list) as f:
        names = [line.strip() for line in f if line.strip()]
    records = []
    for name in names:
        tile_path = tile_dir / f"{name}.tif"
        if not tile_path.exists():
            logger.warning(f"missing tile for eval extent: {tile_path}")
            continue
        with rasterio.open(tile_path) as src:
            records.append({"geometry": box(*src.bounds)})
            tile_crs = src.crs
    if not records:
        return None
    gdf = gpd.GeoDataFrame(records, crs=tile_crs)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def filter_to_extent(gdf: gpd.GeoDataFrame, extent: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    if extent is None:
        return gdf
    if gdf.crs != extent.crs:
        gdf = gdf.to_crs(extent.crs)
    sidx = extent.sindex
    keep = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            keep.append(False)
            continue
        cands = list(sidx.intersection(geom.bounds))
        keep.append(any(geom.intersects(extent.geometry.iloc[i]) for i in cands))
    return gdf[np.array(keep, dtype=bool)].copy()


def evaluate(
    pred: gpd.GeoDataFrame,
    labels: gpd.GeoDataFrame,
    iou_threshold: float,
    score_threshold: float | None,
) -> dict:
    if score_threshold is not None and "score" in pred.columns:
        pred = pred[pred["score"] >= score_threshold].copy()
    matched_pred, matched_labels, ious = match_predictions(pred, labels, iou_threshold)
    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(labels) - len(matched_labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate pivot polygon predictions")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-thresholds", default=None,
                        help="Comma list or start:stop:step range; omitted means all predictions")
    parser.add_argument("--tile-dir", default=None,
                        help="Optional tile directory used with --tile-list to define eval extent")
    parser.add_argument("--tile-list", default=None,
                        help="Optional tile list used with --tile-dir to restrict labels and predictions")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pred = gpd.read_file(args.pred)
    labels = gpd.read_file(args.labels)
    if pred.crs != labels.crs:
        pred = pred.to_crs(labels.crs)

    extent = None
    if args.tile_dir and args.tile_list:
        extent = load_tile_extent(Path(args.tile_dir), Path(args.tile_list), labels.crs)
        pred = filter_to_extent(pred, extent)
        labels = filter_to_extent(labels, extent)

    logger.info(f"Predictions in eval extent: {len(pred)}")
    logger.info(f"Labels in eval extent: {len(labels)}")

    results = []
    for thr in parse_thresholds(args.score_thresholds):
        row = evaluate(pred, labels, args.iou_threshold, thr)
        row["label"] = args.label
        results.append(row)
        logger.info(
            f"score>={thr}: pred={row['num_predictions']} TP={row['tp']} FP={row['fp']} "
            f"FN={row['fn']} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results if len(results) > 1 else results[0], f, indent=2)

    print(f"{'score':>8} {'n_pred':>8} {'TP':>6} {'FP':>7} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7}")
    for row in results:
        score = "all" if row["score_threshold"] is None else f"{row['score_threshold']:.3f}"
        print(
            f"{score:>8} {row['num_predictions']:>8,} {row['tp']:>6} {row['fp']:>7} "
            f"{row['fn']:>6} {row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f}"
        )


if __name__ == "__main__":
    main()
