"""Evaluate predicted pivot polygons against gold holdout (IoU-based matching)."""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def iou(geom_a, geom_b) -> float:
    intersection = geom_a.intersection(geom_b).area
    union = geom_a.union(geom_b).area
    return intersection / union if union > 0 else 0.0


def match_predictions(pred: gpd.GeoDataFrame, gold: gpd.GeoDataFrame,
                      iou_threshold: float = 0.5):
    """Greedy 1-to-1 IoU matching. Each gold and pred polygon matched at most once."""
    if pred.crs != gold.crs:
        pred = pred.to_crs(gold.crs)

    gold_sindex = gold.sindex
    matched_pred = set()
    matched_gold = set()
    match_ious = []

    for pi, pred_geom in enumerate(pred.geometry):
        candidates = list(gold_sindex.intersection(pred_geom.bounds))
        best_iou, best_gi = 0.0, -1
        for gi in candidates:
            if gi in matched_gold:
                continue
            v = iou(pred_geom, gold.geometry.iloc[gi])
            if v > best_iou:
                best_iou, best_gi = v, gi

        if best_iou >= iou_threshold and best_gi >= 0:
            matched_pred.add(pi)
            matched_gold.add(best_gi)
            match_ious.append(best_iou)

    return matched_pred, matched_gold, match_ious


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semseg pivot predictions against gold holdout")
    parser.add_argument("--pred", required=True, help="Predicted pivots GeoPackage")
    parser.add_argument("--gold", required=True, help="Gold holdout GeoPackage")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    pred = gpd.read_file(args.pred)
    gold = gpd.read_file(args.gold)

    logger.info(f"Predictions (total): {len(pred)}")
    logger.info(f"Gold: {len(gold)}")

    # Clip predictions to gold bounding box so precision is meaningful.
    # Without this, predictions outside the gold region are all FP.
    if pred.crs != gold.crs:
        pred = pred.to_crs(gold.crs)
    gold_bounds = gold.total_bounds  # (minx, miny, maxx, maxy)
    pred = pred.cx[gold_bounds[0]:gold_bounds[2], gold_bounds[1]:gold_bounds[3]]
    logger.info(f"Predictions (clipped to gold bbox): {len(pred)}")

    matched_pred, matched_gold, ious = match_predictions(pred, gold, args.iou_threshold)

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gold) - len(matched_gold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        "label": args.label,
        "iou_threshold": args.iou_threshold,
        "num_predictions": len(pred),
        "num_gold": len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou_of_matches": round(float(np.mean(ious)) if ious else 0.0, 4),
    }

    logger.info(f"TP={tp}  FP={fp}  FN={fn}")
    logger.info(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
