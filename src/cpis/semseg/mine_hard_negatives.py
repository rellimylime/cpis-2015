"""Mine hard-negative candidate centers from confident probability-map detections."""

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
from shapely.geometry import Point, shape
from tqdm import tqdm

logger = logging.getLogger(__name__)

NEGATIVE_CRS = "EPSG:6933"


def candidate_components(
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
        centroid = poly.centroid
        records.append({
            "geometry": Point(float(centroid.x), float(centroid.y)),
            "source_geom": poly,
            "tile": prob_path.stem,
            "area_px": round(float(area_px), 1),
            "circularity": round(float(circularity), 3),
            "threshold": float(threshold),
        })
    if not records:
        return gpd.GeoDataFrame(columns=["geometry", "source_geom", "tile", "area_px", "circularity", "threshold"], crs=crs)
    return gpd.GeoDataFrame(records, crs=crs)


def mine_hard_negatives(args) -> gpd.GeoDataFrame:
    prob_dir = Path(args.prob_dir)
    tile_dir = Path(args.tile_dir)
    output_path = Path(args.output)

    labels = gpd.read_file(args.labels)
    if labels.crs is None:
        raise ValueError("labels layer is missing CRS")
    labels_proj = labels.to_crs(NEGATIVE_CRS)
    label_buffers = labels_proj.copy()
    label_buffers["geometry"] = label_buffers.geometry.buffer(args.buffer_m)

    if args.tile_list:
        with open(args.tile_list) as f:
            tile_names = {line.strip() for line in f if line.strip()}
    else:
        tile_names = None

    prob_files = sorted(prob_dir.glob("*.tif"))
    if tile_names is not None:
        prob_files = [p for p in prob_files if p.stem in tile_names]

    all_candidates = []
    for prob_path in tqdm(prob_files, desc="mine-hard-negatives"):
        tile_path = tile_dir / f"{prob_path.stem}.tif"
        if not tile_path.exists():
            logger.warning(f"missing tile for {prob_path.stem}: {tile_path}")
            continue
        gdf = candidate_components(
            prob_path,
            threshold=args.threshold,
            min_area_px=args.min_area_px,
            max_area_px=args.max_area_px,
            min_circularity=args.min_circularity,
        )
        if len(gdf):
            all_candidates.append(gdf)

    if not all_candidates:
        mined = gpd.GeoDataFrame(
            columns=["geometry", "tile", "area_px", "circularity", "threshold", "distance_to_anchor_m"],
            crs="EPSG:4326",
        )
    else:
        merged = gpd.GeoDataFrame(pd.concat(all_candidates, ignore_index=True), crs=all_candidates[0].crs)
        merged_proj = merged.to_crs(NEGATIVE_CRS)
        nearest = merged_proj.geometry.apply(lambda geom: labels_proj.distance(geom).min() if len(labels_proj) else np.inf)
        merged_proj["distance_to_anchor_m"] = nearest.astype(float)

        keep = np.ones(len(merged_proj), dtype=bool)
        buffer_sidx = label_buffers.sindex
        for i, geom in enumerate(merged_proj.geometry):
            candidate_ids = list(buffer_sidx.intersection(geom.bounds))
            if any(geom.intersects(label_buffers.geometry.iloc[j]) for j in candidate_ids):
                keep[i] = False
        mined = merged_proj[keep].copy()
        mined = mined.drop(columns=["source_geom"], errors="ignore").to_crs("EPSG:4326")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mined.to_file(output_path, driver="GPKG")

    summary = {
        "num_hard_negatives": int(len(mined)),
        "threshold": float(args.threshold),
        "buffer_m": float(args.buffer_m),
        "median_area_px": round(float(mined["area_px"].median()), 1) if len(mined) else 0.0,
        "median_distance_to_anchor_m": round(float(mined["distance_to_anchor_m"].median()), 1) if len(mined) else 0.0,
    }
    with open(output_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {len(mined)} hard negatives to {output_path}")
    logger.info(f"Summary: {summary}")
    return mined


def main():
    parser = argparse.ArgumentParser(description="Mine hard-negative centers from prob maps")
    parser.add_argument("--prob-dir", required=True)
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile-list", default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--buffer-m", type=float, default=600.0)
    parser.add_argument("--min-area-px", type=int, default=20)
    parser.add_argument("--max-area-px", type=int, default=2000)
    parser.add_argument("--min-circularity", type=float, default=0.35)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    mine_hard_negatives(args)


if __name__ == "__main__":
    main()
