"""Detect center-pivot circle proposals from semantic-segmentation probability maps."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import center_of_mass, distance_transform_edt, find_objects, gaussian_filter, label
from shapely.geometry import Polygon
from skimage.feature import peak_local_max
from tqdm import tqdm

from cpis.geo_utils import circle_overlap_ratio, circle_polygon_wgs84, pixel_size_m

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RingOffsets:
    radius_px: int
    disk_dy: np.ndarray
    disk_dx: np.ndarray
    ring_dy: np.ndarray
    ring_dx: np.ndarray


@dataclass
class CircleProposal:
    row: int
    col: int
    radius_px: int
    score: float
    disk_mean: float
    ring_mean: float
    center_prob: float
    tile: str


def build_ring_offsets(
    radii_px: list[int],
    ring_gap_px: int,
    ring_width_px: int,
) -> dict[int, RingOffsets]:
    offsets = {}
    for radius_px in radii_px:
        outer = int(math.ceil(radius_px + ring_gap_px + ring_width_px))
        yy, xx = np.mgrid[-outer:outer + 1, -outer:outer + 1]
        dist = np.sqrt((yy * yy) + (xx * xx))
        disk = dist <= radius_px
        ring = (dist > radius_px + ring_gap_px) & (dist <= radius_px + ring_gap_px + ring_width_px)
        offsets[radius_px] = RingOffsets(
            radius_px=radius_px,
            disk_dy=yy[disk].astype(np.int32),
            disk_dx=xx[disk].astype(np.int32),
            ring_dy=yy[ring].astype(np.int32),
            ring_dx=xx[ring].astype(np.int32),
        )
    return offsets


def mean_at_offsets(arr: np.ndarray, row: int, col: int, dy: np.ndarray, dx: np.ndarray) -> float:
    yy = row + dy
    xx = col + dx
    valid = (yy >= 0) & (yy < arr.shape[0]) & (xx >= 0) & (xx < arr.shape[1])
    if not np.any(valid):
        return 0.0
    return float(arr[yy[valid], xx[valid]].mean())


def score_seed(
    prob: np.ndarray,
    row: int,
    col: int,
    offsets_by_radius: dict[int, RingOffsets],
    ring_weight: float,
    center_weight: float,
    radius_score_power: float,
    radius_norm_px: float,
) -> CircleProposal | None:
    best: CircleProposal | None = None
    center_prob = float(prob[row, col])

    for radius_px, offsets in offsets_by_radius.items():
        disk_mean = mean_at_offsets(prob, row, col, offsets.disk_dy, offsets.disk_dx)
        ring_mean = mean_at_offsets(prob, row, col, offsets.ring_dy, offsets.ring_dx)
        score = disk_mean - (ring_weight * ring_mean)

        # Optional terms let val-tuned runs avoid the small-hotspot failure mode.
        score += center_weight * center_prob
        if radius_score_power and score > 0.0:
            score *= (float(radius_px) / radius_norm_px) ** radius_score_power

        if best is None or score > best.score:
            best = CircleProposal(
                row=row,
                col=col,
                radius_px=radius_px,
                score=float(score),
                disk_mean=float(disk_mean),
                ring_mean=float(ring_mean),
                center_prob=center_prob,
                tile="",
            )
    return best


def component_seeds(
    smooth: np.ndarray,
    threshold: float,
    border: int,
    min_area_px: int,
    max_area_px: int,
    max_components: int,
) -> np.ndarray:
    binary = smooth >= threshold
    if border > 0:
        binary[:border, :] = False
        binary[-border:, :] = False
        binary[:, :border] = False
        binary[:, -border:] = False

    labeled, n_components = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if n_components == 0:
        return np.empty((0, 2), dtype=np.int32)

    slices = find_objects(labeled)
    components = []
    for comp_id, comp_slice in enumerate(slices, start=1):
        if comp_slice is None:
            continue
        comp_mask = labeled[comp_slice] == comp_id
        area = int(comp_mask.sum())
        if area < min_area_px or area > max_area_px:
            continue
        comp_vals = smooth[comp_slice][comp_mask]
        if comp_vals.size == 0:
            continue
        components.append((float(comp_vals.max()), area, comp_id, comp_slice, comp_mask))

    components.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if max_components > 0:
        components = components[:max_components]

    seeds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, _, comp_id, comp_slice, comp_mask in components:
        row0 = comp_slice[0].start
        col0 = comp_slice[1].start
        vals = smooth[comp_slice]

        candidates: list[tuple[float, float]] = []

        cm = center_of_mass(vals, labels=(labeled[comp_slice] == comp_id), index=True)
        if np.all(np.isfinite(cm)):
            candidates.append((row0 + float(cm[0]), col0 + float(cm[1])))

        yy, xx = np.nonzero(comp_mask)
        if yy.size:
            weights = vals[comp_mask]
            weight_sum = float(weights.sum())
            if weight_sum > 0:
                candidates.append((
                    row0 + float(np.average(yy, weights=weights)),
                    col0 + float(np.average(xx, weights=weights)),
                ))
            candidates.append((row0 + float(np.mean(yy)), col0 + float(np.mean(xx))))

        dist = distance_transform_edt(comp_mask)
        max_pos = np.unravel_index(int(np.argmax(dist)), dist.shape)
        candidates.append((row0 + float(max_pos[0]), col0 + float(max_pos[1])))

        local_peak = np.unravel_index(int(np.argmax(np.where(comp_mask, vals, -np.inf))), vals.shape)
        candidates.append((row0 + float(local_peak[0]), col0 + float(local_peak[1])))

        bbox_center = (
            0.5 * (comp_slice[0].start + comp_slice[0].stop - 1),
            0.5 * (comp_slice[1].start + comp_slice[1].stop - 1),
        )
        candidates.append(bbox_center)

        for row_f, col_f in candidates:
            row = int(round(row_f))
            col = int(round(col_f))
            if row < border or row >= smooth.shape[0] - border:
                continue
            if col < border or col >= smooth.shape[1] - border:
                continue
            key = (row, col)
            if key not in seen:
                seen.add(key)
                seeds.append(key)

    if not seeds:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(seeds, dtype=np.int32)


def nms_circles(
    proposals: list[CircleProposal],
    overlap_threshold: float,
    min_center_fraction: float,
) -> list[CircleProposal]:
    kept: list[CircleProposal] = []
    for prop in sorted(proposals, key=lambda p: p.score, reverse=True):
        suppress = False
        for other in kept:
            dist = math.hypot(prop.col - other.col, prop.row - other.row)
            min_center_dist = min(prop.radius_px, other.radius_px) * min_center_fraction
            if dist < min_center_dist:
                suppress = True
                break
            overlap = circle_overlap_ratio(
                prop.col, prop.row, prop.radius_px,
                other.col, other.row, other.radius_px,
            )
            if overlap >= overlap_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(prop)
    return kept


def detect_tile(
    prob_path: Path,
    radii_px: list[int],
    seed_threshold: float,
    score_threshold: float,
    min_disk_mean: float,
    min_distance_px: int,
    seed_sigma: float,
    max_peaks: int,
    seed_mode: str,
    min_component_area_px: int,
    max_component_area_px: int,
    ring_gap_px: int,
    ring_width_px: int,
    ring_weight: float,
    center_weight: float,
    radius_score_power: float,
    nms_overlap: float,
    nms_center_fraction: float,
) -> gpd.GeoDataFrame:
    offsets_by_radius = build_ring_offsets(radii_px, ring_gap_px, ring_width_px)

    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype(np.float32)
        crs = src.crs
        transform = src.transform
        gt = src.transform.to_gdal()

    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob[prob < 0] = 0.0
    prob[prob > 1] = 1.0

    smooth = gaussian_filter(prob, sigma=seed_sigma)
    border = max(radii_px) + ring_gap_px + ring_width_px + 1
    seed_arrays = []
    if seed_mode in {"peaks", "both"}:
        seed_arrays.append(peak_local_max(
            smooth,
            min_distance=min_distance_px,
            threshold_abs=seed_threshold,
            exclude_border=border,
            num_peaks=max_peaks,
        ))
    if seed_mode in {"components", "both"}:
        seed_arrays.append(component_seeds(
            smooth,
            threshold=seed_threshold,
            border=border,
            min_area_px=min_component_area_px,
            max_area_px=max_component_area_px,
            max_components=max_peaks,
        ))
    seed_arrays = [arr for arr in seed_arrays if len(arr)]
    if seed_arrays:
        peaks = np.unique(np.vstack(seed_arrays), axis=0)
    else:
        peaks = np.empty((0, 2), dtype=np.int32)

    proposals: list[CircleProposal] = []
    tile_name = prob_path.stem
    radius_norm_px = float(np.median(radii_px))
    for row, col in peaks:
        prop = score_seed(
            prob,
            int(row),
            int(col),
            offsets_by_radius,
            ring_weight,
            center_weight,
            radius_score_power,
            radius_norm_px,
        )
        if prop is None:
            continue
        if prop.score < score_threshold or prop.disk_mean < min_disk_mean:
            continue
        prop.tile = tile_name
        proposals.append(prop)

    proposals = nms_circles(proposals, nms_overlap, nms_center_fraction)

    records = []
    for prop in proposals:
        lon, lat = rasterio.transform.xy(transform, prop.row, prop.col, offset="center")
        px_m = pixel_size_m(gt, float(lat))
        radius_m = prop.radius_px * px_m
        geom = circle_polygon_wgs84(float(lon), float(lat), radius_m)
        if not isinstance(geom, Polygon) or geom.is_empty:
            continue
        records.append({
            "geometry": geom,
            "center_lon": round(float(lon), 6),
            "center_lat": round(float(lat), 6),
            "radius_px": int(prop.radius_px),
            "radius_m": round(float(radius_m), 1),
            "score": round(float(prop.score), 4),
            "disk_mean": round(float(prop.disk_mean), 4),
            "ring_mean": round(float(prop.ring_mean), 4),
            "center_prob": round(float(prop.center_prob), 4),
            "tile": tile_name,
        })

    columns = [
        "geometry", "center_lon", "center_lat", "radius_px", "radius_m",
        "score", "disk_mean", "ring_mean", "center_prob", "tile",
    ]
    if not records:
        return gpd.GeoDataFrame(columns=columns, crs=crs)
    return gpd.GeoDataFrame(records, crs=crs)


def process_prob_maps(
    prob_dir: Path,
    output_path: Path,
    tile_names: set[str] | None = None,
    limit: int | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    prob_files = sorted(prob_dir.glob("*.tif"))
    if tile_names is not None:
        prob_files = [p for p in prob_files if p.stem in tile_names]
    if limit is not None:
        prob_files = prob_files[:limit]

    all_gdfs = []
    for prob_path in tqdm(prob_files, desc="circle-detect"):
        gdf = detect_tile(prob_path, **kwargs)
        logger.info(f"{prob_path.name}: {len(gdf)} circle proposals")
        if len(gdf) > 0:
            all_gdfs.append(gdf)

    if all_gdfs:
        merged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
    else:
        merged = gpd.GeoDataFrame(
            columns=[
                "geometry", "center_lon", "center_lat", "radius_px", "radius_m",
                "score", "disk_mean", "ring_mean", "center_prob", "tile",
            ],
            crs="EPSG:4326",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_path, driver="GPKG")

    summary = {
        "total_proposals": int(len(merged)),
        "tiles_with_proposals": int(merged["tile"].nunique()) if len(merged) else 0,
        "median_score": round(float(merged["score"].median()), 4) if len(merged) else 0.0,
        "median_radius_px": round(float(merged["radius_px"].median()), 2) if len(merged) else 0.0,
    }
    with open(output_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {len(merged)} proposals to {output_path}")
    logger.info(f"Summary: {summary}")
    return merged


def parse_radii(text: str) -> list[int]:
    radii = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            radii.extend(range(int(lo), int(hi) + 1))
        else:
            radii.append(int(part))
    radii = sorted(set(radii))
    if not radii:
        raise ValueError("at least one radius is required")
    return radii


def main():
    parser = argparse.ArgumentParser(description="Detect circle proposals from prob maps")
    parser.add_argument("--prob-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radii-px", default="4-18")
    parser.add_argument("--seed-threshold", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.18)
    parser.add_argument("--min-disk-mean", type=float, default=0.35)
    parser.add_argument("--min-distance-px", type=int, default=6)
    parser.add_argument("--seed-sigma", type=float, default=2.5)
    parser.add_argument("--max-peaks-per-tile", type=int, default=12000)
    parser.add_argument("--seed-mode", choices=["peaks", "components", "both"], default="peaks")
    parser.add_argument("--min-component-area-px", type=int, default=12)
    parser.add_argument("--max-component-area-px", type=int, default=5000)
    parser.add_argument("--ring-gap-px", type=int, default=2)
    parser.add_argument("--ring-width-px", type=int, default=4)
    parser.add_argument("--ring-weight", type=float, default=0.55)
    parser.add_argument("--center-weight", type=float, default=0.10)
    parser.add_argument("--radius-score-power", type=float, default=0.0)
    parser.add_argument("--nms-overlap", type=float, default=0.25)
    parser.add_argument("--nms-center-fraction", type=float, default=0.60)
    parser.add_argument("--tile-list", default=None,
                        help="Optional text file listing prob-map tile stems to process")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of prob maps, useful for quick smoke tests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tile_names = None
    if args.tile_list:
        with open(args.tile_list) as f:
            tile_names = {line.strip() for line in f if line.strip()}

    process_prob_maps(
        Path(args.prob_dir),
        Path(args.output),
        tile_names=tile_names,
        limit=args.limit,
        radii_px=parse_radii(args.radii_px),
        seed_threshold=args.seed_threshold,
        score_threshold=args.score_threshold,
        min_disk_mean=args.min_disk_mean,
        min_distance_px=args.min_distance_px,
        seed_sigma=args.seed_sigma,
        max_peaks=args.max_peaks_per_tile,
        seed_mode=args.seed_mode,
        min_component_area_px=args.min_component_area_px,
        max_component_area_px=args.max_component_area_px,
        ring_gap_px=args.ring_gap_px,
        ring_width_px=args.ring_width_px,
        ring_weight=args.ring_weight,
        center_weight=args.center_weight,
        radius_score_power=args.radius_score_power,
        nms_overlap=args.nms_overlap,
        nms_center_fraction=args.nms_center_fraction,
    )


if __name__ == "__main__":
    main()
