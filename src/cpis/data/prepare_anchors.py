"""Normalize 2000/2021 anchor inventories and derive stable/change strata."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.prepared import prep
from shapely.strtree import STRtree

from cpis.common.constants import (
    DEFAULT_ANCHOR_2000,
    DEFAULT_ANCHOR_2021,
    DEFAULT_EQUAL_AREA_CRS,
    DEFAULT_RSE2023_2015_RUN_DIR,
)
from cpis.common.file_utils import ensure_dir, save_json
from cpis.common.logging_utils import build_logger


def _safe_union(series: gpd.GeoSeries):
    if hasattr(series, "union_all"):
        return series.union_all()
    return series.unary_union


def _iter_polygonal_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for part in geom.geoms:
            out.extend(_iter_polygonal_parts(part))
        return out
    return []


def _geometry_from_parts(parts: list[Polygon]):
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def _repair_geom(geom, *, label: str, log):
    if geom is None or geom.is_empty:
        return geom
    try:
        if bool(geom.is_valid):
            return geom
    except Exception:
        pass

    repaired = None
    try:
        repaired = geom.buffer(0)
    except Exception:
        repaired = None

    if repaired is None or repaired.is_empty:
        try:
            from shapely.validation import make_valid

            repaired = make_valid(geom)
        except Exception:
            repaired = None

    parts = _iter_polygonal_parts(repaired) if repaired is not None else []
    out = _geometry_from_parts(parts) if parts else repaired
    if out is not None and not out.is_empty:
        try:
            if not bool(out.is_valid):
                out = out.buffer(0)
        except Exception:
            pass
        log(f"{label}: repaired invalid geometry")
        return out

    log(f"{label}: could not repair geometry cleanly; keeping original")
    return geom


def _repair_polygonal(gdf: gpd.GeoDataFrame, label: str, log) -> gpd.GeoDataFrame:
    work = gdf.copy()
    work = work[work.geometry.notna()].copy()
    work = work[~work.geometry.is_empty].copy()
    if work.empty:
        raise RuntimeError(f"{label}: no non-empty geometries found.")

    invalid = ~work.geometry.is_valid
    if bool(invalid.any()):
        log(f"{label}: repairing {int(invalid.sum())} invalid geometries with buffer(0)")
        work.loc[invalid, "geometry"] = work.loc[invalid, "geometry"].buffer(0)

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        parts = _iter_polygonal_parts(row.geometry)
        geom = _geometry_from_parts(parts)
        if geom is None or geom.is_empty:
            continue
        payload = row.to_dict()
        payload["geometry"] = geom
        rows.append(payload)

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=work.crs)
    if out.empty:
        raise RuntimeError(f"{label}: no polygonal geometries remained after repair.")
    out = out.reset_index(drop=True)
    return out


def _country_value(row: pd.Series, preferred_field: str) -> str:
    candidates = [preferred_field, "COUNTRYAFF", "NAME", "ISO_3DIGIT", "ISO_2DIGIT", "FIPS_CNTRY"]
    for field in candidates:
        if field and field in row.index:
            val = row[field]
            if pd.notna(val):
                text = str(val).strip()
                if text:
                    return text
    return "unknown"


def _source_id_value(row: pd.Series, fallback_idx: int) -> str:
    for field in ("Id", "OBJECTID"):
        if field in row.index and pd.notna(row[field]):
            return str(row[field]).strip()
    return str(int(fallback_idx) + 1)


def _load_anchor_layer(
    path: Path,
    *,
    year: int,
    equal_area_crs: str,
    country_field: str,
    log,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Anchor layer not found: {path}")
    raw = gpd.read_file(path)
    if raw.empty:
        raise RuntimeError(f"Anchor layer is empty: {path}")
    if raw.crs is None:
        raise RuntimeError(f"Anchor layer missing CRS: {path}")

    raw = _repair_polygonal(raw, label=f"anchor_{year}", log=log)
    raw = raw.reset_index(drop=True)
    eq = raw.to_crs(equal_area_crs)
    centroids_eq = eq.geometry.centroid
    centroids_wgs = gpd.GeoSeries(centroids_eq, crs=equal_area_crs).to_crs("EPSG:4326")

    eq = eq.copy()
    eq["anchor_year"] = int(year)
    eq["anchor_id"] = [f"{year}_{i + 1:07d}" for i in range(len(eq))]
    eq["source_id"] = [_source_id_value(eq.iloc[i], i) for i in range(len(eq))]
    eq["country_name"] = [_country_value(eq.iloc[i], country_field) for i in range(len(eq))]
    eq["source_path"] = str(path.resolve())
    eq["area_m2_src"] = pd.to_numeric(eq.get("Area_m2"), errors="coerce") if "Area_m2" in eq.columns else np.nan
    eq["area_m2_eq"] = eq.geometry.area.astype(float)
    eq["radius_m_eq"] = np.sqrt(eq["area_m2_eq"].clip(lower=0.0) / math.pi)
    eq["center_x_eq"] = centroids_eq.x.astype(float)
    eq["center_y_eq"] = centroids_eq.y.astype(float)
    eq["center_lon"] = centroids_wgs.x.astype(float)
    eq["center_lat"] = centroids_wgs.y.astype(float)

    wgs = eq.to_crs("EPSG:4326")
    return eq, wgs


def _query_tree_indices(tree: STRtree, geom, geom_to_idx: dict[int, int]) -> list[int]:
    raw = tree.query(geom)
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        if np.issubdtype(raw.dtype, np.integer):
            return [int(v) for v in raw.tolist()]
        return [geom_to_idx[id(v)] for v in raw.tolist()]
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], (int, np.integer)):
            return [int(v) for v in raw]
        return [geom_to_idx[id(v)] for v in raw]
    if isinstance(raw, (int, np.integer)):
        return [int(raw)]
    return []


def _stable_match_rows(lhs_eq: gpd.GeoDataFrame, rhs_eq: gpd.GeoDataFrame, min_iou: float) -> pd.DataFrame:
    rhs_geoms = list(rhs_eq.geometry)
    tree = STRtree(rhs_geoms)
    geom_to_idx = {id(g): idx for idx, g in enumerate(rhs_geoms)}
    rows: list[dict[str, Any]] = []

    for lhs_idx, lhs_row in lhs_eq.iterrows():
        lhs_geom = lhs_row.geometry
        if lhs_geom is None or lhs_geom.is_empty:
            continue
        for rhs_idx in _query_tree_indices(tree, lhs_geom, geom_to_idx):
            rhs_geom = rhs_geoms[rhs_idx]
            if rhs_geom is None or rhs_geom.is_empty:
                continue
            inter = lhs_geom.intersection(rhs_geom)
            inter_area = float(inter.area) if inter is not None and not inter.is_empty else 0.0
            if inter_area <= 0.0:
                continue
            union_area = float(lhs_geom.area + rhs_geom.area - inter_area)
            if union_area <= 0.0:
                continue
            iou = inter_area / union_area
            if iou < min_iou:
                continue
            area_ratio = float(min(lhs_geom.area, rhs_geom.area) / max(lhs_geom.area, rhs_geom.area))
            rows.append(
                {
                    "lhs_idx": int(lhs_idx),
                    "rhs_idx": int(rhs_idx),
                    "iou": float(iou),
                    "inter_area_m2": inter_area,
                    "union_area_m2": union_area,
                    "area_ratio": area_ratio,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["lhs_idx", "rhs_idx", "iou", "inter_area_m2", "union_area_m2", "area_ratio"])

    cand = pd.DataFrame(rows).sort_values(
        by=["iou", "inter_area_m2", "area_ratio"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    keep: list[dict[str, Any]] = []
    used_lhs: set[int] = set()
    used_rhs: set[int] = set()
    for row in cand.to_dict(orient="records"):
        li = int(row["lhs_idx"])
        ri = int(row["rhs_idx"])
        if li in used_lhs or ri in used_rhs:
            continue
        used_lhs.add(li)
        used_rhs.add(ri)
        keep.append(row)
    return pd.DataFrame(keep)


def _stable_core_geometry(lhs_geom, rhs_geom):
    inter = lhs_geom.intersection(rhs_geom)
    parts = _iter_polygonal_parts(inter)
    if parts:
        return _geometry_from_parts(parts), "intersection"
    return (lhs_geom if lhs_geom.area <= rhs_geom.area else rhs_geom), "smaller_source"


def _build_stable_pivots(
    anchor_2000_eq: gpd.GeoDataFrame,
    anchor_2021_eq: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    equal_area_crs: str,
) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    for idx, match in enumerate(matches.to_dict(orient="records"), 1):
        row_2000 = anchor_2000_eq.iloc[int(match["lhs_idx"])]
        row_2021 = anchor_2021_eq.iloc[int(match["rhs_idx"])]
        geom, geometry_mode = _stable_core_geometry(row_2000.geometry, row_2021.geometry)
        cent = gpd.GeoSeries([geom.centroid], crs=equal_area_crs).to_crs("EPSG:4326").iloc[0]
        area_m2_eq = float(geom.area)
        rows.append(
            {
                "stable_id": f"stable_{idx:07d}",
                "anchor_id_2000": row_2000["anchor_id"],
                "anchor_id_2021": row_2021["anchor_id"],
                "country_2000": row_2000["country_name"],
                "country_2021": row_2021["country_name"],
                "country_name": row_2021["country_name"] if row_2021["country_name"] == row_2000["country_name"] else "mixed",
                "iou": float(match["iou"]),
                "area_ratio": float(match["area_ratio"]),
                "area_m2_2000_eq": float(row_2000["area_m2_eq"]),
                "area_m2_2021_eq": float(row_2021["area_m2_eq"]),
                "area_m2_eq": area_m2_eq,
                "radius_m_eq": float(math.sqrt(max(area_m2_eq, 0.0) / math.pi)),
                "center_lon": float(cent.x),
                "center_lat": float(cent.y),
                "geometry_mode": geometry_mode,
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=equal_area_crs)


def _dissolved_components(seed_eq: gpd.GeoDataFrame, *, equal_area_crs: str) -> gpd.GeoDataFrame:
    if seed_eq.empty:
        return gpd.GeoDataFrame(columns=["zone_id", "seed_count", "has_2000", "has_2021", "geometry"], geometry="geometry", crs=equal_area_crs)

    merged = _safe_union(seed_eq.geometry)
    parts = _iter_polygonal_parts(merged)
    if not parts:
        return gpd.GeoDataFrame(columns=["zone_id", "seed_count", "has_2000", "has_2021", "geometry"], geometry="geometry", crs=equal_area_crs)

    tree = STRtree(list(seed_eq.geometry))
    geom_to_idx = {id(g): idx for idx, g in enumerate(list(seed_eq.geometry))}
    rows: list[dict[str, Any]] = []
    for idx, geom in enumerate(parts, 1):
        hit_indices = _query_tree_indices(tree, geom, geom_to_idx)
        hit_indices = [i for i in hit_indices if geom.intersects(seed_eq.geometry.iloc[i])]
        years = set(int(seed_eq.iloc[i]["anchor_year"]) for i in hit_indices)
        rows.append(
            {
                "zone_id": f"change_{idx:07d}",
                "seed_count": int(len(hit_indices)),
                "has_2000": bool(2000 in years),
                "has_2021": bool(2021 in years),
                "area_m2_eq": float(geom.area),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=equal_area_crs)


def _square_grid(bounds: tuple[float, float, float, float], cell_size_m: float) -> list[Polygon]:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    cells: list[Polygon] = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, min(x + cell_size_m, maxx), min(y + cell_size_m, maxy)))
            y += cell_size_m
        x += cell_size_m
    return cells


def _load_aoi_geometry(path: str, equal_area_crs: str, anchor_2000_eq: gpd.GeoDataFrame, anchor_2021_eq: gpd.GeoDataFrame):
    if path:
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise RuntimeError(f"AOI layer is empty: {path}")
        if gdf.crs is None:
            raise RuntimeError(f"AOI layer missing CRS: {path}")
        return _safe_union(gdf.to_crs(equal_area_crs).geometry), "user"

    minx = min(float(anchor_2000_eq.total_bounds[0]), float(anchor_2021_eq.total_bounds[0]))
    miny = min(float(anchor_2000_eq.total_bounds[1]), float(anchor_2021_eq.total_bounds[1]))
    maxx = max(float(anchor_2000_eq.total_bounds[2]), float(anchor_2021_eq.total_bounds[2]))
    maxy = max(float(anchor_2000_eq.total_bounds[3]), float(anchor_2021_eq.total_bounds[3]))
    return box(minx, miny, maxx, maxy), "combined_envelope"


def _build_background_layers(
    *,
    aoi_geom_eq,
    anchor_2000_eq: gpd.GeoDataFrame,
    anchor_2021_eq: gpd.GeoDataFrame,
    change_zones_eq: gpd.GeoDataFrame,
    equal_area_crs: str,
    pivot_buffer_m: float,
    background_cell_km: float,
    log,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    excluded_parts = []
    excluded_inputs = [
        ("anchor_2000_buffer", _safe_union(anchor_2000_eq.geometry.buffer(pivot_buffer_m))),
        ("anchor_2021_buffer", _safe_union(anchor_2021_eq.geometry.buffer(pivot_buffer_m))),
    ]
    if not change_zones_eq.empty:
        excluded_inputs.append(("change_zones", _safe_union(change_zones_eq.geometry)))

    for label, geom in excluded_inputs:
        geom = _repair_geom(geom, label=label, log=log)
        excluded_parts.extend(_iter_polygonal_parts(geom))

    excluded = _geometry_from_parts(excluded_parts)
    excluded = _repair_geom(excluded, label="excluded_union", log=log)
    excluded_prepared = prep(excluded) if excluded is not None and not excluded.is_empty else None
    cell_size_m = float(background_cell_km) * 1000.0
    rows: list[dict[str, Any]] = []
    for idx, cell in enumerate(_square_grid(aoi_geom_eq.bounds, cell_size_m), 1):
        clipped = cell.intersection(aoi_geom_eq)
        if excluded is not None and not excluded.is_empty:
            try:
                clipped = clipped.difference(excluded)
            except GEOSException:
                clipped = _repair_geom(clipped, label=f"bg_cell_{idx:07d}_retry", log=log)
                excluded = _repair_geom(excluded, label="excluded_union_retry", log=log)
                excluded_prepared = prep(excluded) if excluded is not None and not excluded.is_empty else None
                clipped = clipped.difference(excluded) if excluded is not None and not excluded.is_empty else clipped

        parts = _iter_polygonal_parts(clipped)
        if not parts:
            continue
        for part_idx, geom in enumerate(parts, 1):
            cent = geom.centroid
            if excluded is not None and not excluded.is_empty:
                try:
                    if excluded_prepared is not None and bool(excluded_prepared.intersects(cent)):
                        continue
                except GEOSException:
                    excluded = _repair_geom(excluded, label="excluded_union_retry", log=log)
                    excluded_prepared = prep(excluded) if excluded is not None and not excluded.is_empty else None
                    if excluded_prepared is not None and bool(excluded_prepared.intersects(cent)):
                        continue
            rows.append(
                {
                    "bg_id": f"bg_{idx:07d}_p{part_idx:02d}",
                    "cell_km": float(background_cell_km),
                    "area_m2_eq": float(geom.area),
                    "geometry": geom,
                }
            )

    cells_eq = gpd.GeoDataFrame(rows, geometry="geometry", crs=equal_area_crs)
    if cells_eq.empty:
        points_eq = gpd.GeoDataFrame(columns=["bg_id", "center_x_eq", "center_y_eq", "geometry"], geometry="geometry", crs=equal_area_crs)
    else:
        centroids = cells_eq.geometry.centroid
        points_eq = gpd.GeoDataFrame(
            {
                "bg_id": cells_eq["bg_id"].tolist(),
                "center_x_eq": centroids.x.astype(float).tolist(),
                "center_y_eq": centroids.y.astype(float).tolist(),
            },
            geometry=centroids,
            crs=equal_area_crs,
        )
    return cells_eq, points_eq


def _write_layer(gdf: gpd.GeoDataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")


def run_prepare_anchor_truth(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root)
    log = build_logger(args.log_file if args.log_file else (out_root / "prepare_anchor_truth.log"))
    anchors_dir = ensure_dir(out_root / "anchors")
    overlays_dir = ensure_dir(out_root / "overlays")

    log(f"Loading anchor layers: 2000={args.truth_2000} 2021={args.truth_2021}")
    anchor_2000_eq, anchor_2000_wgs = _load_anchor_layer(
        Path(args.truth_2000),
        year=2000,
        equal_area_crs=args.equal_area_crs,
        country_field=args.country_field,
        log=log,
    )
    anchor_2021_eq, anchor_2021_wgs = _load_anchor_layer(
        Path(args.truth_2021),
        year=2021,
        equal_area_crs=args.equal_area_crs,
        country_field=args.country_field,
        log=log,
    )

    log(
        f"Loaded anchors: 2000={len(anchor_2000_eq)} polygons, "
        f"2021={len(anchor_2021_eq)} polygons, equal_area_crs={args.equal_area_crs}"
    )

    log(f"Matching stable pivots with IoU >= {float(args.stable_iou_min):.2f}")
    matches = _stable_match_rows(anchor_2000_eq, anchor_2021_eq, min_iou=float(args.stable_iou_min))
    log(f"Stable matching complete: matches={len(matches)}")
    stable_eq = _build_stable_pivots(anchor_2000_eq, anchor_2021_eq, matches, equal_area_crs=args.equal_area_crs)
    matched_2000 = set(matches["lhs_idx"].astype(int).tolist()) if not matches.empty else set()
    matched_2021 = set(matches["rhs_idx"].astype(int).tolist()) if not matches.empty else set()

    unmatched_2000 = anchor_2000_eq.loc[[idx for idx in range(len(anchor_2000_eq)) if idx not in matched_2000]].copy()
    unmatched_2021 = anchor_2021_eq.loc[[idx for idx in range(len(anchor_2021_eq)) if idx not in matched_2021]].copy()

    seed_rows: list[gpd.GeoDataFrame] = []
    if not unmatched_2000.empty:
        tmp = unmatched_2000[["anchor_id", "anchor_year", "country_name", "geometry"]].copy()
        seed_rows.append(tmp)
    if not unmatched_2021.empty:
        tmp = unmatched_2021[["anchor_id", "anchor_year", "country_name", "geometry"]].copy()
        seed_rows.append(tmp)
    if seed_rows:
        change_seed_eq = gpd.GeoDataFrame(pd.concat(seed_rows, ignore_index=True), geometry="geometry", crs=args.equal_area_crs)
        change_seed_eq["geometry"] = change_seed_eq.geometry.buffer(float(args.change_buffer_m))
    else:
        change_seed_eq = gpd.GeoDataFrame(columns=["anchor_id", "anchor_year", "country_name", "geometry"], geometry="geometry", crs=args.equal_area_crs)
    log(
        f"Building change zones from unmatched anchors: "
        f"unmatched_2000={len(unmatched_2000)} unmatched_2021={len(unmatched_2021)}"
    )
    change_zones_eq = _dissolved_components(change_seed_eq, equal_area_crs=args.equal_area_crs)

    aoi_geom_eq, aoi_mode = _load_aoi_geometry(
        args.aoi,
        equal_area_crs=args.equal_area_crs,
        anchor_2000_eq=anchor_2000_eq,
        anchor_2021_eq=anchor_2021_eq,
    )
    log(
        f"Building stable background layers: aoi_mode={aoi_mode} "
        f"pivot_buffer_m={float(args.pivot_buffer_m)} cell_km={float(args.background_cell_km)}"
    )
    bg_cells_eq, bg_points_eq = _build_background_layers(
        aoi_geom_eq=aoi_geom_eq,
        anchor_2000_eq=anchor_2000_eq,
        anchor_2021_eq=anchor_2021_eq,
        change_zones_eq=change_zones_eq,
        equal_area_crs=args.equal_area_crs,
        pivot_buffer_m=float(args.pivot_buffer_m),
        background_cell_km=float(args.background_cell_km),
        log=log,
    )

    stable_wgs = stable_eq.to_crs("EPSG:4326")
    change_zones_wgs = change_zones_eq.to_crs("EPSG:4326")
    bg_cells_wgs = bg_cells_eq.to_crs("EPSG:4326")
    bg_points_wgs = bg_points_eq.to_crs("EPSG:4326")
    study_area_wgs = gpd.GeoDataFrame(
        [{"aoi_mode": aoi_mode, "geometry": aoi_geom_eq}],
        geometry="geometry",
        crs=args.equal_area_crs,
    ).to_crs("EPSG:4326")

    out_anchor_2000 = anchors_dir / "anchor_2000_normalized.gpkg"
    out_anchor_2021 = anchors_dir / "anchor_2021_normalized.gpkg"
    out_stable = overlays_dir / "stable_pivots.gpkg"
    out_change = overlays_dir / "change_zones.gpkg"
    out_bg_cells = overlays_dir / "stable_background_cells.gpkg"
    out_bg_points = overlays_dir / "stable_background_points.gpkg"
    out_study_area = overlays_dir / "study_area.gpkg"

    log("Writing normalized anchor and overlay layers")
    _write_layer(anchor_2000_wgs, out_anchor_2000)
    _write_layer(anchor_2021_wgs, out_anchor_2021)
    _write_layer(stable_wgs, out_stable)
    _write_layer(change_zones_wgs, out_change)
    _write_layer(bg_cells_wgs, out_bg_cells)
    _write_layer(bg_points_wgs, out_bg_points)
    _write_layer(study_area_wgs, out_study_area)

    summary = {
        "truth_2000": str(Path(args.truth_2000).resolve()),
        "truth_2021": str(Path(args.truth_2021).resolve()),
        "equal_area_crs": str(args.equal_area_crs),
        "stable_iou_min": float(args.stable_iou_min),
        "pivot_buffer_m": float(args.pivot_buffer_m),
        "change_buffer_m": float(args.change_buffer_m),
        "background_cell_km": float(args.background_cell_km),
        "aoi_mode": aoi_mode,
        "aoi_path": str(Path(args.aoi).resolve()) if args.aoi else "",
        "counts": {
            "anchor_2000": int(len(anchor_2000_wgs)),
            "anchor_2021": int(len(anchor_2021_wgs)),
            "stable_pivots": int(len(stable_wgs)),
            "change_zones": int(len(change_zones_wgs)),
            "stable_background_cells": int(len(bg_cells_wgs)),
            "stable_background_points": int(len(bg_points_wgs)),
            "unmatched_2000": int(len(unmatched_2000)),
            "unmatched_2021": int(len(unmatched_2021)),
        },
        "outputs": {
            "anchor_2000": str(out_anchor_2000.resolve()),
            "anchor_2021": str(out_anchor_2021.resolve()),
            "stable_pivots": str(out_stable.resolve()),
            "change_zones": str(out_change.resolve()),
            "stable_background_cells": str(out_bg_cells.resolve()),
            "stable_background_points": str(out_bg_points.resolve()),
            "study_area": str(out_study_area.resolve()),
        },
    }
    save_json(out_root / "summary.json", summary)
    log(f"Wrote anchor prep summary: {out_root / 'summary.json'}")
    return 0


def build_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "prepare-anchor-truth",
        help="Normalize 2000/2021 anchor inventories and derive stable/change strata",
    )
    p.add_argument("--truth-2000", default=str(DEFAULT_ANCHOR_2000), help="2000 anchor polygon layer")
    p.add_argument("--truth-2021", default=str(DEFAULT_ANCHOR_2021), help="2021 anchor polygon layer")
    p.add_argument(
        "--out-root",
        default=str(DEFAULT_RSE2023_2015_RUN_DIR / "anchor_truth"),
        help="Output directory for normalized anchors and overlay strata",
    )
    p.add_argument("--aoi", default="", help="Optional AOI polygon layer; defaults to combined anchor envelope")
    p.add_argument("--country-field", default="NAME", help="Preferred field to use as country label")
    p.add_argument("--equal-area-crs", default=DEFAULT_EQUAL_AREA_CRS, help="Equal-area CRS for metric calculations")
    p.add_argument("--stable-iou-min", type=float, default=0.20, help="Minimum IoU to classify a pivot as stable")
    p.add_argument("--pivot-buffer-m", type=float, default=1000.0, help="Buffer around pivots excluded from stable background")
    p.add_argument("--change-buffer-m", type=float, default=1500.0, help="Buffer applied to unmatched pivots before dissolving change zones")
    p.add_argument("--background-cell-km", type=float, default=50.0, help="Stable background sampling cell size in km")
    p.add_argument("--log-file", default="", help="Optional log file")
    p.set_defaults(func=run_prepare_anchor_truth)
