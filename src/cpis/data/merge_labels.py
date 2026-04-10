"""Merge manual reviewed training polygons into a weak-label layer.

Two merge modes are supported:

- replace: on reviewed source tiles, manual polygons replace weak labels
  entirely.
- augment: keep existing weak labels, but replace any weak polygon clearly
  superseded by a manual polygon and add manual polygons as high-confidence
  positives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from osgeo import gdal
from shapely.geometry import box
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cpis.common.file_utils import ensure_dir, save_json  # noqa: E402
from cpis.common.logging_utils import build_logger  # noqa: E402


gdal.UseExceptions()


def _raster_bounds(tif_path: Path):
    ds = gdal.Open(str(tif_path))
    if ds is None:
        raise RuntimeError(f"Could not open raster: {tif_path}")
    gt = ds.GetGeoTransform()
    width = int(ds.RasterXSize)
    height = int(ds.RasterYSize)
    xs = [0, width]
    ys = [0, height]
    corners = []
    for px in xs:
        for py in ys:
            mx = gt[0] + (px * gt[1]) + (py * gt[2])
            my = gt[3] + (px * gt[4]) + (py * gt[5])
            corners.append((mx, my))
    proj_wkt = ds.GetProjectionRef()
    ds = None
    minx = min(x for x, _ in corners)
    maxx = max(x for x, _ in corners)
    miny = min(y for _, y in corners)
    maxy = max(y for _, y in corners)
    return box(minx, miny, maxx, maxy), proj_wkt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagery-dir", required=True, help="Directory containing source TIFFs")
    ap.add_argument("--weak-labels", required=True, help="Weak polygon label layer (.gpkg/.shp)")
    ap.add_argument("--manual-labels", required=True, help="Manual reviewed training labels (.gpkg/.shp)")
    ap.add_argument("--out", required=True, help="Merged output polygon layer (.gpkg/.shp)")
    ap.add_argument(
        "--merge-mode",
        choices=["replace", "augment"],
        default="replace",
        help="How manual reviewed tiles should be merged with weak labels",
    )
    ap.add_argument("--manual-source-field", default="source_name", help="Field storing source tile names")
    ap.add_argument("--manual-label-set-field", default="label_set", help="Optional label-set field")
    ap.add_argument(
        "--manual-label-set",
        default="train",
        help="Manual label-set value to keep; empty disables label-set filtering",
    )
    ap.add_argument("--manual-status-field", default="status", help="Optional status field")
    ap.add_argument(
        "--include-status",
        nargs="*",
        default=["confirmed"],
        help="Manual status values to keep; empty disables status filtering",
    )
    ap.add_argument("--log-file", default="", help="Optional log file")
    args = ap.parse_args()

    imagery_dir = Path(args.imagery_dir)
    if not imagery_dir.exists():
        raise FileNotFoundError(f"Imagery dir not found: {imagery_dir}")

    weak_path = Path(args.weak_labels)
    if not weak_path.exists():
        raise FileNotFoundError(f"Weak labels not found: {weak_path}")

    manual_path = Path(args.manual_labels)
    if not manual_path.exists():
        raise FileNotFoundError(f"Manual labels not found: {manual_path}")

    out_path = Path(args.out)
    log = build_logger(args.log_file if args.log_file else (out_path.parent / "merge_train_review_labels.log"))

    weak = gpd.read_file(weak_path)
    if weak.empty:
        raise RuntimeError(f"Weak labels are empty: {weak_path}")
    if weak.crs is None:
        raise RuntimeError(f"Weak labels missing CRS: {weak_path}")

    manual = gpd.read_file(manual_path)
    if manual.empty:
        raise RuntimeError(f"Manual labels are empty: {manual_path}")
    if manual.crs is None:
        raise RuntimeError(f"Manual labels missing CRS: {manual_path}")
    if args.manual_source_field not in manual.columns:
        raise RuntimeError(f"Manual labels missing source field: {args.manual_source_field}")

    selected = manual.copy()
    if args.manual_label_set and args.manual_label_set_field in selected.columns:
        selected = selected[selected[args.manual_label_set_field].astype(str) == str(args.manual_label_set)].copy()
    if args.include_status and args.manual_status_field in selected.columns:
        allowed = {str(v) for v in args.include_status}
        selected = selected[selected[args.manual_status_field].astype(str).isin(allowed)].copy()
    selected = selected[~selected.geometry.isna()].copy()
    selected = selected[~selected.geometry.is_empty].copy()
    if selected.empty:
        raise RuntimeError("No manual labels remain after filtering")

    reviewed_sources = sorted(
        {str(v) for v in selected[args.manual_source_field].dropna().unique() if str(v).strip()}
    )
    if not reviewed_sources:
        raise RuntimeError(f"No usable source names found in field: {args.manual_source_field}")

    footprint_rows: list[dict[str, object]] = []
    proj_wkt: str | None = None
    for source_name in reviewed_sources:
        tif_path = imagery_dir / f"{source_name}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"Reviewed source TIFF not found: {tif_path}")
        geom, this_proj_wkt = _raster_bounds(tif_path)
        if proj_wkt is None:
            proj_wkt = this_proj_wkt
        elif str(proj_wkt) != str(this_proj_wkt):
            raise RuntimeError(f"Mixed raster projections across reviewed sources: {source_name}")
        footprint_rows.append({"source_name": source_name, "geometry": geom})
    assert proj_wkt is not None

    footprints = gpd.GeoDataFrame(footprint_rows, crs=proj_wkt)
    footprints_in_weak = footprints.to_crs(weak.crs)
    reviewed_union = unary_union(list(footprints_in_weak.geometry))
    manual_in_weak = selected.to_crs(weak.crs).copy()

    weak_reviewed = weak[weak.geometry.intersects(reviewed_union)].copy()
    weak_unreviewed = weak[~weak.geometry.intersects(reviewed_union)].copy()

    if str(args.merge_mode) == "replace":
        weak_kept = weak_unreviewed.copy()
        weak_removed = weak_reviewed.copy()
    else:
        # In augment mode, keep weak labels unless a weak polygon is clearly
        # being replaced by a manual polygon. Representative points are used
        # instead of raw intersects so nearby pivots are not accidentally
        # dropped just because two approximate circles touch.
        manual_union = unary_union(list(manual_in_weak.geometry))
        reviewed_points = weak_reviewed.geometry.representative_point()
        replaced_mask = reviewed_points.within(manual_union)
        weak_removed = weak_reviewed[replaced_mask].copy()
        weak_kept = pd.concat(
            [weak_unreviewed, weak_reviewed[~replaced_mask].copy()],
            ignore_index=False,
            sort=False,
        )
        weak_kept = gpd.GeoDataFrame(weak_kept, crs=weak.crs)

    if "label_origin" not in weak_kept.columns:
        weak_kept["label_origin"] = "weak"
    else:
        weak_kept["label_origin"] = weak_kept["label_origin"].fillna("weak")
    manual_in_weak["label_origin"] = "manual_review"

    merged = gpd.GeoDataFrame(
        pd.concat([weak_kept, manual_in_weak], ignore_index=True, sort=False),
        crs=weak.crs,
    )
    merged = merged[~merged.geometry.isna()].copy()
    merged = merged[~merged.geometry.is_empty].copy()

    ensure_dir(out_path.parent)
    if out_path.suffix.lower() == ".gpkg":
        merged.to_file(out_path, driver="GPKG")
    else:
        merged.to_file(out_path)

    footprints_path = out_path.with_name(out_path.stem + "_reviewed_sources.gpkg")
    footprints_in_weak.to_file(footprints_path, driver="GPKG")

    summary = {
        "imagery_dir": str(imagery_dir.resolve()),
        "weak_labels": str(weak_path.resolve()),
        "manual_labels": str(manual_path.resolve()),
        "out": str(out_path.resolve()),
        "merge_mode": str(args.merge_mode),
        "reviewed_sources_path": str(footprints_path.resolve()),
        "reviewed_source_count": int(len(reviewed_sources)),
        "reviewed_sources": reviewed_sources,
        "weak_input_count": int(len(weak)),
        "weak_reviewed_source_count": int(len(weak_reviewed)),
        "weak_removed_count": int(len(weak_removed)),
        "manual_selected_count": int(len(manual_in_weak)),
        "merged_count": int(len(merged)),
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    save_json(summary_path, summary)
    log(
        f"Merged labels written: mode={args.merge_mode} weak_in={len(weak)} "
        f"weak_reviewed={len(weak_reviewed)} weak_removed={len(weak_removed)} "
        f"manual_added={len(manual_in_weak)} merged={len(merged)} out={out_path}"
    )
    log(f"Wrote merge summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
