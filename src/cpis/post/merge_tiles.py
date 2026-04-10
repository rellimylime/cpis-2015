"""Merge per-tile detections into year-level shapefile."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from cpis.common.constants import DEFAULT_FINAL_SHAPEFILE_TEMPLATE
from cpis.common.file_utils import ensure_dir, save_json
from cpis.common.logging_utils import build_logger
from cpis.common.time_utils import utc_now_iso


def run_merge_year(args: argparse.Namespace) -> int:
    log = build_logger(args.log_file if args.log_file else "")
    year = int(args.year)
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.rglob(args.glob_pattern))
    files = [p for p in files if p.suffix.lower() in {".shp", ".gpkg"}]
    if not files:
        raise RuntimeError(f"No vector files matched in {input_dir} with pattern {args.glob_pattern}")

    parts = []
    for p in files:
        try:
            g = gpd.read_file(p)
            if g.empty:
                continue
            if g.crs is None:
                g = g.set_crs("EPSG:4326")
            elif str(g.crs).upper() != "EPSG:4326":
                g = g.to_crs("EPSG:4326")
            parts.append(g)
            log(f"Loaded {p} ({len(g)} rows)")
        except Exception as exc:
            log(f"[warn] skip {p}: {exc}")

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326") if parts else gpd.GeoDataFrame()
    if not merged.empty and "year" not in merged.columns:
        merged["year"] = year

    out_shp = Path(args.out_shp) if args.out_shp else (Path("outputs") / "final" / DEFAULT_FINAL_SHAPEFILE_TEMPLATE.format(year=year))
    ensure_dir(out_shp.parent)
    if out_shp.exists():
        for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
            f = out_shp.with_suffix(ext)
            if f.exists():
                f.unlink()
    if merged.empty:
        log("Merged dataset is empty; writing empty shapefile.")
        empty = gpd.GeoDataFrame({"year": []}, geometry=[], crs="EPSG:4326")
        empty.to_file(out_shp)
    else:
        merged.to_file(out_shp)
    log(f"Wrote merged shapefile: {out_shp}")

    save_json(
        Path(args.summary_json) if args.summary_json else (out_shp.parent / f"{out_shp.stem}_summary.json"),
        {
            "generated_at": utc_now_iso(),
            "year": year,
            "input_dir": str(input_dir.resolve()),
            "input_files": [str(p.resolve()) for p in files],
            "rows": int(len(merged)),
            "output_shp": str(out_shp.resolve()),
        },
    )
    return 0


def build_parser(subparsers) -> None:
    p = subparsers.add_parser("merge-year", help="Merge per-tile vector outputs into one yearly shapefile")
    p.add_argument("--year", type=int, required=True, help="Year")
    p.add_argument("--input-dir", required=True, help="Input directory with per-tile outputs")
    p.add_argument("--glob-pattern", default="*.gpkg", help="Glob pattern to match vector files")
    p.add_argument("--out-shp", default="", help="Output shapefile path")
    p.add_argument("--summary-json", default="", help="Summary JSON path")
    p.add_argument("--log-file", default="", help="Optional log file")
    p.set_defaults(func=run_merge_year)

