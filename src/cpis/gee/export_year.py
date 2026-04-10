"""Export yearly Landsat composites from GEE for arid SSA region tiles."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import Polygon

from cpis.common.constants import (
    DEFAULT_REGION_GEOJSON,
    DEFAULT_RUNS_DIR,
    EXPORT_STATUS_FAILED,
    EXPORT_STATUS_QUEUED,
    EXPORT_STATUS_RETRIED,
    EXPORT_STATUS_RUNNING,
    EXPORT_STATUS_SUCCEEDED,
)
from cpis.common.file_utils import ensure_dir
from cpis.common.geo_utils import make_grid
from cpis.common.lock_utils import manifest_lock
from cpis.common.logging_utils import build_logger
from cpis.common.manifest import load_manifest, save_manifest, tile_status_counts
from cpis.common.time_utils import utc_now_iso


TILE_ID_RE = re.compile(r"tile_(\d+)", re.IGNORECASE)


def _collection_ids_for_year(year: int) -> list[str]:
    ids: list[str] = []
    if 1984 <= year <= 2011:
        ids.append("LANDSAT/LT05/C02/T1_L2")
    if 1999 <= year <= 2022:
        ids.append("LANDSAT/LE07/C02/T1_L2")
    if year >= 2013:
        ids.append("LANDSAT/LC08/C02/T1_L2")
    if year >= 2021:
        ids.append("LANDSAT/LC09/C02/T1_L2")
    if not ids:
        raise RuntimeError(f"No supported Landsat C2 SR collections for year={year}")
    return ids


def _load_region(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Region file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise RuntimeError(f"Region geometry is empty: {path}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _manifest_template(year: int, region_path: str, tile_size_km: float, collections: list[str]) -> dict[str, Any]:
    return {
        "version": 1,
        "year": int(year),
        "region": region_path,
        "tile_size_km": float(tile_size_km),
        "source_collections": collections,
        "feature_contract": "stats_v1",
        "created_at": utc_now_iso(),
        "updated_at": None,
        "tiles": {},
    }


def _local_tile_key(name: str) -> str | None:
    m = TILE_ID_RE.search(str(name))
    if not m:
        return None
    return f"tile_{int(m.group(1)):06d}"


def _scan_local_tiles(paths: list[str], log) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            log(f"[warn] skip_local_dir missing: {path}")
            continue
        for tif_path in sorted(path.rglob("*.tif")):
            tile_id = _local_tile_key(tif_path.name)
            if tile_id is None:
                continue
            hits.setdefault(tile_id, []).append(str(tif_path.resolve()))
    if hits:
        log(f"Local tile scan complete: matched_tiles={len(hits)} from_dirs={len(paths)}")
    return hits


def _tile_rows(region_geom: Polygon, tile_size_km: float, max_tiles: int) -> list[dict[str, Any]]:
    all_tiles = make_grid(region_geom.bounds, tile_size_km)
    rows: list[dict[str, Any]] = []
    idx = 0
    for tile in all_tiles:
        if not tile.intersects(region_geom):
            continue
        tile = tile.intersection(region_geom.envelope)
        minx, miny, maxx, maxy = tile.bounds
        tile_id = f"tile_{idx:06d}"
        rows.append(
            {
                "tile_id": tile_id,
                "bbox": [float(minx), float(miny), float(maxx), float(maxy)],
                "status": EXPORT_STATUS_QUEUED,
                "retries": 0,
                "task_id": "",
                "started_at": None,
                "ended_at": None,
                "error": "",
            }
        )
        idx += 1
        if max_tiles > 0 and len(rows) >= max_tiles:
            break
    return rows


def _init_ee(project: str | None):
    import ee

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def _build_landsat_collection(ee, collection_id: str, region_rect, year: int, cloud_max: float):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    col = (
        ee.ImageCollection(collection_id)
        .filterBounds(region_rect)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
    )

    def _mask_scale(img):
        qa = img.select("QA_PIXEL")
        cloud = qa.bitwiseAnd(1 << 3).eq(0)
        cloud_shadow = qa.bitwiseAnd(1 << 4).eq(0)
        snow = qa.bitwiseAnd(1 << 5).eq(0)
        dilated = qa.bitwiseAnd(1 << 1).eq(0)
        mask = cloud.And(cloud_shadow).And(snow).And(dilated)

        if "LC08" in collection_id or "LC09" in collection_id:
            src = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
        else:
            src = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
        dst = ["blue", "green", "red", "nir", "swir1", "swir2"]
        scaled = img.select(src).multiply(0.0000275).add(-0.2).rename(dst)
        return scaled.updateMask(mask).copyProperties(img, img.propertyNames())

    return col.map(_mask_scale)


def _feature_band_names(feature_contract: str) -> list[str]:
    contract = str(feature_contract).strip().lower()
    if contract == "stats_v1":
        return [
            "blue_median",
            "green_median",
            "red_median",
            "nir_median",
            "swir1_median",
            "swir2_median",
            "ndvi_p10",
            "ndvi_p50",
            "ndvi_p90",
            "ndvi_amp",
            "ndwi_p50",
        ]
    if contract == "paper_rgbnir_v1":
        return ["blue", "green", "red", "nir"]
    raise RuntimeError(f"Unsupported feature_contract={feature_contract}")


def _build_feature_image(
    ee,
    collections: list[str],
    bbox: list[float],
    year: int,
    cloud_max: float,
    feature_contract: str = "stats_v1",
):
    region_rect = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    merged = None
    for cid in collections:
        c = _build_landsat_collection(ee, cid, region_rect, year, cloud_max)
        merged = c if merged is None else merged.merge(c)

    if merged is None:
        raise RuntimeError("No Landsat collections constructed.")

    contract = str(feature_contract).strip().lower()
    if contract == "paper_rgbnir_v1":
        final = merged.select(["blue", "green", "red", "nir"]).median().rename(_feature_band_names(contract))
        return final.clip(region_rect).toFloat(), region_rect

    if contract != "stats_v1":
        raise RuntimeError(f"Unsupported feature_contract={feature_contract}")

    median = merged.select(["blue", "green", "red", "nir", "swir1", "swir2"]).median()
    median = median.rename(_feature_band_names(contract)[:6])

    ndvi_col = merged.map(lambda img: img.normalizedDifference(["nir", "red"]).rename("ndvi"))
    ndwi_col = merged.map(lambda img: img.normalizedDifference(["green", "nir"]).rename("ndwi"))

    ndvi_pct = ndvi_col.reduce(ee.Reducer.percentile([10, 50, 90]))
    ndwi_pct = ndwi_col.reduce(ee.Reducer.percentile([50]))

    ndvi_amp = ndvi_pct.select("ndvi_p90").subtract(ndvi_pct.select("ndvi_p10")).rename("ndvi_amp")
    final = median.addBands(
        [
            ndvi_pct.select("ndvi_p10").rename("ndvi_p10"),
            ndvi_pct.select("ndvi_p50").rename("ndvi_p50"),
            ndvi_pct.select("ndvi_p90").rename("ndvi_p90"),
            ndvi_amp,
            ndwi_pct.select("ndwi_p50").rename("ndwi_p50"),
        ]
    ).clip(region_rect).toFloat()

    return final, region_rect


def _start_export_task(ee, image, region_rect, year: int, tile_id: str, drive_folder: str):
    desc = f"cpis_landsat_{year}_{tile_id}"
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=desc,
        folder=drive_folder,
        fileNamePrefix=desc,
        region=region_rect,
        crs="EPSG:4326",
        scale=30,
        fileFormat="GeoTIFF",
        maxPixels=1e13,
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task, desc


def _refresh_statuses(ee, tiles: dict[str, dict[str, Any]], log):
    for tile_id, row in tiles.items():
        task_id = row.get("task_id", "")
        if not task_id:
            continue
        try:
            raw = ee.data.getTaskStatus(task_id)
            if isinstance(raw, list) and raw:
                status = raw[0]
            else:
                status = {}
        except Exception as exc:  # pragma: no cover - remote API variance
            log(f"[warn] {tile_id}: status refresh failed for task {task_id}: {exc}")
            continue

        state = str(status.get("state", "")).upper()
        if state == "COMPLETED":
            row["status"] = EXPORT_STATUS_SUCCEEDED
            row["ended_at"] = row.get("ended_at") or utc_now_iso()
            row["error"] = ""
        elif state in {"FAILED", "CANCELLED"}:
            row["status"] = EXPORT_STATUS_FAILED
            row["ended_at"] = row.get("ended_at") or utc_now_iso()
            row["error"] = str(status.get("error_message", "task_failed"))
        elif state in {"READY", "RUNNING"}:
            row["status"] = EXPORT_STATUS_RUNNING
        log(f"[status] {tile_id}: {row['status']} ({state})")


def run_export_year(args: argparse.Namespace) -> int:
    year = int(args.year)
    collections = _collection_ids_for_year(year)
    region_path = Path(args.region)
    log = build_logger(args.log_file if args.log_file else "")
    log(
        f"Starting export-year for {year}; collections={collections}; "
        f"feature_contract={args.feature_contract}"
    )

    region_gdf = _load_region(region_path)
    region_union = region_gdf.geometry.union_all() if hasattr(region_gdf.geometry, "union_all") else region_gdf.unary_union
    run_dir = Path(args.run_dir) if args.run_dir else (DEFAULT_RUNS_DIR / "export" / str(year))
    ensure_dir(run_dir)
    manifest_path = Path(args.manifest) if args.manifest else (run_dir / "export_manifest.json")

    manifest = load_manifest(
        manifest_path,
        _manifest_template(year=year, region_path=str(region_path.resolve()), tile_size_km=float(args.tile_size_km), collections=collections),
    )
    manifest["source_collections"] = collections
    manifest["region"] = str(region_path.resolve())
    manifest["tile_size_km"] = float(args.tile_size_km)
    manifest["feature_contract"] = str(args.feature_contract)
    manifest["band_names"] = _feature_band_names(args.feature_contract)
    manifest.setdefault("tiles", {})

    if manifest["tiles"] and not args.rebuild_tiles:
        log(f"Using existing manifest tile set: {len(manifest['tiles'])} tiles")
    else:
        tiles = _tile_rows(region_union, float(args.tile_size_km), int(args.max_tiles))
        if not tiles:
            raise RuntimeError("No intersecting tiles generated for region.")
        log(f"Generated {len(tiles)} intersecting tiles at tile_size_km={args.tile_size_km}")
        manifest["tiles"] = {}
        for row in tiles:
            manifest["tiles"][row["tile_id"]] = row

    if args.skip_local_dir:
        local_hits = _scan_local_tiles(list(args.skip_local_dir), log=log)
        for tile_id, row in manifest["tiles"].items():
            matches = local_hits.get(tile_id, [])
            row["local_exists"] = bool(matches)
            row["local_files"] = matches[:8]
    else:
        local_hits = {}

    ee = None
    if args.start_tasks or args.refresh_status:
        ee = _init_ee(args.project if args.project else None)
        log("Earth Engine initialized")

    lock_path = run_dir / ".export_manifest.lock.json"
    with manifest_lock(lock_path=lock_path, stale_seconds=int(args.stale_lock_seconds), force=bool(args.force_lock)):
        if args.refresh_status and ee is not None:
            _refresh_statuses(ee, manifest["tiles"], log)
            save_manifest(manifest_path, manifest)
            counts = tile_status_counts(manifest["tiles"])
            log(f"Status refresh complete. counts={counts}")

        if args.start_tasks:
            started = 0
            for tile_id, row in manifest["tiles"].items():
                if int(args.max_start_tasks) > 0 and started >= int(args.max_start_tasks):
                    log(f"Reached max_start_tasks={int(args.max_start_tasks)}; stopping task submission")
                    break
                status = row.get("status", EXPORT_STATUS_QUEUED)
                if bool(row.get("local_exists", False)):
                    continue
                retries = int(row.get("retries", 0))
                should_skip = args.resume and status == EXPORT_STATUS_SUCCEEDED
                if should_skip:
                    continue

                if status == EXPORT_STATUS_FAILED and retries >= int(args.max_retries):
                    continue

                if status in {EXPORT_STATUS_RUNNING, EXPORT_STATUS_SUCCEEDED}:
                    continue

                if status == EXPORT_STATUS_FAILED:
                    row["status"] = EXPORT_STATUS_RETRIED
                    row["retries"] = retries + 1
                bbox = row["bbox"]
                image, region_rect = _build_feature_image(
                    ee=ee,
                    collections=collections,
                    bbox=bbox,
                    year=year,
                    cloud_max=float(args.cloud_max),
                    feature_contract=str(args.feature_contract),
                )
                task, desc = _start_export_task(
                    ee=ee,
                    image=image,
                    region_rect=region_rect,
                    year=year,
                    tile_id=tile_id,
                    drive_folder=args.drive_folder,
                )
                row["task_id"] = str(task.id)
                row["description"] = desc
                row["status"] = EXPORT_STATUS_RUNNING
                row["started_at"] = utc_now_iso()
                row["error"] = ""
                started += 1
                log(f"[start] {tile_id} task_id={task.id}")
            log(f"Started {started} tasks")
        else:
            log("start_tasks not enabled; manifest updated only.")

        save_manifest(manifest_path, manifest)
    counts = tile_status_counts(manifest["tiles"])
    log(f"Wrote manifest: {manifest_path}")
    log(f"Tile status counts: {counts}")
    return 0


def build_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "export-year",
        help="Queue or manage yearly Landsat composite exports for arid SSA tiles",
    )
    p.add_argument("--year", type=int, required=True, help="Target year")
    p.add_argument("--region", default=str(DEFAULT_REGION_GEOJSON), help="Region GeoJSON path")
    p.add_argument("--tile-size-km", type=float, default=200.0, help="Approx tile size in km")
    p.add_argument("--cloud-max", type=float, default=40.0, help="Max CLOUD_COVER per scene")
    p.add_argument(
        "--feature-contract",
        choices=["stats_v1", "paper_rgbnir_v1"],
        default="stats_v1",
        help="Band contract for exported composites",
    )
    p.add_argument("--drive-folder", default="CPIS_LANDSAT_EXPORTS", help="GDrive folder for exports")
    p.add_argument("--project", default="africa-irrigation-mine", help="GEE project id")
    p.add_argument("--run-dir", default="", help="Run directory (default: runs/export/{year})")
    p.add_argument("--manifest", default="", help="Export manifest path")
    p.add_argument(
        "--skip-local-dir",
        action="append",
        default=[],
        help="Local directory to scan for existing tile TIFFs; matched tiles will not be started again",
    )
    p.add_argument("--resume", action="store_true", help="Skip tiles already succeeded")
    p.add_argument("--rebuild-tiles", action="store_true", help="Rebuild tile list in manifest from region grid")
    p.add_argument("--start-tasks", action="store_true", help="Actually queue GEE tasks")
    p.add_argument("--max-start-tasks", type=int, default=0, help="Limit tasks started in this invocation; 0 means no cap")
    p.add_argument("--refresh-status", action="store_true", help="Refresh statuses from GEE task API")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries for failed tiles")
    p.add_argument("--max-tiles", type=int, default=0, help="Limit number of generated tiles (smoke runs)")
    p.add_argument("--stale-lock-seconds", type=int, default=7200, help="Lock recovery timeout")
    p.add_argument("--force-lock", action="store_true", help="Force lock takeover")
    p.add_argument("--log-file", default="", help="Optional log file")
    p.set_defaults(func=run_export_year)
