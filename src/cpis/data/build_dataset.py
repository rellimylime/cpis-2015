"""Build a one-category COCO dataset from multiband rasters and polygon truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from osgeo import gdal
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import transform as shapely_transform


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cpis.common.file_utils import ensure_dir, save_json  # noqa: E402
from cpis.common.logging_utils import build_logger  # noqa: E402


gdal.UseExceptions()

ONE_CAT_TEMPLATE = {
    "info": {
        "description": "CPIS one-category dataset",
        "version": "rse2023_2015_v1",
    },
    "licenses": [{"id": 0, "name": None, "url": None}],
    "type": "instances",
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "crops_completed_circle", "supercategory": None}],
}

CHUNK_SUFFIX_RE = re.compile(r"^(?P<base>.+)-\d{10}-\d{10}$")


def _iter_polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for part in geom.geoms:
            out.extend(_iter_polygon_parts(part))
        return out
    return []


def _window_starts(total: int, chip_size: int, overlap: int) -> list[int]:
    if total <= chip_size:
        return [0]
    stride = max(1, chip_size - overlap)
    starts = list(range(0, max(1, total - chip_size + 1), stride))
    last = total - chip_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(int(v) for v in starts))


def _window_geotransform(gt: tuple[float, float, float, float, float, float], xoff: int, yoff: int):
    return (
        gt[0] + (xoff * gt[1]) + (yoff * gt[2]),
        gt[1],
        gt[2],
        gt[3] + (xoff * gt[4]) + (yoff * gt[5]),
        gt[4],
        gt[5],
    )


def _window_bounds(gt: tuple[float, float, float, float, float, float], xoff: int, yoff: int, xsize: int, ysize: int):
    xs = [xoff, xoff + xsize]
    ys = [yoff, yoff + ysize]
    corners = []
    for px in xs:
        for py in ys:
            mx = gt[0] + (px * gt[1]) + (py * gt[2])
            my = gt[3] + (px * gt[4]) + (py * gt[5])
            corners.append((mx, my))
    minx = min(x for x, _ in corners)
    maxx = max(x for x, _ in corners)
    miny = min(y for _, y in corners)
    maxy = max(y for _, y in corners)
    return box(minx, miny, maxx, maxy)


def _inverse_geotransform(gt: tuple[float, float, float, float, float, float]):
    mat = np.array([[float(gt[1]), float(gt[2])], [float(gt[4]), float(gt[5])]], dtype=float)
    inv = np.linalg.inv(mat)
    origin = np.array([float(gt[0]), float(gt[3])], dtype=float)
    return inv, origin


def _map_geom_to_chip_pixels(geom, *, inv_mat, origin, xoff: int, yoff: int):
    def _fn(x, y, z=None):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        pts = np.vstack([x_arr - origin[0], y_arr - origin[1]])
        px_py = inv_mat @ pts
        px = px_py[0] - float(xoff)
        py = px_py[1] - float(yoff)
        return px, py

    return shapely_transform(_fn, geom)


def _polygon_to_coco_parts(geom_px, *, min_area_px: float) -> list[dict[str, object]]:
    anns: list[dict[str, object]] = []
    for part in _iter_polygon_parts(geom_px):
        if part.is_empty:
            continue
        coords = list(part.exterior.coords)
        if len(coords) < 4:
            continue
        flat: list[float] = []
        for x, y in coords[:-1]:
            flat.extend([float(x), float(y)])
        if len(flat) < 6:
            continue
        minx, miny, maxx, maxy = part.bounds
        area = float(part.area)
        if area < min_area_px:
            continue
        anns.append(
            {
                "segmentation": [flat],
                "bbox": [float(minx), float(miny), float(maxx - minx), float(maxy - miny)],
                "area": area,
            }
        )
    return anns


def _split_rank(source_name: str) -> int:
    digest = hashlib.md5(source_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _source_name_for_path(path_like: str | Path) -> str:
    stem = Path(path_like).stem
    match = CHUNK_SUFFIX_RE.match(stem)
    if match:
        return str(match.group("base"))
    return str(stem)


def _build_source_split_map(source_names: list[str], val_fraction: float) -> dict[str, str]:
    ordered = sorted(set(str(name) for name in source_names))
    if not ordered or val_fraction <= 0.0:
        return {name: "train" for name in ordered}

    if len(ordered) == 1:
        return {ordered[0]: "train"}

    ranked = sorted(ordered, key=_split_rank)
    n_val = int(round(len(ranked) * float(val_fraction)))
    n_val = max(1, n_val)
    n_val = min(len(ranked) - 1, n_val)
    val_names = set(ranked[:n_val])
    return {name: ("val" if name in val_names else "train") for name in ordered}


def _read_source_list(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(_source_name_for_path(line))
    return names


def _resolve_source_splits(
    tif_paths: list[Path],
    *,
    val_fraction: float,
    val_source_file: str,
    log,
) -> dict[str, str]:
    source_names = [_source_name_for_path(p) for p in tif_paths]
    split_map = _build_source_split_map(source_names, float(val_fraction))
    if not val_source_file:
        return split_map

    val_path = Path(val_source_file)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation source file not found: {val_path}")
    requested_val = _read_source_list(val_path)
    if not requested_val:
        raise RuntimeError(f"Validation source file is empty: {val_path}")

    known_sources = set(source_names)
    unknown = sorted(requested_val - known_sources)
    if unknown:
        raise RuntimeError(
            f"Validation source file listed {len(unknown)} unknown rasters, e.g. {unknown[:5]}"
        )

    split_map = {name: ("val" if name in requested_val else "train") for name in source_names}
    log(
        f"Using explicit validation-source override from {val_path}: "
        f"val_sources={len(requested_val)} train_sources={len(source_names) - len(requested_val)}"
    )
    return split_map


def _write_chip(ds, array: np.ndarray, out_path: Path, gt, proj_wkt: str) -> None:
    ensure_dir(out_path.parent)
    driver = gdal.GetDriverByName("GTiff")
    bands, height, width = array.shape
    dtype = ds.GetRasterBand(1).DataType
    out_ds = driver.Create(str(out_path), width, height, bands, dtype, options=["COMPRESS=LZW"])
    if out_ds is None:
        raise RuntimeError(f"Could not create chip: {out_path}")
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj_wkt)
    for band_idx in range(bands):
        src_band = ds.GetRasterBand(band_idx + 1)
        dst_band = out_ds.GetRasterBand(band_idx + 1)
        dst_band.WriteArray(array[band_idx])
        nodata = src_band.GetNoDataValue()
        if nodata is not None:
            dst_band.SetNoDataValue(float(nodata))
    out_ds.FlushCache()
    out_ds = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagery-dir", required=True, help="Directory containing multiband GeoTIFFs")
    ap.add_argument("--labels", required=True, help="Polygon truth layer (.shp/.gpkg)")
    ap.add_argument("--out-root", required=True, help="Output dataset root")
    ap.add_argument("--chip-size", type=int, default=1024, help="Chip size in pixels")
    ap.add_argument("--chip-overlap", type=int, default=128, help="Chip overlap in pixels")
    ap.add_argument("--val-fraction", type=float, default=0.20, help="Validation split fraction by source raster")
    ap.add_argument(
        "--val-sources-file",
        default="",
        help="Optional newline-delimited list of source raster names/stems to force into validation; all others go to train",
    )
    ap.add_argument("--keep-empty", action="store_true", help="Keep chips with no intersecting truth polygons")
    ap.add_argument("--min-ann-area-px", type=float, default=16.0, help="Minimum clipped instance area in pixels")
    ap.add_argument("--max-chips-per-raster", type=int, default=0, help="Limit chips per raster for smoke runs")
    ap.add_argument("--log-file", default="", help="Optional log file")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    log = build_logger(args.log_file if args.log_file else (out_root / "build_paper_dataset.log"))
    imagery_dir = Path(args.imagery_dir)
    if not imagery_dir.exists():
        raise FileNotFoundError(f"Imagery dir not found: {imagery_dir}")

    label_gdf = gpd.read_file(args.labels)
    if label_gdf.empty:
        raise RuntimeError(f"Labels layer is empty: {args.labels}")
    if label_gdf.crs is None:
        raise RuntimeError(f"Labels layer missing CRS: {args.labels}")

    tif_paths = sorted(p for p in imagery_dir.glob("*.tif") if p.is_file())
    if not tif_paths:
        raise RuntimeError(f"No GeoTIFFs found in {imagery_dir}")
    split_map = _resolve_source_splits(
        tif_paths,
        val_fraction=float(args.val_fraction),
        val_source_file=str(args.val_sources_file),
        log=log,
    )
    val_sources = sum(1 for split in split_map.values() if split == "val")
    train_sources = sum(1 for split in split_map.values() if split == "train")
    log(
        f"Source split assignment complete: train_sources={train_sources} "
        f"val_sources={val_sources} val_fraction={float(args.val_fraction)}"
    )

    ensure_dir(out_root / "train" / "images")
    ensure_dir(out_root / "val" / "images")
    ensure_dir(out_root / "annotations")

    label_cache: dict[str, gpd.GeoDataFrame] = {}
    datasets = {
        "train": json.loads(json.dumps(ONE_CAT_TEMPLATE)),
        "val": json.loads(json.dumps(ONE_CAT_TEMPLATE)),
    }
    image_id = 1
    ann_id = 1
    totals = {"train_images": 0, "val_images": 0, "train_annotations": 0, "val_annotations": 0}
    nonfinite_chip_count = 0
    source_stats: dict[str, dict[str, int | str]] = {}

    for tif_path in tif_paths:
        ds = gdal.Open(str(tif_path))
        if ds is None:
            raise RuntimeError(f"Could not open raster: {tif_path}")
        proj_wkt = ds.GetProjectionRef()
        gt = ds.GetGeoTransform()
        width = int(ds.RasterXSize)
        height = int(ds.RasterYSize)
        band_count = int(ds.RasterCount)
        source_name = _source_name_for_path(tif_path)
        split = split_map.get(source_name, "train")

        if proj_wkt not in label_cache:
            label_cache[proj_wkt] = label_gdf.to_crs(proj_wkt)
        labels_proj = label_cache[proj_wkt]
        raster_bbox = _window_bounds(gt, 0, 0, width, height)
        labels_raster = labels_proj[labels_proj.geometry.intersects(raster_bbox)].copy()
        inv_mat, origin = _inverse_geotransform(gt)
        stats = source_stats.setdefault(
            source_name,
            {
                "split": split,
                "bands": band_count,
                "source_files": 0,
                "labels_in_raster": 0,
                "chips_written": 0,
                "annotations_written": 0,
                "empty_chips_skipped": 0,
                "sanitized_nonfinite_chips": 0,
            },
        )
        stats["source_files"] += 1
        stats["labels_in_raster"] += int(len(labels_raster))

        chip_counter = 0
        for yoff in _window_starts(height, int(args.chip_size), int(args.chip_overlap)):
            for xoff in _window_starts(width, int(args.chip_size), int(args.chip_overlap)):
                xsize = min(int(args.chip_size), width - int(xoff))
                ysize = min(int(args.chip_size), height - int(yoff))
                chip_bbox = _window_bounds(gt, int(xoff), int(yoff), int(xsize), int(ysize))
                chip_labels = labels_raster[labels_raster.geometry.intersects(chip_bbox)].copy()

                ann_payloads: list[dict[str, object]] = []
                for _, label_row in chip_labels.iterrows():
                    clipped = label_row.geometry.intersection(chip_bbox)
                    if clipped.is_empty:
                        continue
                    clipped_px = _map_geom_to_chip_pixels(
                        clipped,
                        inv_mat=inv_mat,
                        origin=origin,
                        xoff=int(xoff),
                        yoff=int(yoff),
                    )
                    ann_payloads.extend(_polygon_to_coco_parts(clipped_px, min_area_px=float(args.min_ann_area_px)))

                if not ann_payloads and not bool(args.keep_empty):
                    stats["empty_chips_skipped"] += 1
                    continue

                array = ds.ReadAsArray(int(xoff), int(yoff), int(xsize), int(ysize))
                if array is None:
                    raise RuntimeError(f"Failed reading raster window for {tif_path}")
                if array.ndim == 2:
                    array = array[np.newaxis, ...]
                if int(array.shape[0]) != band_count:
                    raise RuntimeError(f"Unexpected band count from window read: {array.shape} vs {band_count}")
                if not np.isfinite(array).all():
                    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                    nonfinite_chip_count += 1
                    stats["sanitized_nonfinite_chips"] += 1

                chip_name = f"{tif_path.stem}__x{int(xoff):05d}_y{int(yoff):05d}.tif"
                chip_path = out_root / split / "images" / chip_name
                chip_gt = _window_geotransform(gt, int(xoff), int(yoff))
                _write_chip(ds, array, chip_path, chip_gt, proj_wkt)

                datasets[split]["images"].append(
                    {
                        "height": int(ysize),
                        "width": int(xsize),
                        "id": int(image_id),
                        "file_name": chip_name,
                        "source_name": source_name,
                        "source_raster": tif_path.name,
                        "xoff": int(xoff),
                        "yoff": int(yoff),
                    }
                )
                totals[f"{split}_images"] += 1
                stats["chips_written"] += 1
                for ann in ann_payloads:
                    datasets[split]["annotations"].append(
                        {
                            "id": int(ann_id),
                            "image_id": int(image_id),
                            "category_id": 0,
                            "iscrowd": 0,
                            "segmentation": ann["segmentation"],
                            "bbox": ann["bbox"],
                            "area": ann["area"],
                        }
                    )
                    ann_id += 1
                    totals[f"{split}_annotations"] += 1
                    stats["annotations_written"] += 1
                image_id += 1
                chip_counter += 1

                if int(args.max_chips_per_raster) > 0 and chip_counter >= int(args.max_chips_per_raster):
                    break
            if int(args.max_chips_per_raster) > 0 and chip_counter >= int(args.max_chips_per_raster):
                break

        log(
            f"Processed {tif_path.name}: source={source_name} split={split} chips={chip_counter} "
            f"labels_in_raster={len(labels_raster)} bands={band_count}"
        )

    train_json = out_root / "annotations" / "one_cat_train.json"
    val_json = out_root / "annotations" / "one_cat_val.json"
    with train_json.open("w", encoding="utf-8") as f:
        json.dump(datasets["train"], f, indent=2)
    with val_json.open("w", encoding="utf-8") as f:
        json.dump(datasets["val"], f, indent=2)

    summary = {
        "imagery_dir": str(imagery_dir.resolve()),
        "labels": str(Path(args.labels).resolve()),
        "chip_size": int(args.chip_size),
        "chip_overlap": int(args.chip_overlap),
        "val_fraction": float(args.val_fraction),
        "val_sources_file": str(Path(args.val_sources_file).resolve()) if args.val_sources_file else "",
        "source_splits": split_map,
        "source_stats": source_stats,
        "keep_empty": bool(args.keep_empty),
        "min_ann_area_px": float(args.min_ann_area_px),
        "sanitized_nonfinite_chips": int(nonfinite_chip_count),
        "totals": totals,
        "outputs": {
            "train_json": str(train_json.resolve()),
            "val_json": str(val_json.resolve()),
            "train_images": str((out_root / "train" / "images").resolve()),
            "val_images": str((out_root / "val" / "images").resolve()),
        },
    }
    save_json(out_root / "dataset_manifest.json", summary)
    log(f"Wrote dataset manifest: {out_root / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
