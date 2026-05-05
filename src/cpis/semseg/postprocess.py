"""Convert binary pivot masks to polygon inventory via vectorization + shape filtering."""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from tqdm import tqdm

logger = logging.getLogger(__name__)

MIN_AREA_PX = 30
MAX_AREA_PX = 2000
MIN_CIRCULARITY = 0.4


def mask_to_pivots(mask_path: Path, min_area_px: int = MIN_AREA_PX,
                   max_area_px: int = MAX_AREA_PX,
                   min_circularity: float = MIN_CIRCULARITY):
    """Extract pivot polygons from a binary mask GeoTIFF.

    Uses rasterio.features.shapes (GDAL-backed) instead of skimage.label
    so it stays fast on large tiles.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        res = src.res[0]  # degrees/pixel for EPSG:4326 tiles

    records = []
    for geom_dict, val in rio_shapes(mask, mask=mask, transform=transform):
        if val != 1:
            continue
        poly = shape(geom_dict)

        # Area in pixels (res is degrees/px; poly.area is in degrees²)
        area_px = poly.area / (res ** 2)
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # Circularity is dimensionless — correct regardless of CRS units
        circularity = (4 * np.pi * poly.area) / (poly.length ** 2 + 1e-12)
        if circularity < min_circularity:
            continue

        centroid = poly.centroid
        # equiv_radius in degrees (approximate; only used for summary stats)
        equiv_radius_deg = np.sqrt(poly.area / np.pi)

        records.append({
            "geometry": poly,
            "center_lon": round(centroid.x, 6),
            "center_lat": round(centroid.y, 6),
            "equiv_radius_deg": round(equiv_radius_deg, 6),
            "area_px": round(area_px, 1),
            "circularity": round(circularity, 3),
            "tile": mask_path.stem,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=["geometry", "center_lon", "center_lat",
                     "equiv_radius_deg", "area_px", "circularity", "tile"],
            crs=crs,
        )

    return gpd.GeoDataFrame(records, crs=crs)


def process_all_masks(mask_dir: Path, output_path: Path, **kwargs):
    mask_files = sorted(mask_dir.glob("*.tif"))
    all_gdfs = []

    for mask_path in tqdm(mask_files, desc="polygonize"):
        gdf = mask_to_pivots(mask_path, **kwargs)
        if len(gdf) > 0:
            all_gdfs.append(gdf)
            logger.info(f"{mask_path.name}: {len(gdf)} pivots")
        else:
            logger.info(f"{mask_path.name}: no pivots")

    if not all_gdfs:
        logger.warning("No pivots detected in any tile")
        return

    import pandas as pd
    merged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True),
                              crs=all_gdfs[0].crs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_path, driver="GPKG")
    logger.info(f"Saved {len(merged)} pivots to {output_path}")

    summary = {
        "total_pivots": len(merged),
        "tiles_with_pivots": merged["tile"].nunique(),
        "median_area_px": round(float(merged["area_px"].median()), 1),
        "mean_circularity": round(float(merged["circularity"].mean()), 3),
    }
    with open(output_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert binary masks to pivot polygon inventory")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output", default="outputs/semseg/pivots_2015.gpkg")
    parser.add_argument("--min-area-px", type=int, default=MIN_AREA_PX)
    parser.add_argument("--max-area-px", type=int, default=MAX_AREA_PX)
    parser.add_argument("--min-circularity", type=float, default=MIN_CIRCULARITY)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_all_masks(
        Path(args.mask_dir), Path(args.output),
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        min_circularity=args.min_circularity,
    )


if __name__ == "__main__":
    main()
