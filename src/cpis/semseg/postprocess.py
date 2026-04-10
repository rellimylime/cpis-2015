"""Convert binary pivot masks to polygon inventory via connected components + circle fitting."""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.measure import label, regionprops
from tqdm import tqdm

logger = logging.getLogger(__name__)

# At 30m, a 50m-radius pivot is ~3px across (noise). 100m radius is ~7px.
MIN_AREA_PX = 30       # ~27,000 m² at 30m resolution
MAX_AREA_PX = 2000     # ~1,800,000 m² — anything bigger is not a single pivot
MIN_CIRCULARITY = 0.4  # reject very elongated blobs


def mask_to_pivots(mask_path: Path, min_area_px: int = MIN_AREA_PX,
                   max_area_px: int = MAX_AREA_PX,
                   min_circularity: float = MIN_CIRCULARITY):
    """Extract pivot polygons from a binary mask GeoTIFF.

    Returns a GeoDataFrame with columns: geometry, center_x, center_y,
    equiv_radius_m, area_m2, circularity, tile.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        res = src.res[0]  # pixel size in CRS units (meters for UTM)

    labeled = label(mask, connectivity=1)
    props = regionprops(labeled)

    records = []
    for prop in props:
        if prop.area < min_area_px or prop.area > max_area_px:
            continue

        # Circularity: 4π·area / perimeter²
        circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2 + 1e-6)
        if circularity < min_circularity:
            continue

        # Extract polygon via rasterio
        component_mask = (labeled == prop.label).astype(np.uint8)
        polys = []
        for geom, val in rio_shapes(component_mask, mask=component_mask,
                                     transform=transform):
            if val == 1:
                polys.append(shape(geom))

        if not polys:
            continue

        polygon = unary_union(polys)
        centroid = polygon.centroid
        area_m2 = prop.area * (res ** 2)
        equiv_radius_m = np.sqrt(area_m2 / np.pi)

        records.append({
            "geometry": polygon,
            "center_x": centroid.x,
            "center_y": centroid.y,
            "equiv_radius_m": round(equiv_radius_m, 1),
            "area_m2": round(area_m2, 1),
            "circularity": round(circularity, 3),
            "tile": mask_path.stem,
        })

    if not records:
        return gpd.GeoDataFrame(columns=[
            "geometry", "center_x", "center_y", "equiv_radius_m",
            "area_m2", "circularity", "tile"
        ], crs=crs)

    return gpd.GeoDataFrame(records, crs=crs)


def process_all_masks(mask_dir: Path, output_path: Path, **kwargs):
    """Process all mask tiles and merge into a single inventory."""
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

    merged = gpd.GeoDataFrame(
        data=__import__("pandas").concat(all_gdfs, ignore_index=True),
        crs=all_gdfs[0].crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_path, driver="GPKG")
    logger.info(f"Saved {len(merged)} pivots to {output_path}")

    # Summary
    summary = {
        "total_pivots": len(merged),
        "tiles_with_pivots": merged["tile"].nunique(),
        "median_radius_m": round(float(merged["equiv_radius_m"].median()), 1),
        "mean_circularity": round(float(merged["circularity"].mean()), 3),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert binary masks to pivot polygon inventory")
    parser.add_argument("--mask-dir", required=True,
                        help="Directory of binary mask GeoTIFFs")
    parser.add_argument("--output", default="outputs/semseg/pivots_2015.gpkg",
                        help="Output GeoPackage path")
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
