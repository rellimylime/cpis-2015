"""Rasterize pivot polygon labels to binary masks aligned to Landsat tiles.

Used by the semantic segmentation branch to create training masks from
the anchor truth or manually reviewed label layers.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def rasterize_labels_for_tile(tile_path: Path, labels_gdf: gpd.GeoDataFrame,
                              output_path: Path):
    """Burn pivot polygons into a binary mask matching a tile's grid."""
    with rasterio.open(tile_path) as src:
        transform = src.transform
        shape = (src.height, src.width)
        profile = src.profile.copy()
        tile_bounds = src.bounds

    # Clip labels to tile extent
    from shapely.geometry import box
    tile_box = box(*tile_bounds)
    clipped = labels_gdf[labels_gdf.intersects(tile_box)]

    if len(clipped) == 0:
        mask = np.zeros(shape, dtype=np.uint8)
    else:
        geoms = [(geom, 1) for geom in clipped.geometry if geom is not None]
        mask = rasterize(geoms, out_shape=shape, transform=transform,
                         fill=0, dtype=np.uint8)

    profile.update(count=1, dtype="uint8", nodata=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    return int(mask.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize pivot labels to binary masks for semseg training")
    parser.add_argument("--tile-dir", required=True,
                        help="Directory of Landsat .tif tiles")
    parser.add_argument("--labels", required=True,
                        help="Path to label polygons (shapefile or gpkg)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output mask .tif files")
    parser.add_argument("--tile-list", default=None,
                        help="Optional text file listing tile names to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    labels_gdf = gpd.read_file(args.labels)
    tile_dir = Path(args.tile_dir)
    output_dir = Path(args.output_dir)

    if args.tile_list:
        with open(args.tile_list) as f:
            names = [line.strip() for line in f if line.strip()]
        tiles = [tile_dir / f"{n}.tif" for n in names]
    else:
        tiles = sorted(tile_dir.glob("*.tif"))

    for tile_path in tqdm(tiles, desc="rasterize"):
        if not tile_path.exists():
            logger.warning(f"missing: {tile_path}")
            continue
        out_name = tile_path.stem + "_mask.tif"
        n_px = rasterize_labels_for_tile(
            tile_path, labels_gdf, output_dir / out_name)
        logger.info(f"{tile_path.name}: {n_px} pivot pixels")


if __name__ == "__main__":
    main()
