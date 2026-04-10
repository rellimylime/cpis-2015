"""Run U-Net inference on Landsat tiles and produce binary masks."""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from cpis.semseg.model import UNet

logger = logging.getLogger(__name__)


def infer_tile(model, tile_path: Path, output_path: Path, device,
               patch_size: int = 256, overlap: int = 32, threshold: float = 0.5):
    """Run sliding-window inference on a single tile, write binary mask GeoTIFF."""
    with rasterio.open(tile_path) as src:
        img = src.read().astype(np.float32)  # (4, H, W)
        profile = src.profile.copy()

    _, h, w = img.shape
    stride = patch_size - overlap
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                ye, xe = min(y + patch_size, h), min(x + patch_size, w)
                ys, xs = ye - patch_size, xe - patch_size
                ys, xs = max(ys, 0), max(xs, 0)

                patch = img[:, ys:ye, xs:xe]
                # Pad if needed
                ph, pw = patch.shape[1], patch.shape[2]
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((4, patch_size, patch_size), dtype=np.float32)
                    padded[:, :ph, :pw] = patch
                    patch = padded

                inp = torch.from_numpy(patch).unsqueeze(0).to(device)
                pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

                prob_map[ys:ye, xs:xe] += pred[:ye - ys, :xe - xs]
                count_map[ys:ye, xs:xe] += 1.0

    count_map[count_map == 0] = 1.0
    prob_map /= count_map
    binary = (prob_map >= threshold).astype(np.uint8)

    profile.update(count=1, dtype="uint8", nodata=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(binary, 1)

    n_pivot_px = int(binary.sum())
    return n_pivot_px


def main():
    parser = argparse.ArgumentParser(description="Run U-Net inference")
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/semseg/masks")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--base-filters", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=4, base_filters=args.base_filters).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))

    tile_dir = Path(args.tile_dir)
    output_dir = Path(args.output_dir)
    tiles = sorted(tile_dir.glob("*.tif"))

    for tile_path in tqdm(tiles, desc="inference"):
        out_path = output_dir / tile_path.name
        n_px = infer_tile(model, tile_path, out_path, device,
                          args.patch_size, args.overlap, args.threshold)
        logger.info(f"{tile_path.name}: {n_px} pivot pixels")


if __name__ == "__main__":
    main()
