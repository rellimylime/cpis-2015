"""Run U-Net inference on Landsat tiles and produce binary masks."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from cpis.semseg.model import UNet

logger = logging.getLogger(__name__)


def infer_tile(model, tile_path: Path, output_path: Path, device,
               patch_size: int = 256, overlap: int = 32, threshold: float = 0.5,
               mean: np.ndarray | None = None, std: np.ndarray | None = None,
               save_probs: bool = False):
    """Run sliding-window inference on a single tile.

    save_probs=True: write float32 probability map (for offline threshold sweep).
    save_probs=False: write uint8 binary mask thresholded at `threshold`.
    Always returns number of pixels >= threshold (for sanity checks / logging).
    """
    with rasterio.open(tile_path) as src:
        img = src.read().astype(np.float32)  # (n_channels, H, W)
        profile = src.profile.copy()

    n_channels = img.shape[0]
    np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if mean is not None and std is not None:
        if len(mean) != n_channels:
            raise ValueError(
                f"band_stats has {len(mean)} channels but tile {tile_path.name} has {n_channels}"
            )
        for b in range(n_channels):
            img[b] = (img[b] - mean[b]) / std[b]

    _, h, w = img.shape
    stride = patch_size - overlap
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                ye = min(y + patch_size, h)
                xe = min(x + patch_size, w)
                ys = max(ye - patch_size, 0)
                xs = max(xe - patch_size, 0)

                patch = img[:, ys:ye, xs:xe]
                ph, pw = patch.shape[1], patch.shape[2]
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((n_channels, patch_size, patch_size), dtype=np.float32)
                    padded[:, :ph, :pw] = patch
                    patch = padded

                inp = torch.from_numpy(patch).unsqueeze(0).to(device)
                pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

                prob_map[ys:ye, xs:xe] += pred[:ye - ys, :xe - xs]
                count_map[ys:ye, xs:xe] += 1.0

    count_map[count_map == 0] = 1.0
    prob_map /= count_map

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_probs:
        profile.update(count=1, dtype="float32", nodata=-1.0)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prob_map, 1)
    else:
        binary = (prob_map >= threshold).astype(np.uint8)
        profile.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(binary, 1)

    return int((prob_map >= threshold).sum())


def load_model(checkpoint_path: Path, base_filters: int, in_channels: int, device) -> UNet:
    model = UNet(in_channels=in_channels, base_filters=base_filters).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run U-Net inference")
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/semseg/masks")
    parser.add_argument("--band-stats", default=None,
                        help="Path to band_stats.json from training run")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=None,
                        help="Number of input bands. If omitted, derived from band_stats.json or first tile.")
    parser.add_argument("--save-probs", action="store_true",
                        help="Save float32 prob maps instead of binary masks")
    parser.add_argument("--tile-list", default=None,
                        help="Optional text file listing tile stems to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = None, None
    if args.band_stats:
        with open(args.band_stats) as f:
            stats = json.load(f)
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        logger.info(f"Loaded band stats: mean={mean.tolist()}")
    else:
        logger.warning("No --band-stats provided — inference will use raw pixel values (not recommended)")

    tile_dir = Path(args.tile_dir)
    output_dir = Path(args.output_dir)
    if args.tile_list:
        with open(args.tile_list) as f:
            names = [line.strip() for line in f if line.strip()]
        tiles = [tile_dir / f"{name}.tif" for name in names]
    else:
        tiles = sorted(tile_dir.glob("*.tif"))

    if args.in_channels is not None:
        n_channels = args.in_channels
    elif mean is not None:
        n_channels = len(mean)
    else:
        with rasterio.open(tiles[0]) as src:
            n_channels = src.count
        logger.info(f"Auto-detected in_channels={n_channels} from {tiles[0].name}")

    model = load_model(Path(args.checkpoint), args.base_filters, n_channels, device)

    if args.save_probs:
        logger.info("Saving float32 probability maps (no threshold applied to output)")

    total_px = 0
    for tile_path in tqdm(tiles, desc="inference"):
        out_path = output_dir / tile_path.name
        n_px = infer_tile(model, tile_path, out_path, device,
                          args.patch_size, args.overlap, args.threshold,
                          mean=mean, std=std, save_probs=args.save_probs)
        total_px += n_px
        logger.info(f"{tile_path.name}: {n_px} px >= {args.threshold}")

    logger.info(f"Done. {len(tiles)} tiles, total px >= {args.threshold}: {total_px}")


if __name__ == "__main__":
    main()
