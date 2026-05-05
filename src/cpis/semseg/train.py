"""Train U-Net for binary pivot segmentation on Landsat 2015 tiles."""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from cpis.semseg.model import UNet

logger = logging.getLogger(__name__)


def compute_band_stats(tile_dir: Path, tile_list: list[str], n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-band mean and std from training tiles, ignoring NaN."""
    sums = np.zeros(n_channels, dtype=np.float64)
    sumsqs = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.float64)
    for tile_name in tile_list:
        with rasterio.open(tile_dir / f"{tile_name}.tif") as src:
            arr = src.read().astype(np.float64)
        for b in range(n_channels):
            vals = arr[b].ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            sums[b] += vals.sum()
            sumsqs[b] += (vals ** 2).sum()
            counts[b] += vals.size
    mean = sums / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(sumsqs / np.maximum(counts, 1) - mean ** 2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class PivotSegDataset(Dataset):
    """Landsat tiles + binary masks with balanced pivot/background sampling."""

    def __init__(self, tile_dir: Path, mask_dir: Path, tile_list: list[str],
                 n_channels: int,
                 patch_size: int = 256, patches_per_tile: int = 16,
                 pivot_sample_prob: float = 0.5,
                 negative_sample_prob: float = 0.0,
                 negative_centers_path: str | None = None,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 deterministic: bool = False, augment: bool = False):
        self.tile_dir = Path(tile_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_list = tile_list
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.patches_per_tile = patches_per_tile
        self.pivot_sample_prob = pivot_sample_prob
        self.negative_sample_prob = negative_sample_prob
        self.negative_centers_path = Path(negative_centers_path) if negative_centers_path else None
        self.mean = mean  # (n_channels,) or None
        self.std = std    # (n_channels,) or None
        self.deterministic = deterministic
        self.augment = augment

        # Pre-cache mask shapes and pivot pixel locations for fast sampling.
        self._pivot_coords: dict[str, np.ndarray] = {}
        self._negative_coords: dict[str, np.ndarray] = {}
        self._shapes: dict[str, tuple[int, int]] = {}
        negative_points = self._load_negative_centers(tile_list)
        for tile_name in tile_list:
            mask_path = self.mask_dir / f"{tile_name}_mask.tif"
            tile_path = self.tile_dir / f"{tile_name}.tif"
            if mask_path.exists():
                with rasterio.open(mask_path) as src:
                    self._shapes[tile_name] = (src.height, src.width)
                    mask = src.read(1)
                ys, xs = np.where(mask > 0)
                self._pivot_coords[tile_name] = np.stack([ys, xs], axis=1) if len(ys) > 0 else np.empty((0, 2), dtype=int)
            else:
                with rasterio.open(tile_path) as src:
                    self._shapes[tile_name] = (src.height, src.width)
                self._pivot_coords[tile_name] = np.empty((0, 2), dtype=int)
            self._negative_coords[tile_name] = self._map_negative_points_to_pixels(tile_path, negative_points.get(tile_name, []))

    def _load_negative_centers(self, tile_list: list[str]) -> dict[str, list[tuple[float, float]]]:
        out = {tile_name: [] for tile_name in tile_list}
        if self.negative_centers_path is None or not self.negative_centers_path.exists():
            return out

        gdf = gpd.read_file(self.negative_centers_path)
        if "tile" not in gdf.columns:
            raise ValueError(f"negative centers file missing 'tile' column: {self.negative_centers_path}")
        tile_set = set(tile_list)
        gdf = gdf[gdf["tile"].isin(tile_set)].copy()
        if len(gdf) == 0:
            return out
        if gdf.crs is None:
            raise ValueError(f"negative centers file missing CRS: {self.negative_centers_path}")
        if gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        for tile_name, geom in zip(gdf["tile"], gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            out[str(tile_name)].append((float(geom.x), float(geom.y)))
        return out

    def _map_negative_points_to_pixels(self, tile_path: Path, lonlat_points: list[tuple[float, float]]) -> np.ndarray:
        if not lonlat_points:
            return np.empty((0, 2), dtype=int)
        rows_cols = []
        with rasterio.open(tile_path) as src:
            for lon, lat in lonlat_points:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    rows_cols.append((int(row), int(col)))
        if not rows_cols:
            return np.empty((0, 2), dtype=int)
        arr = np.array(rows_cols, dtype=int)
        return np.unique(arr, axis=0)

    def __len__(self):
        return len(self.tile_list) * self.patches_per_tile

    def _get_crop_origin(self, tile_name: str, h: int, w: int, patch_idx: int) -> tuple[int, int]:
        ps = self.patch_size
        pivot_coords = self._pivot_coords[tile_name]
        negative_coords = self._negative_coords[tile_name]

        if self.deterministic:
            # Fixed grid: tile crops deterministically by patch_idx
            stride = ps // 2
            cols = max((w - ps) // stride + 1, 1)
            row = patch_idx // cols
            col = patch_idx % cols
            y = min(row * stride, max(h - ps, 0))
            x = min(col * stride, max(w - ps, 0))
            return y, x

        has_pivots = len(pivot_coords) > 0
        has_negatives = len(negative_coords) > 0
        draw = np.random.random()
        use_pivot = has_pivots and draw < self.pivot_sample_prob
        use_negative = (not use_pivot) and has_negatives and draw < (self.pivot_sample_prob + self.negative_sample_prob)
        if use_pivot:
            # Center crop on a random pivot pixel, jittered up to half patch size
            idx = np.random.randint(len(pivot_coords))
            cy, cx = pivot_coords[idx]
            jitter = ps // 2
            y = cy - ps // 2 + np.random.randint(-jitter, jitter + 1)
            x = cx - ps // 2 + np.random.randint(-jitter, jitter + 1)
        elif use_negative:
            idx = np.random.randint(len(negative_coords))
            cy, cx = negative_coords[idx]
            jitter = ps // 2
            y = cy - ps // 2 + np.random.randint(-jitter, jitter + 1)
            x = cx - ps // 2 + np.random.randint(-jitter, jitter + 1)
        else:
            y = np.random.randint(0, max(h - ps, 1))
            x = np.random.randint(0, max(w - ps, 1))

        y = int(np.clip(y, 0, max(h - ps, 0)))
        x = int(np.clip(x, 0, max(w - ps, 0)))
        return y, x

    def __getitem__(self, idx):
        tile_idx = idx // self.patches_per_tile
        patch_idx = idx % self.patches_per_tile
        tile_name = self.tile_list[tile_idx]

        h, w = self._shapes[tile_name]
        ps = self.patch_size
        y, x = self._get_crop_origin(tile_name, h, w, patch_idx)
        window = Window(x, y, min(ps, w - x), min(ps, h - y))

        with rasterio.open(self.tile_dir / f"{tile_name}.tif") as src:
            img_patch = src.read(window=window).astype(np.float32)
        np.nan_to_num(img_patch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        with rasterio.open(self.mask_dir / f"{tile_name}_mask.tif") as src:
            mask_patch = src.read(1, window=window).astype(np.float32)

        if self.mean is not None and self.std is not None:
            for b in range(self.n_channels):
                img_patch[b] = (img_patch[b] - self.mean[b]) / self.std[b]

        # Pad if smaller than patch_size (edge tiles)
        if img_patch.shape[1] < ps or img_patch.shape[2] < ps:
            pad_img = np.zeros((self.n_channels, ps, ps), dtype=np.float32)
            pad_mask = np.zeros((ps, ps), dtype=np.float32)
            pad_img[:, :img_patch.shape[1], :img_patch.shape[2]] = img_patch
            pad_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch
            img_patch, mask_patch = pad_img, pad_mask

        if self.augment:
            k = np.random.randint(0, 4)
            if k:
                img_patch = np.rot90(img_patch, k=k, axes=(1, 2))
                mask_patch = np.rot90(mask_patch, k=k, axes=(0, 1))
            if np.random.random() < 0.5:
                img_patch = img_patch[:, :, ::-1]
                mask_patch = mask_patch[:, ::-1]
            if np.random.random() < 0.5:
                img_patch = img_patch[:, ::-1, :]
                mask_patch = mask_patch[::-1, :]
            img_patch = np.ascontiguousarray(img_patch)
            mask_patch = np.ascontiguousarray(mask_patch)

        return torch.from_numpy(img_patch), torch.from_numpy(mask_patch).unsqueeze(0)


class TileGroupedSampler(Sampler):
    """Shuffle tile order each epoch but keep all patches from a tile contiguous.

    Keeps related window reads close together, which is friendlier to raster IO
    than fully random patch order.
    """

    def __init__(self, n_tiles: int, patches_per_tile: int):
        self.n_tiles = n_tiles
        self.patches_per_tile = patches_per_tile

    def __iter__(self):
        for tile_idx in torch.randperm(self.n_tiles).tolist():
            for patch_idx in range(self.patches_per_tile):
                yield tile_idx * self.patches_per_tile + patch_idx

    def __len__(self):
        return self.n_tiles * self.patches_per_tile


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_tiles) as f:
        train_tiles = [line.strip() for line in f if line.strip()]
    with open(args.val_tiles) as f:
        val_tiles = [line.strip() for line in f if line.strip()]

    if args.in_channels is not None:
        n_channels = args.in_channels
    else:
        with rasterio.open(Path(args.tile_dir) / f"{train_tiles[0]}.tif") as src:
            n_channels = src.count
        logger.info(f"Auto-detected in_channels={n_channels} from {train_tiles[0]}.tif")

    stats_path = run_dir / "band_stats.json"
    if stats_path.exists():
        logger.info(f"Loading existing band stats from {stats_path}")
        with open(stats_path) as f:
            stats = json.load(f)
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        if len(mean) != n_channels:
            raise ValueError(
                f"band_stats.json has {len(mean)} channels but in_channels={n_channels}; "
                f"delete {stats_path} to recompute, or pass --in-channels {len(mean)}"
            )
    else:
        logger.info(f"Computing per-band stats over {n_channels} channels from training tiles...")
        mean, std = compute_band_stats(Path(args.tile_dir), train_tiles, n_channels)
        with open(stats_path, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    logger.info(f"mean={mean.tolist()} std={std.tolist()}")

    train_ds = PivotSegDataset(
        args.tile_dir, args.mask_dir, train_tiles, n_channels,
        patch_size=args.patch_size, patches_per_tile=args.patches_per_tile,
        pivot_sample_prob=args.pivot_sample_prob,
        negative_sample_prob=args.negative_sample_prob,
        negative_centers_path=args.negative_centers,
        mean=mean, std=std,
        deterministic=False, augment=args.augment,
    )
    val_ds = PivotSegDataset(
        args.tile_dir, args.mask_dir, val_tiles, n_channels,
        patch_size=args.patch_size, patches_per_tile=args.patches_per_tile,
        pivot_sample_prob=0.0, mean=mean, std=std, deterministic=True,
    )

    train_sampler = TileGroupedSampler(len(train_tiles), patches_per_tile=args.patches_per_tile)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=n_channels, base_filters=args.base_filters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))

    best_val_loss = float("inf")
    start_epoch = 0
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]
            best_val_loss = ckpt["best_val_loss"]
            history = ckpt.get("history", [])
            logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            # Old format: model state_dict only — restore weights, fast-forward scheduler
            model.load_state_dict(ckpt)
            start_epoch = args.resume_epoch
            best_val_loss = args.resume_best_val
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(f"Resumed weights-only checkpoint, continuing from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch} train"):
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            loss = bce(pred, masks) + dice_loss(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                pred = model(imgs)
                loss = bce(pred, masks) + dice_loss(pred, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        logger.info(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, run_dir / "best.pth")
            logger.info(f"  saved best model (val_loss={val_loss:.4f})")

        torch.save(ckpt, run_dir / "latest.pth")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net pivot segmentation")
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--train-tiles", required=True)
    parser.add_argument("--val-tiles", required=True)
    parser.add_argument("--run-dir", default="runs/semseg")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-tile", type=int, default=16)
    parser.add_argument("--pivot-sample-prob", type=float, default=0.5)
    parser.add_argument("--negative-sample-prob", type=float, default=0.0)
    parser.add_argument("--negative-centers", default=None,
                        help="Optional GPKG/GeoJSON of mined hard-negative point centers with a tile column")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=None,
                        help="Number of input bands. If omitted, auto-detected from the first training tile.")
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    parser.add_argument("--resume-epoch", type=int, default=0,
                        help="epoch number of weights-only checkpoint (required if checkpoint has no epoch field)")
    parser.add_argument("--resume-best-val", type=float, default=float("inf"),
                        help="best val loss at resume point (for weights-only checkpoints)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args)


if __name__ == "__main__":
    main()
