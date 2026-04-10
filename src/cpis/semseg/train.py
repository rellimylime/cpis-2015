"""Train U-Net for binary pivot segmentation on Landsat 2015 tiles."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cpis.semseg.model import UNet

logger = logging.getLogger(__name__)


class PivotSegDataset(Dataset):
    """Loads Landsat 4-band tiles paired with rasterized binary pivot masks."""

    def __init__(self, tile_dir: Path, mask_dir: Path, tile_list: list[str],
                 patch_size: int = 256, patches_per_tile: int = 16):
        self.tile_dir = Path(tile_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_list = tile_list
        self.patch_size = patch_size
        self.patches_per_tile = patches_per_tile

    def __len__(self):
        return len(self.tile_list) * self.patches_per_tile

    def __getitem__(self, idx):
        tile_idx = idx // self.patches_per_tile
        tile_name = self.tile_list[tile_idx]

        with rasterio.open(self.tile_dir / f"{tile_name}.tif") as src:
            img = src.read().astype(np.float32)  # (4, H, W)

        with rasterio.open(self.mask_dir / f"{tile_name}_mask.tif") as src:
            mask = src.read(1).astype(np.float32)  # (H, W)

        # Random crop
        _, h, w = img.shape
        ps = self.patch_size
        y = np.random.randint(0, max(h - ps, 1))
        x = np.random.randint(0, max(w - ps, 1))
        img_patch = img[:, y:y + ps, x:x + ps]
        mask_patch = mask[y:y + ps, x:x + ps]

        # Pad if tile is smaller than patch_size
        if img_patch.shape[1] < ps or img_patch.shape[2] < ps:
            pad_img = np.zeros((4, ps, ps), dtype=np.float32)
            pad_mask = np.zeros((ps, ps), dtype=np.float32)
            pad_img[:, :img_patch.shape[1], :img_patch.shape[2]] = img_patch
            pad_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch
            img_patch, mask_patch = pad_img, pad_mask

        return torch.from_numpy(img_patch), torch.from_numpy(mask_patch).unsqueeze(0)


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load tile lists
    with open(args.train_tiles) as f:
        train_tiles = [line.strip() for line in f if line.strip()]
    with open(args.val_tiles) as f:
        val_tiles = [line.strip() for line in f if line.strip()]

    train_ds = PivotSegDataset(args.tile_dir, args.mask_dir, train_tiles,
                               patch_size=args.patch_size)
    val_ds = PivotSegDataset(args.tile_dir, args.mask_dir, val_tiles,
                             patch_size=args.patch_size, patches_per_tile=4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=4, base_filters=args.base_filters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
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

        # Validate
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

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_dir / "best.pth")
            logger.info(f"  saved best model (val_loss={val_loss:.4f})")

        torch.save(model.state_dict(), run_dir / "latest.pth")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net pivot segmentation")
    parser.add_argument("--tile-dir", required=True, help="Directory of Landsat .tif tiles")
    parser.add_argument("--mask-dir", required=True, help="Directory of rasterized binary masks")
    parser.add_argument("--train-tiles", required=True, help="Text file listing train tile names")
    parser.add_argument("--val-tiles", required=True, help="Text file listing val tile names")
    parser.add_argument("--run-dir", default="runs/semseg", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="BCE positive class weight (pivots are sparse)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args)


if __name__ == "__main__":
    main()
