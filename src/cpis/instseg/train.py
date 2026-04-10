"""Train the paper-style instance-segmentation model on a one-category dataset."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(ROOT / "src")
REPO_ROOT = str(ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from cpis.common.file_utils import ensure_dir, save_json  # noqa: E402
from cpis.common.logging_utils import build_logger  # noqa: E402


gdal.UseExceptions()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _compute_band_stats(image_dir: Path, *, max_images: int, max_pixels_per_image: int, log) -> tuple[list[float], list[float]]:
    tif_paths = sorted(image_dir.glob("*.tif"))
    if not tif_paths:
        raise RuntimeError(f"No training images found in {image_dir}")
    if max_images > 0:
        tif_paths = tif_paths[:max_images]

    rng = np.random.default_rng(42)
    sums = None
    sumsqs = None
    counts = None
    skipped_nonfinite = 0
    chips_with_nonfinite = 0
    used_images = 0
    for tif_path in tif_paths:
        ds = gdal.Open(str(tif_path))
        if ds is None:
            raise RuntimeError(f"Could not open training chip: {tif_path}")
        arr = ds.ReadAsArray()
        if arr is None:
            raise RuntimeError(f"Could not read training chip: {tif_path}")
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        arr = arr.astype(np.float64, copy=False)
        bands = int(arr.shape[0])
        if sums is None:
            sums = np.zeros(bands, dtype=np.float64)
            sumsqs = np.zeros(bands, dtype=np.float64)
            counts = np.zeros(bands, dtype=np.float64)

        flat = arr.reshape(bands, -1)
        if not np.isfinite(flat).all():
            chips_with_nonfinite += 1
        if max_pixels_per_image > 0 and flat.shape[1] > max_pixels_per_image:
            idx = rng.choice(flat.shape[1], size=max_pixels_per_image, replace=False)
            flat = flat[:, idx]
        image_used = False
        for band_idx in range(bands):
            data = flat[band_idx]
            nodata = ds.GetRasterBand(band_idx + 1).GetNoDataValue()
            if nodata is not None:
                data = data[data != nodata]
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            sums[band_idx] += float(np.sum(data))
            sumsqs[band_idx] += float(np.sum(np.square(data)))
            counts[band_idx] += float(data.size)
            image_used = True
        if image_used:
            used_images += 1
        else:
            skipped_nonfinite += 1

    if sums is None or counts is None or np.any(counts <= 0):
        raise RuntimeError("Could not compute per-band stats from training imagery.")
    means = sums / counts
    variances = np.maximum((sumsqs / counts) - np.square(means), 1e-12)
    stds = np.sqrt(variances)
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise RuntimeError(
            f"Non-finite band stats computed from {image_dir}; used_images={used_images} skipped_images={skipped_nonfinite}"
        )
    if skipped_nonfinite:
        log(f"Skipped {skipped_nonfinite} training chips with no finite pixels while computing band stats")
    if chips_with_nonfinite:
        log(f"Found {chips_with_nonfinite} training chips containing non-finite pixel values; runtime loader sanitization will replace them with 0.0")
    log(f"Computed band stats: mean={means.tolist()} std={stds.tolist()}")
    return means.astype(float).tolist(), stds.astype(float).tolist()


def _update_normalize_steps(pipeline: list[dict], mean: list[float], std: list[float]) -> None:
    for step in pipeline:
        if step.get("type") == "Normalize":
            step["mean"] = list(mean)
            step["std"] = list(std)
            step["to_rgb"] = False
        transforms = step.get("transforms")
        if isinstance(transforms, list):
            _update_normalize_steps(transforms, mean=mean, std=std)


def _configure_one_category_dataset(cfg, dataset_root: Path) -> None:
    ann_dir = dataset_root / "annotations"
    train_json = ann_dir / "one_cat_train.json"
    val_json = ann_dir / "one_cat_val.json"
    train_img_prefix = str((dataset_root / "train" / "images").resolve()) + "/"
    val_img_prefix = str((dataset_root / "val" / "images").resolve()) + "/"

    for split_name, ann_path, img_prefix in (
        ("train", train_json, train_img_prefix),
        ("val", val_json, val_img_prefix),
        ("test", val_json, val_img_prefix),
    ):
        split_cfg = cfg.data[split_name]
        split_cfg["type"] = "IrLandOneCatDataset"
        split_cfg["ann_file"] = str(ann_path.resolve())
        split_cfg["img_prefix"] = img_prefix

    if cfg.get("custom_hooks", None):
        cfg.custom_hooks = []


def _configure_one_category_model(cfg) -> None:
    roi_head = cfg.model.roi_head
    bbox_heads = roi_head.bbox_head if isinstance(roi_head.bbox_head, list) else [roi_head.bbox_head]
    for head in bbox_heads:
        head["num_classes"] = 1
    roi_head.mask_head["num_classes"] = 1
    roi_head.point_head["num_classes"] = 1


def _disable_unavailable_logging_hooks(cfg, log) -> None:
    hooks = cfg.get("log_config", {}).get("hooks")
    if not isinstance(hooks, list):
        return

    filtered = []
    removed = []
    for hook in hooks:
        hook_type = str(hook.get("type", ""))
        if hook_type == "TensorboardLoggerHook":
            removed.append(hook_type)
            continue
        filtered.append(hook)

    if removed:
        cfg.log_config["hooks"] = filtered
        log(f"Removed unavailable logging hooks: {', '.join(removed)}")


def _resolve_checkpoint_path(config_path: Path, requested_checkpoint: str, log) -> str | None:
    if requested_checkpoint:
        checkpoint_path = Path(requested_checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    sibling_checkpoint = config_path.with_suffix(".pth")
    if sibling_checkpoint.exists():
        log(f"Using default paper checkpoint: {sibling_checkpoint}")
        return str(sibling_checkpoint.resolve())
    return None


def _upgrade_legacy_nms_fields(node, log, path: str = "cfg") -> None:
    if isinstance(node, dict):
        if "nms_thr" in node and "nms" not in node:
            node["nms"] = {"type": "nms", "iou_threshold": float(node["nms_thr"])}
            log(f"Upgraded legacy nms_thr to nms config at {path}")
        if "max_num" in node and "max_per_img" not in node:
            node["max_per_img"] = int(node["max_num"])
            log(f"Upgraded legacy max_num to max_per_img at {path}")
        for key, value in list(node.items()):
            _upgrade_legacy_nms_fields(value, log, path=f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            _upgrade_legacy_nms_fields(value, log, path=f"{path}[{idx}]")


def _configure_best_checkpoint_saving(cfg, *, validate: bool, log) -> None:
    if not validate:
        return
    evaluation_cfg = cfg.get("evaluation", {})
    if not isinstance(evaluation_cfg, dict):
        evaluation_cfg = {}
    evaluation_cfg.setdefault("save_best", "segm_mAP")
    evaluation_cfg.setdefault("rule", "greater")
    cfg.evaluation = evaluation_cfg

    checkpoint_cfg = cfg.get("checkpoint_config", {})
    if not isinstance(checkpoint_cfg, dict):
        checkpoint_cfg = {}
    checkpoint_cfg.setdefault("save_last", True)
    cfg.checkpoint_config = checkpoint_cfg
    log(
        "Enabled best-checkpoint saving: "
        f"save_best={cfg.evaluation.get('save_best')} rule={cfg.evaluation.get('rule')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, help="Root produced by build_paper_dataset.py")
    ap.add_argument("--config", default="model/cascade_mask_rcnn_pointrend_cbam.py", help="Base MMDetection config")
    ap.add_argument("--work-dir", required=True, help="Training work directory")
    ap.add_argument("--checkpoint", default="", help="Optional checkpoint to load before training")
    ap.add_argument("--samples-per-gpu", type=int, default=2, help="Batch size per GPU")
    ap.add_argument("--workers-per-gpu", type=int, default=0, help="Data loader workers per GPU")
    ap.add_argument("--total-epochs", type=int, default=0, help="Override total epochs; 0 keeps config default")
    ap.add_argument("--optimizer-lr", type=float, default=0.0, help="Override optimizer learning rate; 0 keeps config default")
    ap.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=0.0,
        help="Enable gradient clipping with this max norm; 0 keeps config default",
    )
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--validate", action="store_true", help="Run validation during training")
    ap.add_argument("--compute-stats", action="store_true", help="Compute train-image mean/std and override Normalize steps")
    ap.add_argument("--stats-max-images", type=int, default=0, help="Limit images used for band stats; 0 uses all")
    ap.add_argument("--stats-max-pixels-per-image", type=int, default=250000, help="Sample cap per image when computing band stats")
    ap.add_argument("--log-file", default="", help="Optional log file")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    work_dir = ensure_dir(args.work_dir)
    log = build_logger(args.log_file if args.log_file else (work_dir / "train_paper_model.log"))
    _seed_everything(int(args.seed))

    import torch

    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    from mm_scripts.apis.train import train_detector

    config_path = Path(args.config).resolve()
    cfg = Config.fromfile(str(config_path))
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])

    _configure_one_category_dataset(cfg, dataset_root=dataset_root)
    _configure_one_category_model(cfg)
    _disable_unavailable_logging_hooks(cfg, log)
    _upgrade_legacy_nms_fields(cfg.train_cfg, log, path="cfg.train_cfg")
    _upgrade_legacy_nms_fields(cfg.test_cfg, log, path="cfg.test_cfg")
    _configure_best_checkpoint_saving(cfg, validate=bool(args.validate), log=log)
    cfg.work_dir = str(work_dir.resolve())
    cfg.resume_from = None
    cfg.load_from = _resolve_checkpoint_path(config_path, str(args.checkpoint), log)
    requested_device = str(args.device).strip().lower()
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = requested_device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested, but torch.cuda.is_available() is false")
    cfg.gpu_ids = [0] if resolved_device == "cuda" else []
    cfg.device = resolved_device
    cfg.seed = int(args.seed)
    cfg.data.samples_per_gpu = int(args.samples_per_gpu)
    cfg.data.workers_per_gpu = int(args.workers_per_gpu)
    if int(args.total_epochs) > 0:
        cfg.total_epochs = int(args.total_epochs)
    if float(args.optimizer_lr) > 0.0:
        cfg.optimizer["lr"] = float(args.optimizer_lr)
        log(f"Overrode optimizer learning rate: {float(args.optimizer_lr)}")
    if float(args.grad_clip_max_norm) > 0.0:
        cfg.optimizer_config = {
            "grad_clip": {
                "max_norm": float(args.grad_clip_max_norm),
                "norm_type": 2,
            }
        }
        log(f"Enabled gradient clipping: max_norm={float(args.grad_clip_max_norm)}")

    stats_payload = None
    if bool(args.compute_stats):
        train_image_dir = dataset_root / "train" / "images"
        mean, std = _compute_band_stats(
            train_image_dir,
            max_images=int(args.stats_max_images),
            max_pixels_per_image=int(args.stats_max_pixels_per_image),
            log=log,
        )
        cfg.img_norm_cfg["mean"] = list(mean)
        cfg.img_norm_cfg["std"] = list(std)
        _update_normalize_steps(cfg.data.train.pipeline, mean=mean, std=std)
        _update_normalize_steps(cfg.data.val.pipeline, mean=mean, std=std)
        _update_normalize_steps(cfg.data.test.pipeline, mean=mean, std=std)
        stats_payload = {"mean": mean, "std": std}

    resolved_config_path = work_dir / "resolved_config.py"
    cfg.dump(str(resolved_config_path))
    log(f"Wrote resolved config: {resolved_config_path}")
    log(f"Resolved execution device: {resolved_device}")

    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.CLASSES = ("crops_completed_circle",)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta = {
        "seed": int(args.seed),
        "dataset_root": str(dataset_root.resolve()),
        "resolved_config": str(resolved_config_path.resolve()),
    }
    if stats_payload is not None:
        meta["band_stats"] = stats_payload
    save_json(work_dir / "train_meta.json", meta)
    log(
        f"Starting training: dataset_root={dataset_root} work_dir={work_dir} "
        f"validate={bool(args.validate)} device={resolved_device}"
    )
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=bool(args.validate),
        timestamp=timestamp,
        meta=meta,
        device=resolved_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
