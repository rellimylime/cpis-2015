"""Run segmentation evaluation against a gold COCO-style holdout dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(ROOT / "src")
REPO_ROOT = str(ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from cpis.common.file_utils import ensure_dir, save_json  # noqa: E402
from cpis.common.logging_utils import build_logger  # noqa: E402
from tools.evaluation.eval_file import eval_file  # noqa: E402
from tools.sentinel_scripts.detect_dataset import detect_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, help="Gold dataset root prepared by prepare_gold_eval.py")
    ap.add_argument("--config", required=True, help="Model config or resolved_config.py used for evaluation")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    ap.add_argument("--out-root", required=True, help="Output directory for prediction JSON and eval CSV")
    ap.add_argument(
        "--img-scale",
        nargs=2,
        type=int,
        default=[1000, 600],
        metavar=("W", "H"),
        help="Test resize scale passed into the MMDetection test pipeline",
    )
    ap.add_argument("--max-det", type=int, default=10000, help="Maximum detections per image for eval_file")
    ap.add_argument("--log-file", default="", help="Optional log file")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    val_json = dataset_root / "annotations" / "one_cat_val.json"
    val_img_dir = dataset_root / "val" / "images"
    if not val_json.exists():
        raise FileNotFoundError(f"Gold val JSON not found: {val_json}")
    if not val_img_dir.exists():
        raise FileNotFoundError(f"Gold val image dir not found: {val_img_dir}")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_root = ensure_dir(args.out_root)
    log = build_logger(args.log_file if args.log_file else (out_root / "run_gold_eval.log"))
    res_json = out_root / "gold_eval.segm.json"
    file_prefix = out_root / "gold_eval.segm"

    dataset_cfg = {
        "cfg_file": str(config_path.resolve()),
        "img_dir": str(val_img_dir.resolve()) + "/",
        "json_path": str(val_json.resolve()),
        "img_scale": tuple(int(v) for v in args.img_scale),
    }
    model_cfg = {
        "cfg_file": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }

    log(
        f"Running gold evaluation: dataset_root={dataset_root} checkpoint={checkpoint_path} "
        f"img_scale={tuple(int(v) for v in args.img_scale)}"
    )
    detect_dataset(model=model_cfg, dataset=dataset_cfg, out_file=str(res_json.resolve()))

    iou_thrs = [0.50 + (0.05 * idx) for idx in range(10)]
    log(f"Evaluating segmentation results against {val_json}")
    eval_file(
        gt_file=str(val_json.resolve()),
        res_file=str(res_json.resolve()),
        metric="segm",
        max_det=int(args.max_det),
        iou_thrs=iou_thrs,
        file_prefix=str(file_prefix.resolve()),
    )

    save_json(
        out_root / "gold_eval_summary.json",
        {
            "dataset_root": str(dataset_root.resolve()),
            "config": str(config_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "val_json": str(val_json.resolve()),
            "val_images": str(val_img_dir.resolve()),
            "prediction_json": str(res_json.resolve()),
            "csv_prefix": str(file_prefix.resolve()),
            "img_scale": [int(v) for v in args.img_scale],
            "iou_thrs": iou_thrs,
            "max_det": int(args.max_det),
        },
    )
    log(f"Wrote gold evaluation summary: {out_root / 'gold_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
