"""Run the paper-style instance-segmentation model on an arbitrary TIFF directory."""

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
from tools.detect_scripts import detect_sentinel_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagery-dir", required=True, help="Directory containing 4-band GeoTIFFs")
    ap.add_argument("--out-root", required=True, help="Output directory for masks and logs")
    ap.add_argument("--config", default="model/cascade_mask_rcnn_pointrend_cbam.py", help="MMDetection config file")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint to run")
    ap.add_argument("--ref-json", default="model/ann.json", help="Reference category JSON")
    ap.add_argument("--score-thr", nargs="+", type=float, default=[0.3], help="Output score thresholds")
    ap.add_argument("--infer-score-thr", type=float, default=None, help="Inference score threshold override")
    ap.add_argument("--infer-nms-iou", type=float, default=None, help="Inference NMS IoU override")
    ap.add_argument("--infer-max-per-img", type=int, default=None, help="Inference max-per-image override")
    ap.add_argument("--infer-mask-thr", type=float, default=None, help="Inference binary mask threshold override")
    ap.add_argument("--merge-nms-iou", type=float, default=0.1, help="Merge NMS IoU for vectorized segments")
    ap.add_argument("--nms-merge-cats", action="store_true", help="Merge categories during postprocess NMS")
    ap.add_argument("--workdir", default="", help="Temporary work directory")
    ap.add_argument("--log-file", default="", help="Optional log file")
    args = ap.parse_args()

    imagery_dir = Path(args.imagery_dir)
    if not imagery_dir.exists():
        raise FileNotFoundError(f"Imagery dir not found: {imagery_dir}")
    tif_paths = sorted(p for p in imagery_dir.glob("*.tif") if p.is_file())
    if not tif_paths:
        raise RuntimeError(f"No GeoTIFFs found in {imagery_dir}")

    out_root = ensure_dir(args.out_root)
    log = build_logger(args.log_file if args.log_file else (out_root / "run_paper_inference.log"))
    seg_res_path = ensure_dir(out_root / "segmentation_masks")
    workdir = Path(args.workdir) if args.workdir else (out_root / "temp")
    ensure_dir(workdir)

    list_path = out_root / "image_list.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for tif_path in tif_paths:
            f.write(tif_path.name + "\n")

    model_cfg = {
        "cfg_file": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "infer_score_thr": args.infer_score_thr,
        "infer_nms_iou": args.infer_nms_iou,
        "infer_max_per_img": args.infer_max_per_img,
        "infer_mask_thr_binary": args.infer_mask_thr,
    }

    log(
        f"Running paper inference on {len(tif_paths)} images: "
        f"imagery_dir={imagery_dir} out_root={out_root}"
    )
    with list_path.open("r", encoding="utf-8") as f:
        summary = detect_sentinel_batch(
            ori_img_dir=str(imagery_dir.resolve()),
            img_list_file=f,
            workdir=str(workdir.resolve()),
            model_cfg=model_cfg,
            ref_dataset_json=str(Path(args.ref_json).resolve()),
            nms_thr=float(args.merge_nms_iou),
            nms_merge_cats=bool(args.nms_merge_cats),
            score_thr=[float(v) for v in args.score_thr],
            seg_res_path=str(seg_res_path.resolve()),
        )

    payload = {
        "imagery_dir": str(imagery_dir.resolve()),
        "image_list": str(list_path.resolve()),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "segmentation_masks": str(seg_res_path.resolve()),
        "summary": summary,
    }
    save_json(out_root / "inference_summary.json", payload)
    log(f"Wrote inference summary: {out_root / 'inference_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
