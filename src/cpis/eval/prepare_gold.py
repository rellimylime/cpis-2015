"""Prepare a gold evaluation dataset from manual polygon labels."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cpis.common.file_utils import ensure_dir, save_json  # noqa: E402
from cpis.common.logging_utils import build_logger  # noqa: E402


def _stage_raw(src: Path, dst: Path, *, mode: str) -> str:
    ensure_dir(dst.parent)
    if dst.exists():
        return "exists"

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return "symlinked"
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_fallback"
    raise RuntimeError(f"Unsupported copy mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagery-dir", required=True, help="Directory containing raw 4-band TIFFs")
    ap.add_argument("--labels", required=True, help="Manual polygon labels (.gpkg/.shp)")
    ap.add_argument("--out-root", required=True, help="Gold evaluation output root")
    ap.add_argument("--source-field", default="source_name", help="Field storing source tile names")
    ap.add_argument("--label-set-field", default="label_set", help="Optional label-set field")
    ap.add_argument("--label-set", default="val", help="Label-set value to keep; empty disables filtering")
    ap.add_argument("--status-field", default="status", help="Optional status field")
    ap.add_argument(
        "--include-status",
        nargs="*",
        default=["confirmed"],
        help="Status values to keep; empty disables status filtering",
    )
    ap.add_argument("--copy-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    ap.add_argument("--chip-size", type=int, default=1024)
    ap.add_argument("--chip-overlap", type=int, default=128)
    ap.add_argument("--min-ann-area-px", type=float, default=16.0)
    ap.add_argument("--keep-empty", action="store_true", help="Keep empty chips in the gold tiles")
    ap.add_argument("--log-file", default="", help="Optional log file")
    args = ap.parse_args()

    imagery_dir = Path(args.imagery_dir)
    if not imagery_dir.exists():
        raise FileNotFoundError(f"Imagery dir not found: {imagery_dir}")

    out_root = Path(args.out_root)
    log = build_logger(args.log_file if args.log_file else (out_root / "prepare_gold_eval.log"))
    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = gpd.read_file(labels_path)
    if labels.empty:
        raise RuntimeError(f"Manual labels are empty: {labels_path}")
    if labels.crs is None:
        raise RuntimeError(f"Manual labels missing CRS: {labels_path}")
    if args.source_field not in labels.columns:
        raise RuntimeError(f"Manual labels missing source field: {args.source_field}")

    selected = labels.copy()
    if args.label_set and args.label_set_field in selected.columns:
        selected = selected[selected[args.label_set_field].astype(str) == str(args.label_set)].copy()
    if args.include_status and args.status_field in selected.columns:
        allowed = {str(v) for v in args.include_status}
        selected = selected[selected[args.status_field].astype(str).isin(allowed)].copy()
    selected = selected[~selected.geometry.isna()].copy()
    selected = selected[~selected.geometry.is_empty].copy()
    if selected.empty:
        raise RuntimeError("No manual labels remain after filtering")

    selected_sources = sorted({str(v) for v in selected[args.source_field].dropna().unique() if str(v).strip()})
    if not selected_sources:
        raise RuntimeError(f"No usable source names found in field: {args.source_field}")

    source_tif_dir = ensure_dir(out_root / "source_tifs")
    missing_sources: list[str] = []
    staged_rows: list[dict[str, str]] = []
    for source_name in selected_sources:
        src_tif = imagery_dir / f"{source_name}.tif"
        if not src_tif.exists():
            missing_sources.append(source_name)
            continue
        dst_tif = source_tif_dir / src_tif.name
        action = _stage_raw(src_tif, dst_tif, mode=str(args.copy_mode))
        staged_rows.append(
            {
                "source_name": source_name,
                "src_tif": str(src_tif.resolve()),
                "staged_tif": str(dst_tif.resolve()),
                "stage_action": action,
            }
        )
    if missing_sources:
        raise RuntimeError(f"Missing source TIFFs for {len(missing_sources)} labels, e.g. {missing_sources[:5]}")

    filtered_labels = out_root / "gold_labels_filtered.gpkg"
    ensure_dir(filtered_labels.parent)
    selected.to_file(filtered_labels, driver="GPKG")
    log(
        f"Filtered manual labels: rows={len(selected)} sources={len(selected_sources)} "
        f"labels={filtered_labels}"
    )

    val_list_path = out_root / "gold_val_sources.txt"
    val_list_path.write_text("\n".join(selected_sources) + "\n", encoding="utf-8")

    dataset_root = out_root / "dataset"
    build_cmd = [
        sys.executable,
        str(ROOT / "tools" / "new_method" / "build_paper_dataset.py"),
        "--imagery-dir",
        str(source_tif_dir),
        "--labels",
        str(filtered_labels),
        "--out-root",
        str(dataset_root),
        "--chip-size",
        str(int(args.chip_size)),
        "--chip-overlap",
        str(int(args.chip_overlap)),
        "--val-fraction",
        "0",
        "--val-sources-file",
        str(val_list_path),
        "--min-ann-area-px",
        str(float(args.min_ann_area_px)),
    ]
    if bool(args.keep_empty):
        build_cmd.append("--keep-empty")

    log("Preparing gold evaluation COCO dataset")
    log("Build command: " + " ".join(build_cmd))
    subprocess.run(build_cmd, check=True)

    payload = {
        "imagery_dir": str(imagery_dir.resolve()),
        "labels": str(labels_path.resolve()),
        "filtered_labels": str(filtered_labels.resolve()),
        "out_root": str(out_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "source_tif_dir": str(source_tif_dir.resolve()),
        "gold_val_sources": str(val_list_path.resolve()),
        "selected_source_count": int(len(selected_sources)),
        "selected_sources": selected_sources,
        "filtered_label_count": int(len(selected)),
        "copy_mode": str(args.copy_mode),
        "staged_files": staged_rows,
    }
    save_json(out_root / "prepare_gold_eval_summary.json", payload)
    log(f"Wrote gold eval summary: {out_root / 'prepare_gold_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
