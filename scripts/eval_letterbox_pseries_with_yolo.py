"""Evaluate an existing letterbox-trained AdvPatch/P-series run with YOLOv5 val.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inria_dataset import load_inria  # noqa: E402
from run_letterbox_yolomap_experiment import (  # noqa: E402
    load_raw_patch,
    render_dataset,
    run_yolo_val,
    save_results,
)


def load_existing_results(out_dir: Path) -> dict:
    path = out_dir / "official_yolo_pseries_results.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("results", {})
    except json.JSONDecodeError:
        return {}


def save_pseries_results(args, results: dict, meta: dict) -> None:
    original = args.out_dir
    tmp_json = original / "official_yolo_results.json"
    tmp_csv = original / "official_yolo_results.csv"
    p_json = original / "official_yolo_pseries_results.json"
    p_csv = original / "official_yolo_pseries_results.csv"

    save_results(args, results, meta)
    tmp_json.replace(p_json)
    tmp_csv.replace(p_csv)
    print(f"Saved {p_json}")
    print(f"Saved {p_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_dir", default="./INRIAPerson")
    parser.add_argument("--yolov5_dir", type=Path, default=ROOT / "yolov5")
    parser.add_argument("--patch_size", type=int, default=300)
    parser.add_argument("--conf_thres", type=float, default=0.001)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--max_test_images", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    args = parser.parse_args()

    args.repo_dir = ROOT
    args.out_dir = args.out_dir.resolve()
    args.yolov5_dir = args.yolov5_dir.resolve()

    p_dir = args.out_dir / "capgen_p_linear"
    patch_paths = {
        "advpatch": args.out_dir / "best_advpatch.pt",
        "p0": p_dir / "capgen_p_orig_linear.pt",
        "p1": p_dir / "capgen_p1_linear.pt",
        "p2": p_dir / "capgen_p2_linear.pt",
    }
    missing = [str(path) for path in patch_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing patch files:\n" + "\n".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_inria(args.dataset_dir, split="test", max_images=args.max_test_images)
    print(f"Output: {args.out_dir}")
    print(f"Device: {device}")
    print(f"Loaded {len(samples)} INRIA test images")
    print("Resize mode: letterbox")

    results = load_existing_results(args.out_dir)
    meta = {
        "command": " ".join(sys.argv),
        "out_dir": str(args.out_dir),
        "dataset_dir": args.dataset_dir,
        "resize_mode": "letterbox",
        "eval_backend": "yolov5/val.py official metrics",
        "conf_thres": args.conf_thres,
        "patch_frac": 0.25,
        "advpatch": str(patch_paths["advpatch"]),
        "p_dir": str(p_dir),
    }

    for method, patch_path in patch_paths.items():
        if not args.force_eval and method in results and "mAP50" in results[method]:
            print(f"[skip] YOLO val exists: {method}")
            continue
        patch = load_raw_patch(patch_path, device)
        yaml_path = render_dataset(args, method, patch, samples, device)
        results[method] = run_yolo_val(args, method, yaml_path)
        save_pseries_results(args, results, meta)

    save_pseries_results(args, results, meta)

    print("\nYOLOv5 official mAP50")
    print("%-9s %9s %9s %9s %9s" % ("method", "mAP50", "mAP50:95", "P", "R"))
    print("-" * 54)
    for method in patch_paths:
        r = results.get(method, {})
        print("%-9s %9s %9s %9s %9s" % (
            method,
            "" if r.get("mAP50_percent") is None else f"{r['mAP50_percent']:.2f}",
            "" if r.get("mAP50_95_percent") is None else f"{r['mAP50_95_percent']:.2f}",
            "" if r.get("precision") is None else f"{r['precision']:.3f}",
            "" if r.get("recall") is None else f"{r['recall']:.3f}",
        ))


if __name__ == "__main__":
    main()
