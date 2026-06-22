"""Evaluate old squash-trained AdvPatch/P-series runs with YOLOv5 val.py.

Outputs are written under cal_map_using_yolo/output_yolo_pseries_run{n}_official.
This reuses the official YOLO evaluation helpers from
run_letterbox_yolomap_experiment.py, while rendering images with the old squash
protocol used by the 20260621 AdvPatch retrains.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot_transforms import PatchApplier  # noqa: E402
from inria_dataset import load_inria  # noqa: E402
from run_letterbox_yolomap_experiment import (  # noqa: E402
    load_raw_patch,
    run_yolo_val,
    save_results,
    write_label,
    write_yolo_cache,
    write_yolo_yaml,
)


def parse_runs(value: str) -> list[int]:
    if value == "other9":
        return [1, 2, 3, 4, 5, 6, 8, 9, 10]
    return [int(x) for x in value.replace(",", " ").split()]


def source_dir_for_run(repo_dir: Path, run_id: int) -> Path:
    if run_id == 7:
        return repo_dir / "output_advpatch_retrain_20260621-7-best"
    return repo_dir / f"output_advpatch_retrain_20260621-{run_id}"


def squash_positions_and_boxes(img, bboxes, patch_size: int):
    width, height = 640, 640
    iw, ih = img.size
    sx, sy = width / iw, height / ih
    positions = []
    label_boxes = []
    for x1o, y1o, x2o, y2o in bboxes:
        x1, y1 = x1o * sx, y1o * sy
        x2, y2 = x2o * sx, y2o * sy
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1.0 or bh <= 1.0:
            continue
        side = max(1.0, (0.25 * bw * bh) ** 0.5)
        scale = float(max(1e-3, side / patch_size))
        side_i = max(1, int(round(side)))
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        x_off = int(round(cx - side * 0.5))
        y_off = int(round(cy - side * 0.5))
        x_off = max(0, min(width - side_i, x_off))
        y_off = max(0, min(height - side_i, y_off))
        positions.append((x_off, y_off, scale))
        label_boxes.append((x1, y1, x2, y2))
    return positions, label_boxes


def render_squash_dataset(args, out_dir: Path, method: str, patch, samples, device):
    dataset_dir = out_dir / method
    done = dataset_dir / ".complete"
    if done.exists() and not args.force_render:
        print(f"[skip] rendered dataset exists: {dataset_dir}")
        yaml_path = write_yolo_yaml(dataset_dir)
        write_yolo_cache(dataset_dir)
        return yaml_path

    if dataset_dir.exists() and args.force_render:
        shutil.rmtree(dataset_dir)

    img_dir = dataset_dir / "images" / "test"
    lab_dir = dataset_dir / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    applier = PatchApplier(args.patch_size)
    for i, (img, bboxes) in enumerate(tqdm(samples, desc=f"render {out_dir.name}/{method}")):
        arr = np.asarray(img.resize((640, 640)).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
        positions, label_boxes = squash_positions_and_boxes(img, bboxes, args.patch_size)
        with torch.no_grad():
            for x, y, scale in positions:
                tensor = applier.apply_patch(tensor, patch, x, y, scale)
        out = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(out).save(img_dir / f"{i:06d}.jpg", quality=95)
        write_label(lab_dir / f"{i:06d}.txt", label_boxes)

    yaml_path = write_yolo_yaml(dataset_dir)
    write_yolo_cache(dataset_dir)
    done.write_text("ok\n", encoding="utf-8")
    return yaml_path


def load_group_results(out_dir: Path):
    path = out_dir / "official_yolo_results.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("results", {})
    except json.JSONDecodeError:
        return {}


def eval_one_run(args, run_id: int, samples, device):
    source_dir = source_dir_for_run(args.repo_dir, run_id)
    p_dir = source_dir / "capgen_p_linear"
    patch_paths = {
        "advpatch": source_dir / "best_advpatch.pt",
        "p0": p_dir / "capgen_p_orig_linear.pt",
        "p1": p_dir / "capgen_p1_linear.pt",
        "p2": p_dir / "capgen_p2_linear.pt",
    }
    missing = [str(path) for path in patch_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"run{run_id} missing patch files: {missing}")

    out_dir = args.out_root / f"output_yolo_pseries_run{run_id}_official"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = out_dir

    meta = {
        "run_id": run_id,
        "source_dir": str(source_dir),
        "resize_mode": "squash",
        "eval_backend": "yolov5/val.py official metrics",
        "conf_thres": args.conf_thres,
        "patch_frac": 0.25,
    }
    results = load_group_results(out_dir)

    for method, patch_path in patch_paths.items():
        if not args.force_eval and method in results and "mAP50" in results[method]:
            print(f"[skip] YOLO val exists: run{run_id}/{method}")
            continue
        patch = load_raw_patch(patch_path, device)
        yaml_path = render_squash_dataset(args, out_dir, method, patch, samples, device)
        results[method] = run_yolo_val(args, method, yaml_path)
        save_results(args, results, meta)

    save_results(args, results, meta)
    return results


def save_summary(out_root: Path, all_results: dict[str, dict]):
    rows = []
    for run_id, results in all_results.items():
        for method, item in results.items():
            rows.append({
                "run": run_id,
                "method": method,
                "mAP50_percent": item.get("mAP50_percent"),
                "mAP50_95_percent": item.get("mAP50_95_percent"),
                "precision": item.get("precision"),
                "recall": item.get("recall"),
                "images": item.get("images"),
                "instances": item.get("instances"),
            })

    json_path = out_root / "pseries_20260621_yolo_official_summary.json"
    csv_path = out_root / "pseries_20260621_yolo_official_summary.csv"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    import csv
    keys = ["run", "method", "mAP50_percent", "mAP50_95_percent", "precision", "recall", "images", "instances"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="other9")
    parser.add_argument("--out_root", type=Path, default=ROOT / "cal_map_using_yolo")
    parser.add_argument("--dataset_dir", default="./INRIAPerson")
    parser.add_argument("--yolov5_dir", type=Path, default=ROOT / "yolov5")
    parser.add_argument("--patch_size", type=int, default=300)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--conf_thres", type=float, default=0.001)
    parser.add_argument("--force_render", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    args = parser.parse_args()

    args.repo_dir = ROOT
    args.out_root = args.out_root.resolve()
    args.yolov5_dir = args.yolov5_dir.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_inria(args.dataset_dir, split="test")
    runs = parse_runs(args.runs)

    print(f"Device: {device}")
    print(f"Loaded {len(samples)} INRIA test images")
    print(f"Runs: {runs}")

    all_results = {}
    for run_id in runs:
        print(f"\n=== run{run_id} ===")
        all_results[str(run_id)] = eval_one_run(args, run_id, samples, device)
        save_summary(args.out_root, all_results)

    print("\nYOLOv5 official mAP50")
    print("run  advpatch      p0      p1      p2")
    for run_id in runs:
        results = all_results[str(run_id)]
        vals = [results.get(m, {}).get("mAP50_percent") for m in ("advpatch", "p0", "p1", "p2")]
        print(f"{run_id:>3}  " + "  ".join("   n/a " if v is None else f"{v:7.2f}" for v in vals))


if __name__ == "__main__":
    main()

