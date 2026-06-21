"""Eval-only placement ablation for CAPGen patches on INRIA.

This does not retrain patches. It keeps patch_frac and "patch every person"
fixed, and only changes the patch center relative to each person bbox.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from detector import YOLODetector
from eot_transforms import PatchApplier
from eval_inria import compute_ap50, mAP50_backend
from inria_dataset import load_inria
from run_table1 import load_color_prob_patch, load_raw_patch


PLACEMENT_Y = {
    "center": 0.50,
    "chest": 0.34,
    "upper_torso": 0.40,
    "lower_torso": 0.60,
}


def place_patches(img_pil, bboxes, patch_size, placement, target_size=(640, 640), frac=0.25):
    """Return placements and resized GT bboxes for one resized 640x640 image."""
    W, H = target_size
    iw, ih = img_pil.size
    sx, sy = W / iw, H / ih
    y_rel = PLACEMENT_Y[placement]

    placements = []
    resized_bboxes = []
    for x1o, y1o, x2o, y2o in bboxes:
        bx1, by1 = x1o * sx, y1o * sy
        bx2, by2 = x2o * sx, y2o * sy
        bw = bx2 - bx1
        bh = by2 - by1
        cx = (bx1 + bx2) * 0.5
        cy = by1 + bh * y_rel
        side = max(1.0, (frac * bw * bh) ** 0.5)
        scale = float(max(1e-3, side / patch_size))
        placements.append((int(round(cx - side * 0.5)), int(round(cy - side * 0.5)), scale))
        resized_bboxes.append((bx1, by1, bx2, by2))
    return placements, resized_bboxes


def eval_patch(patch, samples, detector, applier, device, placement, patch_frac=0.25, det_thr=0.5):
    to_tensor = T.ToTensor()
    max_conf = []
    records = []
    total_gts = 0

    for img, bboxes in samples:
        img_resized = img.resize((640, 640))
        placements, rbb = place_patches(
            img, bboxes, 300, placement, target_size=(640, 640), frac=patch_frac
        )
        total_gts += len(rbb)

        it = to_tensor(img_resized).to(device)
        if patch is not None:
            with torch.no_grad():
                for x, y, s in placements:
                    it = applier.apply_patch(it, patch, x, y, s)

        pil = Image.fromarray(
            (it.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        dets = [d for d in detector.detect(pil) if d["class"] == 0]
        conf = [d["confidence"] for d in dets]
        max_conf.append(max(conf) if conf else 0.0)
        records.append({"gt": rbb, "dets": [(d["confidence"], tuple(d["bbox"])) for d in dets]})

    arr = np.asarray(max_conf)
    return {
        "mAP50": compute_ap50(records, total_gts, 0.5) * 100.0,
        "det_rate": float((arr >= det_thr).mean()),
        "conf": float(arr.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./INRIAPerson")
    ap.add_argument("--detector", default="yolov5s")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--patch_frac", type=float, default=0.25)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--cp_dir", default="output_new/capgen_p")
    ap.add_argument("--p_dir", default="output_new/capgen_p_linear")
    ap.add_argument("--advpatch", default="output_new/advpatch/best_advpatch.pt")
    ap.add_argument("--t1", default="output_new/capgen_t1/best_color_prob.pt")
    ap.add_argument("--t2", default="output_new/capgen_t2/best_color_prob.pt")
    ap.add_argument("--placements", nargs="+", default=list(PLACEMENT_Y))
    ap.add_argument("--out_dir", default="output_placement_ablation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_inria(args.dataset_dir, split="test", max_images=args.max_images)
    print(f"Loaded {len(samples)} INRIA test images")
    print(f"device={device}, detector={args.detector}, conf={args.conf}, patch_frac={args.patch_frac}")

    detector = YOLODetector(model_name=args.detector, device=device, conf_threshold=args.conf)
    applier = PatchApplier(300)

    def cp(name):
        return os.path.join(args.cp_dir, name)

    def pp(name):
        return os.path.join(args.p_dir, name)

    methods = [
        ("clean", None),
        ("Gray", torch.full((3, 300, 300), 0.5, device=device)),
        ("CAPGen-R1", load_color_prob_patch(cp("capgen_r1_color_prob.pt"), device)),
        ("CAPGen-R2", load_color_prob_patch(cp("capgen_r2_color_prob.pt"), device)),
        ("CAPGen-T1", load_color_prob_patch(args.t1, device)),
        ("CAPGen-T2", load_color_prob_patch(args.t2, device)),
        ("CAPGen-P0", load_raw_patch(pp("capgen_p_orig_linear.pt"), device)),
        ("CAPGen-P1", load_raw_patch(pp("capgen_p1_linear.pt"), device)),
        ("CAPGen-P2", load_raw_patch(pp("capgen_p2_linear.pt"), device)),
        ("AdvPatch", load_raw_patch(args.advpatch, device)),
    ]

    results = {"clean": eval_patch(None, samples, detector, applier, device, "center", args.patch_frac)}
    print(f"clean: mAP50={results['clean']['mAP50']:.2f}")

    for placement in args.placements:
        if placement not in PLACEMENT_Y:
            raise ValueError(f"unknown placement {placement}; choices={sorted(PLACEMENT_Y)}")
        print(f"\nplacement={placement} (bbox y={PLACEMENT_Y[placement]:.2f})")
        results[placement] = {}
        for name, patch in methods[1:]:
            print(f"  eval {name} ...")
            results[placement][name] = eval_patch(
                patch, samples, detector, applier, device, placement, args.patch_frac
            )
            r = results[placement][name]
            print(f"    {name}: mAP50={r['mAP50']:.2f}, detRate={r['det_rate']:.3f}, conf={r['conf']:.3f}")

    out = {
        "command": " ".join([sys.executable] + sys.argv),
        "detector": args.detector,
        "num_images": len(samples),
        "patch_frac": args.patch_frac,
        "placements": {k: PLACEMENT_Y[k] for k in args.placements},
        "mAP50_backend": mAP50_backend(),
        "results": results,
    }
    out_json = os.path.join(args.out_dir, "placement_ablation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
