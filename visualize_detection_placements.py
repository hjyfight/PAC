"""Visualize patch placements together with detector outputs.

For each selected INRIA sample and method, this writes a contact sheet:
clean | center | chest | upper_torso | lower_torso.
Green boxes are GT person boxes, red boxes are patch extents, and blue boxes
are YOLO person detections.
"""
import argparse
import json
import os

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from detector import YOLODetector
from eot_transforms import PatchApplier
from inria_dataset import load_inria
from run_placement_ablation import PLACEMENT_Y
from run_table1 import load_color_prob_patch, load_raw_patch


GT_COLOR = (40, 220, 90)
PATCH_COLOR = (245, 65, 65)
DET_COLOR = (35, 145, 255)
TEXT_BG = (0, 0, 0)
TEXT = (255, 255, 255)


def get_font(size=15):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_label(draw, xy, text, font, fill=TEXT):
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=TEXT_BG)
    draw.text((x, y), text, fill=fill, font=font)


def placement_rows(img, bboxes, placement, patch_size=300, target_size=(640, 640), frac=0.25):
    W, H = target_size
    iw, ih = img.size
    sx, sy = W / iw, H / ih
    y_rel = PLACEMENT_Y[placement]
    rows = []
    for x1o, y1o, x2o, y2o in bboxes:
        bx1, by1 = x1o * sx, y1o * sy
        bx2, by2 = x2o * sx, y2o * sy
        bw = bx2 - bx1
        bh = by2 - by1
        side = max(1.0, (frac * bw * bh) ** 0.5)
        cx = (bx1 + bx2) * 0.5
        cy = by1 + bh * y_rel
        px1 = int(round(cx - side * 0.5))
        py1 = int(round(cy - side * 0.5))
        rows.append({
            "bbox": [bx1, by1, bx2, by2],
            "patch": [px1, py1, int(round(px1 + side)), int(round(py1 + side))],
            "apply": [px1, py1, float(max(1e-3, side / patch_size))],
            "side": side,
        })
    return rows


def draw_boxes(img, rows, detections, title):
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = get_font(14)
    title_font = get_font(18)

    for i, row in enumerate(rows):
        bx1, by1, bx2, by2 = row["bbox"]
        draw.rectangle((bx1, by1, bx2, by2), outline=GT_COLOR, width=4)
        draw_label(draw, (max(0, int(bx1)), max(0, int(by1) - 20)), f"GT {i}", font)
        if "patch" in row:
            px1, py1, px2, py2 = row["patch"]
            draw.rectangle((px1, py1, px2, py2), outline=PATCH_COLOR, width=4)
            draw_label(draw, (max(0, px1), max(0, py1)), f"P {i}", font, fill=(255, 210, 210))

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        draw.rectangle((x1, y1, x2, y2), outline=DET_COLOR, width=3)
        draw_label(draw, (max(0, int(x1)), max(0, int(y1) - 20)), f"D {i} {conf:.2f}", font,
                   fill=(210, 235, 255))

    draw_label(draw, (8, 8), title, title_font)
    return out


def apply_patch_to_image(img, patch, rows, applier, device):
    to_tensor = T.ToTensor()
    it = to_tensor(img.resize((640, 640))).to(device)
    if patch is not None:
        with torch.no_grad():
            for row in rows:
                x, y, s = row["apply"]
                it = applier.apply_patch(it, patch, x, y, s)
    return Image.fromarray(
        (it.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    )


def contact_sheet(images, labels, thumb_w=320, thumb_h=320):
    font = get_font(15)
    sheet = Image.new("RGB", (len(images) * thumb_w, thumb_h + 28), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        thumb = img.resize((thumb_w, thumb_h))
        x = i * thumb_w
        sheet.paste(thumb, (x, 28))
        draw.text((x + 6, 5), label, fill=(255, 255, 255), font=font)
    return sheet


def load_methods(args, device):
    def cp(name):
        return os.path.join(args.cp_dir, name)

    def pp(name):
        return os.path.join(args.p_dir, name)

    all_methods = {
        "Gray": torch.full((3, 300, 300), 0.5, device=device),
        "CAPGen-R1": load_color_prob_patch(cp("capgen_r1_color_prob.pt"), device),
        "CAPGen-R2": load_color_prob_patch(cp("capgen_r2_color_prob.pt"), device),
        "CAPGen-T1": load_color_prob_patch(args.t1, device),
        "CAPGen-T2": load_color_prob_patch(args.t2, device),
        "CAPGen-P0": load_raw_patch(pp("capgen_p_orig_linear.pt"), device),
        "CAPGen-P1": load_raw_patch(pp("capgen_p1_linear.pt"), device),
        "CAPGen-P2": load_raw_patch(pp("capgen_p2_linear.pt"), device),
        "AdvPatch": load_raw_patch(args.advpatch, device),
    }
    return {name: all_methods[name] for name in args.methods}


def det_to_json(det):
    return {
        "bbox": [float(x) for x in det["bbox"]],
        "confidence": float(det["confidence"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./INRIAPerson")
    ap.add_argument("--out_dir", default="output_placement_ablation_full/detection_visualizations")
    ap.add_argument("--max_images", type=int, default=12)
    ap.add_argument("--patch_frac", type=float, default=0.25)
    ap.add_argument("--placements", nargs="+", default=list(PLACEMENT_Y))
    ap.add_argument("--methods", nargs="+",
                    default=["Gray", "CAPGen-R1", "CAPGen-R2", "CAPGen-T1", "CAPGen-T2",
                             "CAPGen-P0", "CAPGen-P1", "CAPGen-P2", "AdvPatch"])
    ap.add_argument("--cp_dir", default="output_new/capgen_p")
    ap.add_argument("--p_dir", default="output_new/capgen_p_linear")
    ap.add_argument("--advpatch", default="output_new/advpatch/best_advpatch.pt")
    ap.add_argument("--t1", default="output_new/capgen_t1/best_color_prob.pt")
    ap.add_argument("--t2", default="output_new/capgen_t2/best_color_prob.pt")
    ap.add_argument("--detector", default="yolov5s")
    ap.add_argument("--vis_conf", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_inria(args.dataset_dir, split="test", max_images=args.max_images)
    detector = YOLODetector(model_name=args.detector, device=device, conf_threshold=args.vis_conf)
    applier = PatchApplier(300)
    methods = load_methods(args, device)

    metadata = []
    labels = ["clean"] + args.placements
    for idx, (img, bboxes) in enumerate(samples):
        clean_rows = placement_rows(img, bboxes, "center", frac=args.patch_frac)
        clean_rows = [{"bbox": row["bbox"]} for row in clean_rows]
        clean_img = img.resize((640, 640))
        clean_dets = [d for d in detector.detect(clean_img) if d["class"] == 0]
        clean_vis = draw_boxes(clean_img, clean_rows, clean_dets, "clean")

        for method_name, patch in methods.items():
            images = [clean_vis]
            method_meta = {
                "sample": idx,
                "method": method_name,
                "clean_detections": [det_to_json(d) for d in clean_dets],
                "placements": {},
            }
            for placement in args.placements:
                rows = placement_rows(img, bboxes, placement, frac=args.patch_frac)
                patched = apply_patch_to_image(img, patch, rows, applier, device)
                dets = [d for d in detector.detect(patched) if d["class"] == 0]
                vis = draw_boxes(patched, rows, dets, f"{method_name} {placement}")
                images.append(vis)
                method_meta["placements"][placement] = {
                    "patch_boxes": [row["patch"] for row in rows],
                    "detections": [det_to_json(d) for d in dets],
                }

            safe_name = method_name.replace("/", "_")
            out_path = os.path.join(args.out_dir, f"sample_{idx:03d}_{safe_name}.png")
            contact_sheet(images, labels).save(out_path)
            method_meta["visualization"] = out_path
            metadata.append(method_meta)

    meta_path = os.path.join(args.out_dir, "detections.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "detector": args.detector,
            "vis_conf": args.vis_conf,
            "patch_frac": args.patch_frac,
            "placements": {k: PLACEMENT_Y[k] for k in args.placements},
            "metadata": metadata,
        }, f, indent=2)

    print(f"saved {len(metadata)} detection sheets to {args.out_dir}")
    print(f"saved detection metadata to {meta_path}")


if __name__ == "__main__":
    main()
