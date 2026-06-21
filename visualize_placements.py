"""Visualize CAPGen patch placements on INRIA test images.

The visualization matches the eval-only ablation protocol:
resize image to 640x640, keep patch area at patch_frac of each person bbox,
and place one patch on every person.
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

from inria_dataset import load_inria
from run_placement_ablation import PLACEMENT_Y


COLORS = {
    "gt": (40, 220, 90),
    "patch": (245, 65, 65),
    "text_bg": (0, 0, 0),
    "text": (255, 255, 255),
}


def get_font(size=16):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def boxes_for(img, bboxes, placement, patch_size=300, target_size=(640, 640), frac=0.25):
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
        px2 = int(round(px1 + side))
        py2 = int(round(py1 + side))
        rows.append({
            "bbox_resized_xyxy": [round(v, 2) for v in (bx1, by1, bx2, by2)],
            "patch_xyxy": [px1, py1, px2, py2],
            "patch_side": round(side, 2),
            "patch_scale": round(side / patch_size, 4),
        })
    return rows


def draw_label(draw, xy, text, font):
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=COLORS["text_bg"])
    draw.text((x, y), text, fill=COLORS["text"], font=font)


def draw_variant(img, bboxes, placement, frac):
    canvas = img.resize((640, 640)).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(16)

    rows = boxes_for(img, bboxes, placement, frac=frac)
    for i, row in enumerate(rows):
        bx1, by1, bx2, by2 = row["bbox_resized_xyxy"]
        px1, py1, px2, py2 = row["patch_xyxy"]
        draw.rectangle((px1, py1, px2, py2), fill=(128, 128, 128, 150),
                       outline=COLORS["patch"] + (255,), width=4)
        draw.rectangle((bx1, by1, bx2, by2), outline=COLORS["gt"] + (255,), width=4)
        draw_label(draw, (max(0, int(bx1)), max(0, int(by1) - 22)), f"GT {i}", font)
        draw_label(draw, (max(0, px1), max(0, py1)), f"P {i}", font)

    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    draw2 = ImageDraw.Draw(canvas)
    title_font = get_font(20)
    draw_label(draw2, (8, 8), placement, title_font)
    return canvas, rows


def make_contact_sheet(images, labels, thumb_w=256, thumb_h=256):
    font = get_font(15)
    cols = len(images)
    sheet = Image.new("RGB", (cols * thumb_w, thumb_h + 28), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        thumb = img.resize((thumb_w, thumb_h))
        x = i * thumb_w
        sheet.paste(thumb, (x, 28))
        draw.text((x + 6, 5), label, fill=(255, 255, 255), font=font)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./INRIAPerson")
    ap.add_argument("--out_dir", default="output_placement_ablation/visualizations")
    ap.add_argument("--max_images", type=int, default=24)
    ap.add_argument("--patch_frac", type=float, default=0.25)
    ap.add_argument("--placements", nargs="+", default=list(PLACEMENT_Y))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_inria(args.dataset_dir, split="test", max_images=args.max_images)

    all_rows = []
    labels = ["clean"] + args.placements
    for idx, (img, bboxes) in enumerate(samples):
        clean = img.resize((640, 640)).convert("RGB")
        draw = ImageDraw.Draw(clean)
        font = get_font(20)
        for j, row in enumerate(boxes_for(img, bboxes, "center", frac=args.patch_frac)):
            bx1, by1, bx2, by2 = row["bbox_resized_xyxy"]
            draw.rectangle((bx1, by1, bx2, by2), outline=COLORS["gt"], width=4)
            draw_label(draw, (max(0, int(bx1)), max(0, int(by1) - 22)), f"GT {j}", font)
        draw_label(draw, (8, 8), "clean GT", font)

        variants = [clean]
        sample_row = {
            "index": idx,
            "original_size": list(img.size),
            "num_bboxes": len(bboxes),
            "bboxes_original_xyxy": [list(map(int, b)) for b in bboxes],
            "placements": {},
        }
        for placement in args.placements:
            vis, rows = draw_variant(img, bboxes, placement, args.patch_frac)
            variants.append(vis)
            sample_row["placements"][placement] = rows

        sheet = make_contact_sheet(variants, labels)
        out_path = os.path.join(args.out_dir, f"sample_{idx:03d}.png")
        sheet.save(out_path)
        sample_row["visualization"] = out_path
        all_rows.append(sample_row)

    with open(os.path.join(args.out_dir, "placement_boxes.json"), "w", encoding="utf-8") as f:
        json.dump({
            "patch_frac": args.patch_frac,
            "placements": {k: PLACEMENT_Y[k] for k in args.placements},
            "samples": all_rows,
        }, f, indent=2)

    print(f"saved {len(samples)} sample sheets to {args.out_dir}")
    print(f"saved box metadata to {os.path.join(args.out_dir, 'placement_boxes.json')}")


if __name__ == "__main__":
    main()
