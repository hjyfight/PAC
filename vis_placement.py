"""Visualize patch PLACEMENT (modified strict-center code) on 10 random images.
For each image, draw two panels (squash | letterbox), both at frac=0.25:
  green box = person GT bbox
  red box   = patch footprint (square, 25% of bbox AREA, centered on bbox center,
              overflow clipped to image -- exactly what training/eval now use).
No detector involved; this is purely to inspect WHERE patches land.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

from inria_dataset import load_inria
from eval_inria import (place_patches_on_all_bboxes, place_patches_letterbox,
                        letterbox_pil)

FRAC = 0.25
PATCH_SIZE = 300
OUT = 'output_v1/vis_placement'
os.makedirs(OUT, exist_ok=True)


def draw_panel(frame_rgb, placements, bboxes_xyxy, title):
    base = frame_rgb.convert('RGBA')
    ov = Image.new('RGBA', base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for (bx1, by1, bx2, by2) in bboxes_xyxy:           # person bbox = green
        d.rectangle([bx1, by1, bx2, by2], outline=(0, 220, 0, 255), width=4)
    for (x, y, scale) in placements:                   # patch footprint = red
        side = scale * PATCH_SIZE
        d.rectangle([x, y, x + side, y + side], fill=(255, 0, 0, 70),
                    outline=(255, 40, 40, 255), width=3)
    out = Image.alpha_composite(base, ov).convert('RGB')
    d2 = ImageDraw.Draw(out)
    d2.rectangle([0, 0, out.size[0] - 1, 18], fill=(0, 0, 0))
    d2.text((4, 4), title, fill=(255, 255, 255))
    return out


def main():
    samples = load_inria('./INRIAPerson', split='test')
    idx = np.random.RandomState(42).choice(len(samples), size=10, replace=False)
    print(f"Visualizing {len(idx)} random images at frac={FRAC} (strict-center placement)")
    for n, i in enumerate(idx):
        img, bboxes = samples[int(i)]

        sq = img.resize((640, 640))
        pl_sq, rbb_sq = place_patches_on_all_bboxes(img, bboxes, PATCH_SIZE, (640, 640), frac=FRAC)
        panel_sq = draw_panel(sq, pl_sq, rbb_sq, f'squash 640x640  frac={FRAC}  ({len(rbb_sq)} ppl)')

        lb, r, left, top = letterbox_pil(img, 640)
        pl_lb, rbb_lb = place_patches_letterbox(bboxes, PATCH_SIZE, r, left, top, 640, frac=FRAC)
        panel_lb = draw_panel(lb, pl_lb, rbb_lb, f'letterbox 640x640  frac={FRAC}  ({len(rbb_lb)} ppl)')

        combo = Image.new('RGB', (640 * 2 + 10, 640), (255, 255, 255))
        combo.paste(panel_sq, (0, 0))
        combo.paste(panel_lb, (650, 0))
        p = os.path.join(OUT, f'vis_{n:02d}.png')
        combo.save(p)
        # report overflow: patch side vs bbox width/height
        notes = []
        for (x, y, s), (bx1, by1, bx2, by2) in zip(pl_lb, rbb_lb):
            side = s * PATCH_SIZE
            spill = (x < 0 or y < 0 or x + side > 640 or y + side > 640)
            wider = side > (bx2 - bx1)
            notes.append(('SPILL' if spill else '') + ('+W>bbox' if wider else ''))
        print(f"  vis_{n:02d}.png  ppl={len(rbb_lb)}  letterbox notes={notes}")
    print(f"\nSaved 10 images to {OUT}/")


if __name__ == '__main__':
    main()
