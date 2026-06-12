"""Faithful pattern>color test.

Take a TRAINED CAPGen-T color-probability matrix (the strong, optimized
pattern), keep it FIXED, and swap the base colors across several very different
palettes (Eq.4). Measure attack strength on the INRIA test set for each palette.

If the attack stays roughly constant across palettes, the PATTERN carries the
attack and COLOR is secondary -> the paper's "pattern > color" claim. This is the
correct experiment: unlike make_capgen_p (which lossily K-means-quantizes a
free-pixel AdvPatch and thereby destroys the attack), here the pattern is an
optimized color-probability matrix -- exactly what Eq.4 is meant to recolor.

PRIMARY metric = DETECTION RATE (fraction of images with a person still detected
at conf>=0.5; lower = stronger attack). mAP50 is also reported but is confounded
by false positives, so do NOT rank by it.

Usage:
    python test_pattern_vs_color.py \
        --color_prob_path output_new/capgen_t/best_color_prob.pt \
        --dataset_dir ./INRIAPerson \
        --out_json output_new/capgen_t/pattern_vs_color.json
"""
import argparse
import colorsys
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from inria_dataset import load_inria
from detector import YOLODetector
from eot_transforms import PatchApplier
from patch_generator import CAPGenGenerator
from eval_inria import place_patches_on_all_bboxes, compute_ap50

# Very different palettes (RGB 0-255). 'orig' is added at runtime from the
# checkpoint's own trained base colors.
PALETTES = {
    'Bc1 (paper)': np.array([[119, 49, 72], [2, 204, 1], [134, 2, 182]], dtype=np.float32),
    'Bc2 (paper)': np.array([[199, 21, 131], [40, 165, 4], [16, 69, 120]], dtype=np.float32),
    'forest':      np.array([[34, 139, 34], [139, 119, 101], [85, 107, 47]], dtype=np.float32),
    'snow':        np.array([[255, 250, 250], [220, 220, 220], [192, 192, 192]], dtype=np.float32),
    'desert':      np.array([[210, 180, 140], [188, 143, 107], [139, 119, 101]], dtype=np.float32),
}


def _lum(c):
    """Rec.601 luminance of an RGB color in [0,255]."""
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def lum_matched(orig_255, mode):
    """Build a palette that PRESERVES each original color's luminance (relative
    magnitude) but changes the hue.

    mode == 'gray'            -> (L, L, L): zero chroma, pure magnitude.
    mode == ('hue', shift, s) -> rotate each hue by `shift` (0..1), force
                                 saturation>=s, then rescale RGB to match the
                                 original color's luminance (clip to [0,255]).
    """
    out = []
    for c in orig_255:
        L = _lum(c)
        if mode == 'gray':
            out.append([L, L, L])
            continue
        _, shift, sat = mode
        r, g, b = [x / 255.0 for x in c]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h = (h + shift) % 1.0
        r2, g2, b2 = colorsys.hsv_to_rgb(h, max(s, sat), 1.0)
        c2 = np.array([r2, g2, b2]) * 255.0
        L2 = _lum(c2)
        if L2 > 1e-6:
            c2 = np.clip(c2 * (L / L2), 0, 255)
        out.append(c2)
    return np.array(out, dtype=np.float32)


def eval_patch(patch, samples, detector, applier, device, det_thr=0.5):
    """Place `patch` on every person bbox of every test image; return attack
    metrics (detection rate, mean max-person-confidence, mAP50)."""
    to_tensor = T.ToTensor()
    max_conf, records, total_gts = [], [], 0
    for img, bboxes in samples:
        img_resized = img.resize((640, 640))
        placements, rbb = place_patches_on_all_bboxes(
            img, bboxes, patch.shape[-1], (640, 640))
        total_gts += len(rbb)
        it = to_tensor(img_resized).to(device)
        with torch.no_grad():
            for (x, y, s) in placements:
                it = applier.apply_patch(it, patch, x, y, s)
        pil = Image.fromarray(
            (it.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
        dets = [d for d in detector.detect(pil) if d['class'] == 0]
        conf = [d['confidence'] for d in dets]
        max_conf.append(max(conf) if conf else 0.0)
        records.append({'gt': rbb,
                        'dets': [(d['confidence'], tuple(d['bbox'])) for d in dets]})
    arr = np.array(max_conf)
    return {
        'det_rate': float((arr >= det_thr).mean()),
        'mean_max_conf': float(arr.mean()),
        'mAP50': compute_ap50(records, total_gts, 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--color_prob_path', default='output_new/capgen_t/best_color_prob.pt')
    ap.add_argument('--dataset_dir', default='./INRIAPerson')
    ap.add_argument('--detector', default='yolov5s')
    ap.add_argument('--max_images', type=int, default=None)
    ap.add_argument('--out_json', default='output_new/capgen_t/pattern_vs_color.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = load_inria(args.dataset_dir, split='test', max_images=args.max_images)
    print(f"Loaded {len(samples)} INRIA test images")

    detector = YOLODetector(model_name=args.detector, device=device, conf_threshold=0.01)
    applier = PatchApplier(300)

    gen = CAPGenGenerator(patch_size=300, num_base_colors=3, temperature=0.1, device=device)
    gen.load_color_prob_matrix(args.color_prob_path)
    orig = gen.base_colors.detach().cpu().numpy() * 255.0

    palettes = {'orig (trained env colors)': orig}
    # luminance-PRESERVING swaps (change hue, keep relative magnitude)
    palettes['gray (lum-matched)'] = lum_matched(orig, 'gray')
    palettes['hueA (lum-matched)'] = lum_matched(orig, ('hue', 0.33, 0.6))
    palettes['hueB (lum-matched)'] = lum_matched(orig, ('hue', 0.66, 0.6))
    # arbitrary swaps (change hue AND magnitude)
    palettes.update(PALETTES)

    orig_lum = np.array([_lum(c) for c in orig])
    print("Evaluating the SAME trained pattern under different palettes...")
    results = {}
    for name, pal in tqdm(list(palettes.items()), desc="palette"):
        patch = gen.transfer_colors(pal).detach()
        r = eval_patch(patch, samples, detector, applier, device)
        plum = np.array([_lum(c) for c in pal])
        r['lum'] = [round(float(x), 1) for x in plum]
        r['lum_l1_from_orig'] = float(np.abs(np.sort(plum) - np.sort(orig_lum)).sum())
        results[name] = r

    drs = [r['det_rate'] for r in results.values()]
    summary = {
        'color_prob_path': args.color_prob_path,
        'metric_note': 'Rank by det_rate (lower=stronger). mAP50 is confounded by false positives.',
        'per_palette': results,
        'det_rate_min': float(min(drs)),
        'det_rate_max': float(max(drs)),
        'det_rate_spread': float(max(drs) - min(drs)),
    }

    print("\n%-28s %8s %7s %7s %9s" % ('palette', 'detRate', 'conf', 'mAP50', 'lumD'))
    print('-' * 64)
    for name, r in results.items():
        print("%-28s %8.4f %7.3f %7.3f %9.1f"
              % (name, r['det_rate'], r['mean_max_conf'], r['mAP50'], r['lum_l1_from_orig']))
    print('-' * 64)
    print("det_rate spread across palettes: %.4f" % summary['det_rate_spread'])
    print("lumD = luminance distance from orig. If lum-matched palettes (low lumD)")
    print("keep det_rate near 'orig' while high-lumD ones degrade => magnitude (not")
    print("hue) carries the attack -> reconciles with the paper's 'relative magnitude'.")

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    json.dump(summary, open(args.out_json, 'w'), indent=2)
    print(f"saved {args.out_json}")


if __name__ == '__main__':
    main()
