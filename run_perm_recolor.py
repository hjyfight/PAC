"""Permutation-recolor experiment: does ANY color correspondence of Bc1/Bc2
preserve the attack of the trained CAPGen-T pattern?

Motivation: Eq.4 (t_ij = sum_k nc_k * r_k) does not specify how the new base
colors nc_k are ORDERED relative to the source colors c_k. The faithful table
(run_table1.py) uses the identity assignment and finds the recolor loses most
of the attack (P1 det 0.799 / P2 det 0.920 vs source P0 0.552). The luminance
analysis predicts why: the identity assignment inverts the luminance order AND
compresses its range (src luma span ~142 vs Bc1 ~58 / Bc2 ~50).

This script evaluates ALL 3! = 6 permutations of Bc1 and of Bc2 (12 recolors
plus the P0 anchor) under the EXACT run_table1.py protocol (same eval_method,
same PatchApplier, same conf floor), so a reviewer cannot object "you just
matched the colors wrong".

Result (288-img run, 2026-06-11, output_new/perm_recolor_yolov5s.json): all 12
permutations land at detRate 0.799-0.934 (vs P0 0.552, T1 0.611) -> the main
claim HOLDS: no correspondence preserves the attack. But the luminance-ORDER
hypothesis was falsified: the rank-matched perms (Bc1-021 0.854, Bc2-021 0.934)
are NOT the strongest -- the best is the identity Bc1-012 (0.799, max lumD!),
and within a palette the ordering follows neither luma order nor lumD (Bc1/Bc2
perm-means 0.872/0.875, spread 0.135 ~= 39/288 imgs). The first-order factor is
the compressed luminance RANGE (cross-palette); the per-slot assignment is a
second-order, luma-unexplained effect.

Anchors (output_new/table1_yolov5s_faithful.json): P0 0.552, P1==Bc1-012 0.799,
P2==Bc2-012 0.920, T1 (trained WITH Bc1) 0.611, AdvPatch 0.396.

Usage:
  python run_perm_recolor.py                  # full INRIA test (13 evals, ~1.3x a table1 run)
  python run_perm_recolor.py --max_images 30  # quick smoke test
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
from PIL import Image

from detector import YOLODetector
from eot_transforms import PatchApplier
from inria_dataset import load_inria
from make_capgen_p import BC1, BC2, save_color_prob
from patch_generator import CAPGenGenerator
from run_table1 import eval_method

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def luma(palette_255):
    return np.asarray(palette_255, dtype=np.float64) @ LUMA_W


def save_png(path, patch_chw):
    arr = (patch_chw.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_matrix', default='output_new/capgen_t/best_color_prob.pt',
                    help='trained CAPGen-T matrix whose pattern is recolored '
                         '(same source as the faithful P0/P1/P2)')
    ap.add_argument('--dataset_dir', default='./INRIAPerson')
    ap.add_argument('--detector', default='yolov5s')
    ap.add_argument('--conf', type=float, default=0.01,
                    help='detector conf floor; keep 0.01 == run_table1.py')
    ap.add_argument('--max_images', type=int, default=None)
    ap.add_argument('--out_dir', default='output_new/capgen_p_perm',
                    help='where the 12 recolored *_color_prob.pt / *.png artifacts go')
    ap.add_argument('--out_json', default='output_new/perm_recolor_yolov5s.json')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- source pattern (trained CAPGen-T) -----
    ck = torch.load(args.source_matrix, map_location='cpu')
    tau = float(ck.get('temperature', 0.1))
    logits_np = ck['logits'].numpy().astype(np.float32)               # (P, P, K)
    src_255 = (ck['base_colors'].numpy() * 255.0).astype(np.float64)  # (K, 3)
    P, _, K = logits_np.shape
    src_luma = luma(src_255)
    src_order = np.argsort(src_luma)

    gen = CAPGenGenerator(patch_size=P, num_base_colors=K, temperature=tau, device=device)
    gen.load_color_prob_matrix(args.source_matrix)

    print(f"Source: {args.source_matrix}  P={P} K={K} tau={tau}")
    print(f"  src colors (RGB):\n{src_255.round(1)}")
    print(f"  src luma: {src_luma.round(1)}  (span {src_luma.max() - src_luma.min():.0f})")

    # ----- build P0 + all 2x3! recolored candidates (same render path as eval) -----
    jobs = [('P0-src', None, None, gen.generate_patch().detach(), src_255)]
    for pal_name, BC in (('Bc1', BC1), ('Bc2', BC2)):
        for perm in itertools.permutations(range(K)):
            new_255 = np.asarray(BC, dtype=np.float64)[list(perm)]
            patch = gen.transfer_colors(new_255).detach()
            jobs.append((f"{pal_name}-" + ''.join(map(str, perm)), pal_name, perm,
                         patch, new_255))

    # ----- eval, identical protocol to run_table1.py -----
    samples = load_inria(args.dataset_dir, split='test', max_images=args.max_images)
    print(f"\nLoaded {len(samples)} INRIA test images")
    detector = YOLODetector(model_name=args.detector, device=device, conf_threshold=args.conf)
    applier = PatchApplier(P)

    rows = []
    for name, pal_name, perm, patch, new_255 in jobs:
        lum = luma(new_255)
        row = {
            'name': name,
            'palette': pal_name,
            'perm': None if perm is None else list(perm),
            'colors_rgb': np.asarray(new_255).round(1).tolist(),
            'luma': lum.round(1).tolist(),
            'luma_span': float(lum.max() - lum.min()),
            'luma_dist_sum': float(np.abs(lum - src_luma).sum()),
            'luma_order_match': bool((np.argsort(lum) == src_order).all()),
        }
        # persist artifacts so any row can be re-checked with eval_inria.py
        slug = name.lower().replace('-', '_')
        save_color_prob(os.path.join(args.out_dir, f'{slug}_color_prob.pt'),
                        logits_np, np.asarray(new_255, dtype=np.float32), tau)
        save_png(os.path.join(args.out_dir, f'{slug}.png'), patch)

        print(f"  eval {name:8s}  luma={'/'.join(f'{v:.0f}' for v in lum):11s} "
              f"ordOK={'Y' if row['luma_order_match'] else '-'} ...")
        row.update(eval_method(patch, samples, detector, applier, device))
        rows.append(row)

    # ----- report, strongest first -----
    rows_sorted = sorted(rows, key=lambda r: r['det_rate'])
    print("\n" + "=" * 78)
    print(f"Permutation recolor  ({args.detector}, INRIA test, {len(samples)} imgs)  "
          f"lower detRate = stronger")
    print("=" * 78)
    print("%-9s %12s %6s %6s %6s %9s %9s %7s"
          % ('Name', 'luma(new)', 'span', 'lumD', 'ordOK', 'mAP50', 'detRate', 'conf'))
    print("-" * 78)
    for r in rows_sorted:
        print("%-9s %12s %6.0f %6.0f %6s %9.2f %9.3f %7.3f"
              % (r['name'], '/'.join(f"{v:.0f}" for v in r['luma']),
                 r['luma_span'], r['luma_dist_sum'],
                 'Y' if r['luma_order_match'] else '-',
                 r['mAP50'], r['det_rate'], r['conf']))
    print("-" * 78)
    print("Anchors (faithful table): P0 0.552 | Bc1-012==P1 0.799 | Bc2-012==P2 0.920")
    print("                          T1 (trained WITH Bc1) 0.611 | AdvPatch 0.396")
    print("Claim to verify: even the BEST of the 12 permutations stays well above")
    print("P0/T1 -> no color correspondence of Bc1/Bc2 preserves the attack, because")
    print("the luminance span stays compressed (recolor != retrain).")

    json.dump({'detector': args.detector, 'num_images': len(samples),
               'conf_floor': args.conf,
               'source_matrix': args.source_matrix,
               'source_colors_rgb': src_255.round(1).tolist(),
               'source_luma': src_luma.round(1).tolist(),
               'rows': rows_sorted},
              open(args.out_json, 'w'), indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == '__main__':
    main()
