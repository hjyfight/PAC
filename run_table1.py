"""Reproduce the paper's Table 1 (white-box, INRIA) for ONE detector (yolov5s).

Evaluates every method on the INRIA test set and prints a table in the paper's
style (mAP50 with drop from clean). We ALSO print DETECTION RATE because mAP50 is
confounded by false positives -- a non-attacking random/gray patch can score a
deceptively low mAP. Lower mAP50 / lower detRate = stronger attack.

Methods (skipped automatically if their checkpoint is missing):
  clean       - no patch (baseline)
  Gray        - solid gray patch
  CAPGen-R1/2 - random CONTINUOUS color-prob matrix (Eq.5) + Bc1/Bc2  (output_new/capgen_p/capgen_r{1,2}_color_prob.pt)
  CAPGen-T1/2 - gradient-trained matrix with fixed Bc1/Bc2            (--t1 / --t2)
  CAPGen-P0   - the AdvPatch pattern recolored on its own RGB basis (P0 == AdvPatch
                under --p_method linear)
  CAPGen-P1/2 - AdvPatch pattern recolored to Bc1/Bc2. Construction selected by
                --p_method. For the 2026-06-16 linear_fixed state, use
                --p_method linear and --p_dir output_new/capgen_p_linear.
  AdvPatch    - free-pixel baseline                                   (--advpatch)

Train T1/T2 first with:
  python main.py --mode train --dataset_dir ./INRIAPerson --base_colors bc1 \
      --num_iterations 200 --output_dir output_new/capgen_t1
  python main.py --mode train --dataset_dir ./INRIAPerson --base_colors bc2 \
      --num_iterations 200 --output_dir output_new/capgen_t2

Usage:
  python run_table1.py            # uses default paths; skips T1/T2 if not trained yet
"""
import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from inria_dataset import load_inria
from detector import YOLODetector
from eot_transforms import PatchApplier
from patch_generator import CAPGenGenerator
from eval_inria import place_patches_on_all_bboxes, compute_ap50

# Paper Table 1, yolov5s column (mAP50; lower = stronger)
PAPER = {'Gray': 85.3, 'CAPGen-R1': 86.3, 'CAPGen-R2': 85.4, 'CAPGen-T1': 71.5,
         'CAPGen-T2': 71.7, 'CAPGen-P1': 32.7, 'CAPGen-P2': 36.8, 'AdvPatch': 31.6}


def load_color_prob_patch(path, device):
    gen = CAPGenGenerator(patch_size=300, num_base_colors=3, temperature=0.1, device=device)
    gen.load_color_prob_matrix(path)
    return gen.generate_patch().detach()


def load_raw_patch(path, device):
    ck = torch.load(path, map_location=device)
    return torch.sigmoid(ck['patch_logits']).to(device).detach()


def eval_method(patch, samples, detector, applier, device, det_thr=0.5):
    """patch=None -> clean (no patch placed). Returns mAP50(0-100), detRate, conf."""
    to_tensor = T.ToTensor()
    max_conf, records, total_gts = [], [], 0
    for img, bboxes in samples:
        img_resized = img.resize((640, 640))
        placements, rbb = place_patches_on_all_bboxes(img, bboxes, 300, (640, 640))
        total_gts += len(rbb)
        it = to_tensor(img_resized).to(device)
        if patch is not None:
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
    return {'mAP50': compute_ap50(records, total_gts, 0.5) * 100.0,
            'det_rate': float((arr >= det_thr).mean()),
            'conf': float(arr.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--advpatch', default='output_new/advpatch/best_advpatch.pt')
    ap.add_argument('--t1', default='output_new/capgen_t1/best_color_prob.pt')
    ap.add_argument('--t2', default='output_new/capgen_t2/best_color_prob.pt')
    ap.add_argument('--cp_dir', default='output_new/capgen_p')
    ap.add_argument('--p_dir', default='output_new/capgen_p_linear',
                    help='dir with linear PAC-P raw patches: '
                         'capgen_p{_orig,1,2}_linear.pt')
    ap.add_argument('--p_method', choices=['linear', 'softmax'], default='linear',
                    help="'linear' loads raw linear-recolor patches from --p_dir "
                         "(6/16 linear_fixed state). 'softmax' loads color-prob "
                         "matrices from --cp_dir.")
    ap.add_argument('--dataset_dir', default='./INRIAPerson')
    ap.add_argument('--detector', default='yolov5s')
    ap.add_argument('--conf', type=float, default=0.01,
                    help='detector confidence floor for the mAP/detection eval. '
                         'Paper does NOT specify one. NOTE: raising it does NOT fix '
                         'the Gray/R mAP collapse (at 0.25 Gray mAP=15.5, see '
                         'table1_yolov5s_conf25.json) -- occlusion displaces/splits '
                         'high-conf person boxes (IoU<0.5). Rank by detRate.')
    ap.add_argument('--max_images', type=int, default=None)
    ap.add_argument('--out_json', default='output_new/table1_yolov5s.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = load_inria(args.dataset_dir, split='test', max_images=args.max_images)
    print(f"Loaded {len(samples)} INRIA test images")
    detector = YOLODetector(model_name=args.detector, device=device, conf_threshold=args.conf)
    applier = PatchApplier(300)

    cp = lambda n: os.path.join(args.cp_dir, n)
    pp = lambda n: os.path.join(args.p_dir, n)
    if args.p_method == 'linear':
        p_entries = [
            ('CAPGen-P0', 'raw', pp('capgen_p_orig_linear.pt')),
            ('CAPGen-P1', 'raw', pp('capgen_p1_linear.pt')),
            ('CAPGen-P2', 'raw', pp('capgen_p2_linear.pt')),
        ]
    else:
        p_entries = [
            ('CAPGen-P0', 'cp', cp('capgen_p_orig_color_prob.pt')),
            ('CAPGen-P1', 'cp', cp('capgen_p1_color_prob.pt')),
            ('CAPGen-P2', 'cp', cp('capgen_p2_color_prob.pt')),
        ]
    plan = [
        ('clean', 'clean', None),
        ('Gray', 'gray', None),
        ('CAPGen-R1', 'cp', cp('capgen_r1_color_prob.pt')),
        ('CAPGen-R2', 'cp', cp('capgen_r2_color_prob.pt')),
        ('CAPGen-T1', 'cp', args.t1),
        ('CAPGen-T2', 'cp', args.t2),
        *p_entries,
        ('AdvPatch', 'raw', args.advpatch),
    ]

    def build(kind, path):
        if kind == 'clean':
            return ('ok', None)
        if kind == 'gray':
            return ('ok', torch.full((3, 300, 300), 0.5, device=device))
        if not path or not os.path.exists(path):
            return ('missing', None)
        if kind == 'cp':
            return ('ok', load_color_prob_patch(path, device))
        return ('ok', load_raw_patch(path, device))

    results = {}
    for name, kind, path in plan:
        status, patch = build(kind, path)
        if status == 'missing':
            print(f"  skip {name} (missing: {path})")
            continue
        print(f"  eval {name} ...")
        results[name] = eval_method(patch, samples, detector, applier, device)

    clean_map = results.get('clean', {}).get('mAP50')
    print("\n" + "=" * 70)
    print(f"Table 1  ({args.detector}, INRIA test, {len(samples)} imgs)  "
          f"lower mAP50 / detRate = stronger")
    print("=" * 70)
    print("%-11s %9s %9s %8s %7s %10s"
          % ('Method', 'mAP50', 'drop', 'detRate', 'conf', 'paper_mAP'))
    print("-" * 70)
    for name, _, _ in plan:
        if name not in results:
            continue
        r = results[name]
        drop = '' if clean_map is None else "%.1f" % (clean_map - r['mAP50'])
        paper = PAPER.get(name, '-')
        print("%-11s %9.2f %9s %8.3f %7.3f %10s"
              % (name, r['mAP50'], drop, r['det_rate'], r['conf'], paper))
    print("-" * 70)
    print("NOTE: rank by detRate. mAP50 is inflated-LOW for non-attacks (R/Gray):")
    print("occlusion displaces/splits high-conf person boxes (IoU<0.5 FPs), so a")
    print("low mAP there does NOT mean a real attack; raising --conf doesn't fix it.")

    json.dump({'detector': args.detector, 'num_images': len(samples),
               'results': results, 'paper_yolov5s': PAPER},
              open(args.out_json, 'w'), indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == '__main__':
    main()
