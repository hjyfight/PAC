"""
Evaluate a trained CAPGen patch on the INRIA Person Test set.

Pipeline (matches paper §4.1):
  1. Resize each test image to 640x640 (no cropping).
  2. Place the patch on EVERY person bbox in the image - patch side =
     sqrt(0.25 * bw * bh), i.e. 25% of the bbox AREA (paper §4.5 defines patch
     size as "the percentage of the patch occupying the object area"), in resized
     640x640 coords. This matches the trainer's prepare_training_data.
  3. Run YOLO clean and patched.
  4. Compute mAP50 on the person class, using GT bboxes (resized to 640) as
     ground truth and the post-NMS detections as predictions. This is the
     paper's Table 1 metric (lower = better attack).

Also reports the legacy image-level diagnostics (max-per-image confidence,
detection rate, mean persons/image) so we can sanity-check.
"""
import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from detector import YOLODetector
from eot_transforms import PatchApplier
from inria_dataset import load_inria
from patch_generator import CAPGenGenerator


def place_patches_on_all_bboxes(img_pil, bboxes, patch_size,
                                target_size=(640, 640)):
    """Return (list[(x_off, y_off, scale)], list[resized_bbox_xyxy]).

    Mirrors the trainer's prepare_training_data: full-image resize to
    target_size, one patch per person bbox with side = sqrt(0.25 * bw * bh)
    (25% of the bbox AREA), in resized-image coords.
    """
    W, H = target_size
    iw, ih = img_pil.size
    sx, sy = W / iw, H / ih

    placements = []
    resized_bboxes = []
    for (x1o, y1o, x2o, y2o) in bboxes:
        bx1, by1 = x1o * sx, y1o * sy
        bx2, by2 = x2o * sx, y2o * sy
        bw = bx2 - bx1
        bh = by2 - by1
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5

        # Paper §4.2/§4.5: patch occupies 25% of the bbox AREA ->
        # side = sqrt(0.25 * bw * bh). Must match prepare_training_data().
        placed = max(1.0, (0.25 * bw * bh) ** 0.5)
        scale = float(max(1e-3, placed / patch_size))
        x_off = int(round(bcx - placed * 0.5))
        y_off = int(round(bcy - placed * 0.5))
        x_off = max(0, min(W - int(placed), x_off))
        y_off = max(0, min(H - int(placed), y_off))
        placements.append((x_off, y_off, scale))
        resized_bboxes.append((bx1, by1, bx2, by2))
    return placements, resized_bboxes


def box_iou(a, b):
    """IoU of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap50(records, total_gts, iou_thresh=0.5):
    """Compute AP@0.5 from per-image detection records.

    records: list of dicts {'gt': list[xyxy], 'dets': list[(conf, xyxy)]}
             (one entry per image)
    total_gts: total number of GT person boxes across all images.

    Returns AP50 in [0, 1] (VOC-style 11-point would be similar; here we use
    the all-points integral, which is what modern COCO/torchmetrics use).
    """
    # Flatten detections, tag with image id
    all_dets = []  # (conf, xyxy, img_idx, det_idx)
    for i, r in enumerate(records):
        for j, (conf, bbox) in enumerate(r['dets']):
            all_dets.append((conf, bbox, i, j))
    if not all_dets or total_gts == 0:
        return 0.0

    all_dets.sort(key=lambda x: -x[0])

    matched = [[False] * len(r['gt']) for r in records]
    tp = np.zeros(len(all_dets), dtype=np.float64)
    fp = np.zeros(len(all_dets), dtype=np.float64)

    for k, (conf, bbox, i, _) in enumerate(all_dets):
        gts = records[i]['gt']
        if not gts:
            fp[k] = 1.0
            continue
        # Best unmatched GT for this detection
        best_iou = 0.0
        best_g = -1
        for g, gt_bbox in enumerate(gts):
            if matched[i][g]:
                continue
            iou = box_iou(bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_g = g
        if best_iou >= iou_thresh and best_g >= 0:
            tp[k] = 1.0
            matched[i][best_g] = True
        else:
            fp[k] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(1, total_gts)
    precision = tp_cum / np.maximum(1e-12, tp_cum + fp_cum)

    # All-points AP (COCO style): make precision monotonically decreasing,
    # then integrate over recall.
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='./INRIAPerson')
    parser.add_argument('--color_prob_path', default=None,
                        help='Path to *_color_prob.pt produced by training '
                             '(CAPGen-T / CAPGen-P / CAPGen-R)')
    parser.add_argument('--raw_patch_pt', default=None,
                        help='Evaluate a raw free-pixel patch instead: an '
                             'advpatch *.pt containing patch_logits (the AdvPatch baseline)')
    parser.add_argument('--patch_size', type=int, default=300)
    parser.add_argument('--num_colors', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--detector', default='yolov5s')
    parser.add_argument('--conf_threshold', type=float, default=0.01,
                        help='Low detector conf for AP calculation '
                             '(paper Table 1 uses mAP-style integration). '
                             'Image-level det_rate is still reported at 0.5.')
    parser.add_argument('--det_rate_threshold', type=float, default=0.5,
                        help='Confidence threshold for the per-image '
                             'detection-rate diagnostic.')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--save_examples', type=int, default=0,
                        help='Save first N clean/patched image pairs to ./eval_examples/')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading INRIA Test images from {args.dataset_dir} ...")
    samples = load_inria(args.dataset_dir, split='test',
                         max_images=args.max_images)
    print(f"Loaded {len(samples)} INRIA Test images with person bboxes")

    print(f"Loading detector {args.detector} (conf={args.conf_threshold})...")
    detector = YOLODetector(model_name=args.detector, device=device,
                            conf_threshold=args.conf_threshold)

    if args.raw_patch_pt is not None:
        print(f"Loading raw free-pixel patch from {args.raw_patch_pt}")
        ck = torch.load(args.raw_patch_pt, map_location=device)
        patch = torch.sigmoid(ck['patch_logits']).to(device).detach()  # (3, P, P)
    else:
        if args.color_prob_path is None:
            raise SystemExit("Provide --color_prob_path (CAPGen-T/P/R) or --raw_patch_pt (AdvPatch)")
        print(f"Reconstructing patch from {args.color_prob_path}")
        gen = CAPGenGenerator(patch_size=args.patch_size,
                              num_base_colors=args.num_colors,
                              temperature=args.temperature, device=device)
        gen.load_color_prob_matrix(args.color_prob_path)
        patch = gen.generate_patch().detach()  # (3, P, P)
        print(f"  base_colors:\n{gen.base_colors.cpu().numpy()}")

    applier = PatchApplier(args.patch_size)
    to_tensor = T.ToTensor()
    target_class = 0  # person

    H, W = 640, 640

    clean_records = []
    patched_records = []
    total_gts = 0

    # Legacy image-level diagnostics
    clean_max_conf = []
    patched_max_conf = []
    clean_num_dets = []
    patched_num_dets = []
    placed_sizes = []

    save_dir = None
    if args.save_examples > 0:
        save_dir = './eval_examples'
        os.makedirs(save_dir, exist_ok=True)

    for i, (img, bboxes) in enumerate(tqdm(samples, desc="Eval")):
        img_resized = img.resize((W, H))

        placements, resized_bboxes = place_patches_on_all_bboxes(
            img, bboxes, args.patch_size, target_size=(W, H),
        )
        total_gts += len(resized_bboxes)
        for (_, _, scale) in placements:
            placed_sizes.append(scale * args.patch_size)

        img_tensor = to_tensor(img_resized).to(device)
        with torch.no_grad():
            image_with_patch = img_tensor
            for (x_off, y_off, scale) in placements:
                image_with_patch = applier.apply_patch(
                    image_with_patch, patch, x_off, y_off, scale,
                )

        patched_np = (image_with_patch.cpu().permute(1, 2, 0).numpy() * 255
                      ).clip(0, 255).astype(np.uint8)
        img_patched_pil = Image.fromarray(patched_np)

        clean_dets = detector.detect(img_resized)
        patched_dets = detector.detect(img_patched_pil)

        c_p_full = [d for d in clean_dets if d['class'] == target_class]
        p_p_full = [d for d in patched_dets if d['class'] == target_class]

        clean_records.append({
            'gt': resized_bboxes,
            'dets': [(d['confidence'], tuple(d['bbox'])) for d in c_p_full],
        })
        patched_records.append({
            'gt': resized_bboxes,
            'dets': [(d['confidence'], tuple(d['bbox'])) for d in p_p_full],
        })

        c_p = [d['confidence'] for d in c_p_full]
        p_p = [d['confidence'] for d in p_p_full]
        clean_max_conf.append(max(c_p) if c_p else 0.0)
        patched_max_conf.append(max(p_p) if p_p else 0.0)
        # Detection rate diagnostic uses a fixed 0.5 cutoff regardless of
        # the AP-friendly low conf_threshold loaded into the detector.
        clean_num_dets.append(sum(1 for v in c_p if v >= args.det_rate_threshold))
        patched_num_dets.append(sum(1 for v in p_p if v >= args.det_rate_threshold))

        if save_dir is not None and i < args.save_examples:
            img_resized.save(os.path.join(save_dir, f'{i:03d}_clean.png'))
            img_patched_pil.save(os.path.join(save_dir, f'{i:03d}_patched.png'))

    # ----- per-bbox mAP50 (paper Table 1 metric) -----
    clean_ap50 = compute_ap50(clean_records, total_gts, args.iou_threshold)
    patched_ap50 = compute_ap50(patched_records, total_gts, args.iou_threshold)

    clean_arr = np.array(clean_max_conf)
    patched_arr = np.array(patched_max_conf)
    clean_n = np.array(clean_num_dets)
    patched_n = np.array(patched_num_dets)
    placed_arr = np.array(placed_sizes) if placed_sizes else np.array([0.0])

    def det_rate(arr, t):
        return float((arr >= t).mean())

    clean_det_rate = det_rate(clean_arr, args.det_rate_threshold)
    patched_det_rate = det_rate(patched_arr, args.det_rate_threshold)
    asr_img = 1.0 - (patched_det_rate / max(1e-6, clean_det_rate))

    # Paper-style ASR derived from mAP50
    asr_map = 1.0 - (patched_ap50 / max(1e-6, clean_ap50))

    results = {
        'num_images': len(samples),
        'num_gt_bboxes': total_gts,
        'detector': args.detector,
        'conf_threshold_detector': args.conf_threshold,
        'det_rate_threshold': args.det_rate_threshold,
        'iou_threshold': args.iou_threshold,
        'color_prob_path': args.color_prob_path or args.raw_patch_pt,

        # Patch sizing sanity check
        'mean_placed_patch_px': float(placed_arr.mean()),
        'min_placed_patch_px': float(placed_arr.min()),
        'max_placed_patch_px': float(placed_arr.max()),

        # Paper Table 1 metric: per-bbox mAP50 on person class
        'clean_mAP50': clean_ap50,
        'patched_mAP50': patched_ap50,
        'mAP50_drop_abs': clean_ap50 - patched_ap50,
        'ASR_from_mAP50': asr_map,

        # Confidence diagnostics (max per image)
        'mean_max_person_conf_clean': float(clean_arr.mean()),
        'mean_max_person_conf_patched': float(patched_arr.mean()),
        'conf_drop_abs': float(clean_arr.mean() - patched_arr.mean()),
        'conf_drop_pct': float(
            100.0 * (clean_arr.mean() - patched_arr.mean())
            / max(1e-6, clean_arr.mean())
        ),

        # Image-level detection rate (legacy diagnostic)
        f'detection_rate_clean_at_{args.det_rate_threshold}': clean_det_rate,
        f'detection_rate_patched_at_{args.det_rate_threshold}': patched_det_rate,
        'ASR_from_image_det_rate': asr_img,

        # Number of person detections per image (mean) above det_rate_threshold
        'mean_persons_per_image_clean': float(clean_n.mean()),
        'mean_persons_per_image_patched': float(patched_n.mean()),
    }

    print("\n" + "=" * 60)
    print("INRIA Test set evaluation  [full-image + per-bbox mAP50]")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:50s}: {v:.4f}")
        else:
            print(f"  {k:50s}: {v}")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
