"""Diagnostic: at the author's conf=0.5 threshold, decompose each method's mAP
into recall (GT person boxes matched at IoU>=0.5) vs precision / false positives.

Tests the author's claim that Gray/PAC-R low mAP is FALSE-POSITIVE driven
(person still detected -> recall stays high; extra boxes -> precision drops),
as opposed to occlusion box-disruption (recall actually collapses).
"""
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from inria_dataset import load_inria
from detector import YOLODetector
from eot_transforms import PatchApplier
from patch_generator import CAPGenGenerator
from eval_inria import place_patches_on_all_bboxes, box_iou, compute_ap50

CONF = 0.5  # author's protocol


def load_cp(path, device):
    gen = CAPGenGenerator(patch_size=300, num_base_colors=3, temperature=0.1, device=device)
    gen.load_color_prob_matrix(path)
    return gen.generate_patch().detach()


def load_raw(path, device):
    ck = torch.load(path, map_location=device)
    return torch.sigmoid(ck['patch_logits']).to(device).detach()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = load_inria('./INRIAPerson', split='test')
    print(f"Loaded {len(samples)} INRIA test images; conf threshold = {CONF}")
    detector = YOLODetector(model_name='yolov5s', device=device, conf_threshold=CONF)
    applier = PatchApplier(300)
    to_tensor = T.ToTensor()

    methods = [
        ('clean', None),
        ('Gray', torch.full((3, 300, 300), 0.5, device=device)),
        ('CAPGen-R1', load_cp('output_new/capgen_p/capgen_r1_color_prob.pt', device)),
        ('CAPGen-T1', load_cp('output_new/capgen_t1/best_color_prob.pt', device)),
        ('CAPGen-P2', load_raw('output_v1/capgen_p2_linear.pt', device)),
        ('AdvPatch', load_raw('output_new/advpatch/best_advpatch.pt', device)),
    ]

    def evaluate(patch):
        records = []
        total_gt = total_pred = total_tp = 0
        for img, bboxes in samples:
            img_r = img.resize((640, 640))
            placements, rbb = place_patches_on_all_bboxes(img, bboxes, 300, (640, 640))
            total_gt += len(rbb)
            if patch is None:
                pil = img_r
            else:
                it = to_tensor(img_r).to(device)
                with torch.no_grad():
                    for (x, y, s) in placements:
                        it = applier.apply_patch(it, patch, x, y, s)
                pil = Image.fromarray(
                    (it.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
            dets = [d for d in detector.detect(pil) if d['class'] == 0]
            preds = [(d['confidence'], tuple(d['bbox'])) for d in dets]
            total_pred += len(preds)
            matched = [False] * len(rbb)
            for _, box in sorted(preds, key=lambda x: -x[0]):
                best_iou, best_g = 0.0, -1
                for g, gt in enumerate(rbb):
                    if matched[g]:
                        continue
                    iou = box_iou(box, gt)
                    if iou > best_iou:
                        best_iou, best_g = iou, g
                if best_iou >= 0.5 and best_g >= 0:
                    matched[best_g] = True
                    total_tp += 1
            records.append({'gt': rbb, 'dets': preds})
        ap = compute_ap50(records, total_gt, 0.5) * 100.0
        recall = total_tp / max(1, total_gt)
        precision = total_tp / max(1, total_pred)
        return ap, recall, precision, total_pred / len(samples), total_gt / len(samples)

    print(f"\n{'Method':11s}{'AP50@.5':>9s}{'recall':>8s}{'prec':>8s}{'pred/img':>9s}{'gt/img':>8s}")
    print("-" * 53)
    for name, patch in methods:
        ap, rec, prec, ppi, gpi = evaluate(patch)
        print(f"{name:11s}{ap:9.2f}{rec:8.3f}{prec:8.3f}{ppi:9.2f}{gpi:8.2f}")
    print("-" * 53)
    print("recall = GT person boxes matched at IoU>=0.5 (real 'is the person found?')")
    print("low recall => occlusion/box-disruption; low prec + high recall => false positives")


if __name__ == '__main__':
    main()
