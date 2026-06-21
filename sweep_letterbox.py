"""Letterbox patch-size sweep: does some (letterbox, frac) reproduce the paper's
Gray~85 AND AdvPatch~31.6 simultaneously on the EXISTING (squash-trained) patches?
If yes -> no retrain needed (eval-protocol fix). If the attack patches are too weak
at the frac where Gray~85 -> retrain attacks at that size.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from inria_dataset import load_inria
from detector import YOLODetector
from eot_transforms import PatchApplier
from eval_inria import box_iou, compute_ap50, letterbox_pil
from patch_generator import CAPGenGenerator

CONF = 0.5
FRACS = [0.05, 0.10, 0.15, 0.25]


def place_lb(bboxes, frac, r, left, top, patch_size=300, S=640):
    placements, rbb = [], []
    for (x1, y1, x2, y2) in bboxes:
        bx1, by1, bx2, by2 = x1 * r + left, y1 * r + top, x2 * r + left, y2 * r + top
        bw, bh = bx2 - bx1, by2 - by1
        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        placed = max(1.0, (frac * bw * bh) ** 0.5)
        scale = float(max(1e-3, placed / patch_size))
        xo = max(0, min(S - int(placed), int(round(bcx - placed * 0.5))))
        yo = max(0, min(S - int(placed), int(round(bcy - placed * 0.5))))
        placements.append((xo, yo, scale))
        rbb.append((bx1, by1, bx2, by2))
    return placements, rbb


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = load_inria('./INRIAPerson', split='test')
    detector = YOLODetector(model_name='yolov5s', device=device, conf_threshold=CONF)
    applier = PatchApplier(300)
    to_tensor = T.ToTensor()

    gray = torch.full((3, 300, 300), 0.5, device=device)
    ck = torch.load('output_new/advpatch/best_advpatch.pt', map_location=device)
    adv = torch.sigmoid(ck['patch_logits']).to(device).detach()
    gen = CAPGenGenerator(patch_size=300, num_base_colors=3, temperature=0.1, device=device)
    gen.load_color_prob_matrix('output_new/capgen_t1/best_color_prob.pt')
    t1 = gen.generate_patch().detach()
    p2 = torch.sigmoid(torch.load('output_v1/capgen_p2_linear.pt', map_location=device)['patch_logits']).to(device).detach()

    methods = [('Gray', gray), ('AdvPatch', adv), ('CAPGen-T1', t1), ('CAPGen-P2', p2)]

    def evaluate(patch, frac):
        records, total_gt, total_pred, total_tp = [], 0, 0, 0
        for img, bboxes in samples:
            lb_img, r, left, top = letterbox_pil(img, 640)
            placements, rbb = place_lb(bboxes, frac, r, left, top)
            total_gt += len(rbb)
            it = to_tensor(lb_img).to(device)
            with torch.no_grad():
                for (x, y, s) in placements:
                    it = applier.apply_patch(it, patch, x, y, s)
            pil = Image.fromarray((it.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
            preds = [(d['confidence'], tuple(d['bbox'])) for d in detector.detect(pil) if d['class'] == 0]
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
        return ap, total_tp / max(1, total_gt)

    print("\nLETTERBOX sweep @conf0.5  (AP50 | recall).  paper: Gray85.3 T71.5 P2 36.8 Adv31.6")
    header = "frac  " + "".join(f"|{m[0]:>18s}" for m in methods)
    print(header)
    print("-" * len(header))
    for f in FRACS:
        cells = []
        for _, patch in methods:
            ap, rec = evaluate(patch, f)
            cells.append(f"|  {ap:6.2f} ({rec:.2f})   ")
        print(f"{f:5.2f} " + "".join(cells))


if __name__ == '__main__':
    main()
