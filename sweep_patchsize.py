"""Patch-size sensitivity: does a smaller patch stop Gray from disrupting the GT
box (recall recovers -> mAP rises toward the paper's Gray~85), while a real
AdvPatch stays strong? Tests whether the Gray/R artifact is driven by our patch
being a large central occluder (side = sqrt(frac * bw * bh)).
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from inria_dataset import load_inria
from detector import YOLODetector
from eot_transforms import PatchApplier
from eval_inria import box_iou, compute_ap50

CONF = 0.5
FRACS = [0.05, 0.10, 0.15, 0.25]  # 0.25 = paper / current pipeline


def place(img_pil, bboxes, frac, patch_size=300, target=(640, 640)):
    W, H = target
    iw, ih = img_pil.size
    sx, sy = W / iw, H / ih
    placements, rbb = [], []
    for (x1, y1, x2, y2) in bboxes:
        bx1, by1, bx2, by2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        bw, bh = bx2 - bx1, by2 - by1
        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        placed = max(1.0, (frac * bw * bh) ** 0.5)
        scale = float(max(1e-3, placed / patch_size))
        xo = max(0, min(W - int(placed), int(round(bcx - placed * 0.5))))
        yo = max(0, min(H - int(placed), int(round(bcy - placed * 0.5))))
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

    def evaluate(patch, frac):
        records, total_gt, total_pred, total_tp = [], 0, 0, 0
        for img, bboxes in samples:
            img_r = img.resize((640, 640))
            placements, rbb = place(img, bboxes, frac)
            total_gt += len(rbb)
            it = to_tensor(img_r).to(device)
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
        return ap, total_tp / max(1, total_gt), total_pred / len(samples)

    print(f"\n{'frac':>6s}{'  | Gray AP50  recall  pred/img':30s}{'  | AdvPatch AP50  recall  pred/img':34s}")
    print("-" * 72)
    for f in FRACS:
        gap, grec, gppi = evaluate(gray, f)
        aap, arec, appi = evaluate(adv, f)
        print(f"{f:6.2f}  |   {gap:6.2f}  {grec:6.3f}    {gppi:5.2f}   |     {aap:6.2f}    {arec:6.3f}    {appi:5.2f}")
    print("-" * 72)
    print("paper Gray=85.3, AdvPatch=31.6. If Gray AP rises to ~85 at small frac")
    print("while Adv stays low, the Gray artifact is patch-size (occlusion) driven.")


if __name__ == '__main__':
    main()
