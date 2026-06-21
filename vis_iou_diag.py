"""Visualize WHY Gray/R/T have low mAP despite 'detecting' people.

For each sample image x method, draw:
  - GT boxes (yellow)                 -> ground-truth persons in 640 coords
  - predicted boxes colored by IoU:
      GREEN  = TP (IoU >= 0.5)        -> counts toward recall
      RED    = FP (IoU < 0.5)        -> false positive, crashes precision
  - caption: "{method}: N preds | {tp} TP / {fp} FP | {gt} GT"

This shows the mechanism behind the protocol artifact: Gray/R produce MANY
fragmented / offset / duplicated boxes (high conf, visible) that don't reach
IoU>=0.5 -> tons of FP -> precision crashes -> mAP is low, even though the
person is visibly 'detected'. True attacks (P/AdvPatch) REMOVE boxes instead.

Matching is greedy by confidence (same as compute_ap50), so TP/FP here match
the mAP accounting exactly.
"""
import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from inria_dataset import load_inria
from detector import YOLODetector
from eot_transforms import PatchApplier
from patch_generator import CAPGenGenerator
from eval_inria import place_patches_on_all_bboxes, box_iou

# Samples picked from manifest person_det_counts where Gray/R >> clean
# (172: clean 4 -> Gray 8; 157: clean 2 -> Gray 4; 181: clean 2 -> Gray 4)
SAMPLES = [172, 157, 181]
CONF = 0.5
FRAC = 0.25
PSIZE = 300
W = H = 640
IOU_TP = 0.5

METHODS = [
    ('clean',    'clean', None),
    ('Gray',     'gray',  None),
    ('CAPGen-R1','cp', 'output_v2_squash/capgen_p/capgen_r1_color_prob.pt'),
    ('CAPGen-R2','cp', 'output_v2_squash/capgen_p/capgen_r2_color_prob.pt'),
    ('CAPGen-T1','cp', 'output_v2_squash/capgen_t1/best_color_prob.pt'),
    ('CAPGen-T2','cp', 'output_v2_squash/capgen_t2/best_color_prob.pt'),
]


def load_patch(kind, path, device):
    if kind == 'clean':
        return None
    if kind == 'gray':
        return torch.full((3, PSIZE, PSIZE), 0.5, device=device)
    gen = CAPGenGenerator(PSIZE, 3, 0.1, device)
    gen.load_color_prob_matrix(path)
    return gen.generate_patch().detach()


def match(preds, gts):
    """Greedy match by confidence (mirrors compute_ap50 TP/FP accounting).
    preds: list[(conf, xyxy)]; gts: list[xyxy]. Returns list[(conf, xy, iou, tp)]."""
    out = []
    used = [False] * len(gts)
    for conf, xy in sorted(preds, key=lambda p: -p[0]):
        best, bi = -1.0, -1
        for g, gt in enumerate(gts):
            if used[g]:
                continue
            iou = box_iou(xy, gt)
            if iou > best:
                best, bi = iou, g
        tp = best >= IOU_TP and bi >= 0
        if tp:
            used[bi] = True
        out.append((conf, xy, best, tp))
    return out


def _font(size):
    for p in ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_panel(img640, matched, gts, name):
    cap_h = 28
    panel = Image.new('RGB', (W, H + cap_h), (24, 24, 24))
    img = img640.convert('RGB').copy()
    d = ImageDraw.Draw(img)
    f_box = _font(11)
    # GT boxes (yellow)
    for gt in gts:
        d.rectangle(gt, outline=(255, 230, 0), width=2)
    # predictions: green TP / red FP
    tp = fp = 0
    for conf, xy, iou, is_tp in matched:
        if is_tp:
            tp += 1
            col = (0, 220, 0)
        else:
            fp += 1
            col = (240, 40, 40)
        d.rectangle(xy, outline=col, width=2)
        d.text((xy[0] + 1, max(0, xy[1] - 11)),
               f'{conf:.2f} i{iou:.2f}', fill=col, font=f_box)
    panel.paste(img, (0, cap_h))
    cd = ImageDraw.Draw(panel)
    cd.text((6, 6),
            f'{name}: {len(matched)} preds  |  {tp} TP / {fp} FP  |  {len(gts)} GT',
            fill=(255, 255, 255), font=_font(15))
    return panel, tp, fp


def main():
    device = 'cpu'  # force CPU: 3090 busy with another job
    samples = load_inria('./INRIAPerson', split='test', max_images=None)
    det = YOLODetector('yolov5s', device, conf_threshold=CONF)
    applier = PatchApplier(PSIZE)
    to_t = T.ToTensor()
    patches = {n: load_patch(k, p, device) for n, k, p in METHODS}

    out = 'output_v2_squash/iou_diag'
    os.makedirs(out, exist_ok=True)
    print(f'writing panels to {out}/')

    for idx in SAMPLES:
        img, bboxes = samples[idx]
        img640 = img.resize((W, H))
        placements, gts = place_patches_on_all_bboxes(
            img, bboxes, PSIZE, (W, H), frac=FRAC)
        panels = []
        for name, kind, _ in METHODS:
            it = to_t(img640).to(device)
            p = patches[name]
            if p is not None:
                with torch.no_grad():
                    for (x, y, s) in placements:
                        it = applier.apply_patch(it, p, x, y, s)
            pil = Image.fromarray(
                (it.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype('uint8'))
            dets = det.detect(pil)
            preds = [(d['confidence'], tuple(d['bbox']))
                     for d in dets if d['class'] == 0]
            matched = match(preds, gts)
            panel, tp, fp = draw_panel(pil, matched, gts, name)
            panels.append(panel)
            print(f'  img#{idx:3d} {name:11s}: {len(preds):2d} preds '
                  f'({tp} TP / {fp} FP), {len(gts)} GT')

        row = Image.new('RGB', (W * len(panels), H + 28), (0, 0, 0))
        for i, pn in enumerate(panels):
            row.paste(pn, (i * W, 0))
        fn = f'{out}/iou_diag_idx{idx}.png'
        row.save(fn)
        print(f'  saved {fn}')


if __name__ == '__main__':
    main()
