"""Free-pixel AdvPatch trainer for CAPGen.

The paper's headline result (CAPGen-P, and the "pattern > color" finding) requires
a COLOR-UNRESTRICTED adversarial patch — i.e. an AdvPatch (Thys et al. 2019) whose
pixels are free RGB values, not constrained to k base colors. CAPGen-T (what
capgen_trainer.py trains) optimizes a color-probability matrix over 3 base colors;
this trainer instead optimizes the raw patch pixels directly.

Pipeline reuses the exact same differentiable machinery as CAPGen-T so results are
comparable: DifferentiableEOT + PatchApplier + YOLODetector.objectness_attack_loss,
bbox-based placement at 25% of bbox AREA, Adam lr=0.03, batch=8, 200 epochs,
full-dataset epochs.

Usage:
    python advpatch_trainer.py --dataset_dir ./INRIAPerson --num_iterations 200 \
        --output_dir ./output_advpatch

Output:
    best_patch.png / final_patch.png  — the trained free-pixel AdvPatch
    best_advpatch.pt / final_advpatch.pt — raw patch tensor (3, P, P) in [0,1]

This patch is the SOURCE PATTERN for CAPGen-P: feed it to make_capgen_p.py to recolor
it with environment base colors (Eq.4) without retraining.
"""
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from eot_transforms import DifferentiableEOT, PatchApplier
from detector import YOLODetector
from inria_dataset import load_inria


class FreePixelPatch(torch.nn.Module):
    """A color-unrestricted adversarial patch: raw RGB pixels in [0,1].

    The patch is parameterized by unconstrained logits passed through a sigmoid
    so the rendered patch always stays in [0,1] (same trick used for the color
    probability matrix, but here every pixel is free instead of a mix of k base
    colors).
    """

    def __init__(self, patch_size=300):
        super().__init__()
        self.patch_size = patch_size # 补丁的分辨率 300 * 300
        # init around 0 -> sigmoid -> ~0.5 grey; small noise breaks symmetry
        self.logits = torch.nn.Parameter(torch.randn(3, patch_size, patch_size) * 0.1) # 生成一个形状为（3，300，300）可学习张量

    def forward(self):
        return torch.sigmoid(self.logits)  # (3, P, P) in (0,1) ，带有微小噪声的灰色图片，sigmoid（0）= 0.5


def build_training_data(samples, patch_size, image_size=(640, 640)):
    """
    补丁 占据 bbox 面积的25%，这段代码作用是计算补丁应该缩放多少，以及贴到哪个位置上
    Mirror capgen_trainer.prepare_training_data: one (x, y, scale) per person
    bbox, patch side = sqrt(0.25 * bw * bh) (25% of bbox AREA), in resized coords.
    """
    H, W = image_size
    data = []
    for img, bboxes in samples:
        iw, ih = img.size # 原始图片的宽高
        sx, sy = W / iw, H / ih # 宽高的缩放比例
        positions = []
        for (x1, y1, x2, y2) in bboxes:
            bx1, by1, bx2, by2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy # 得到 640 * 640 中人的新坐标
            bw, bh = bx2 - bx1, by2 - by1 # 算出人在640 * 640中的宽和高
            bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5 # 算出人的中心坐标
            placed = max(1.0, (0.25 * bw * bh) ** 0.5) # 放置为25%的面积
            scale = float(max(1e-3, placed / patch_size)) 
            x = int(round(bcx - placed * 0.5))
            y = int(round(bcy - placed * 0.5))
            x = max(0, min(W - int(placed), x))
            y = max(0, min(H - int(placed), y)) # 算 补丁的位置
            positions.append((x, y, scale))
        if positions:
            data.append((img, positions))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', default='./INRIAPerson')
    ap.add_argument('--patch_size', type=int, default=300)
    ap.add_argument('--num_iterations', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--detector', default='yolov5s')
    ap.add_argument('--max_train_images', type=int, default=None)
    ap.add_argument('--save_interval', type=int, default=50)
    ap.add_argument('--output_dir', default='./output_advpatch')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | patch={args.patch_size} | batch={args.batch_size}"
          f" | lr={args.lr} | epochs={args.num_iterations}")

    samples = load_inria(args.dataset_dir, split='train',
                         max_images=args.max_train_images)
    if not samples:
        raise SystemExit(f"No INRIA train samples under {args.dataset_dir}")
    print(f"Loaded {len(samples)} INRIA train images")
    training_data = build_training_data(samples, args.patch_size)

    detector = YOLODetector(model_name=args.detector, device=device, conf_threshold=0.3)
    eot = DifferentiableEOT().to(device); eot.train()
    applier = PatchApplier(args.patch_size)
    patch_module = FreePixelPatch(args.patch_size).to(device)
    optimizer = optim.Adam(patch_module.parameters(), lr=args.lr)

    transform = transforms.Compose([
        transforms.Resize((640, 640)), transforms.ToTensor(),
    ])

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join('./runs', 'advpatch'))
    except ImportError:
        writer = None

    n = len(training_data)
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_iterations), desc="AdvPatch"):
        perm = np.random.permutation(n)
        ep_loss, seen = 0.0, 0
        for b in range(0, n, args.batch_size):
            idxs = perm[b:b + args.batch_size]
            optimizer.zero_grad()
            for i in idxs:
                img, positions = training_data[int(i)] # 计算补丁的放置位置以及缩放
                img_t = transform(img).to(device)
                patch = patch_module()
                img_with = img_t
                for (x, y, scale) in positions:
                    p_eot = eot(patch.unsqueeze(0)).squeeze(0) # 对补丁进行 EOT 
                    img_with = applier.apply_patch(img_with, p_eot, x, y, scale) # 贴补丁
                loss = detector.objectness_attack_loss(img_with.unsqueeze(0), target_class=0) # 计算损失
                (loss / len(idxs)).backward()
                ep_loss += loss.item(); seen += 1
            optimizer.step()
        avg = ep_loss / max(1, seen)
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}/{args.num_iterations}, Attack Loss: {avg:.4f}")
        if writer:
            writer.add_scalar('Loss/attack', avg, epoch)

        def _save(stem):
            p = patch_module().detach().cpu()
            arr = (p.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(args.output_dir, f'{stem}.png'))
            torch.save({'patch_logits': patch_module.logits.detach().cpu(),
                        'patch_size': args.patch_size},
                       os.path.join(args.output_dir, f'{stem.replace("patch","advpatch")}.pt'))

        if avg < best_loss:
            best_loss = avg
            _save('best_patch')
        if (epoch + 1) % args.save_interval == 0:
            _save(f'patch_epoch_{epoch+1}')

    _save_final = patch_module().detach().cpu()
    arr = (_save_final.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(args.output_dir, 'final_patch.png'))
    torch.save({'patch_logits': patch_module.logits.detach().cpu(), 'patch_size': args.patch_size},
               os.path.join(args.output_dir, 'final_advpatch.pt'))
    print(f"Done. Best attack loss: {best_loss:.4f}. Saved to {args.output_dir}/")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
