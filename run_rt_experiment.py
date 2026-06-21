"""Run the CAPGen R/T-only experiment.

Protocol:
  - Draw one initial color-probability matrix m_r.
  - Render CAPGen-R1/R2 from that same m_r with Bc1/Bc2.
  - Train CAPGen-T1/T2 from that same m_r with Bc1/Bc2, producing m_t.
  - Evaluate only clean + R/T methods on INRIA test.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from capgen_trainer import CAPGenTrainer
from config import Config
from detector import YOLODetector
from eot_transforms import PatchApplier
from eval_inria import mAP50_backend
from inria_dataset import load_inria
from patch_generator import CAPGenGenerator
from run_table1 import eval_method, load_color_prob_patch


BC1 = np.array([[119, 49, 72], [2, 204, 1], [134, 2, 182]], dtype=np.float32)
BC2 = np.array([[199, 21, 131], [40, 165, 4], [16, 69, 120]], dtype=np.float32)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_color_prob(path, logits, base_colors_255, tau, patch_size):
    ckpt = {
        'logits': logits.detach().cpu().float(),
        'base_colors': torch.tensor(base_colors_255 / 255.0, dtype=torch.float32),
        'num_base_colors': int(logits.shape[-1]),
        'temperature': float(tau),
        'patch_size': int(patch_size),
    }
    torch.save(ckpt, path)


def render_color_prob_png(path, logits, base_colors_255, tau, patch_size, device):
    gen = CAPGenGenerator(
        patch_size=patch_size,
        num_base_colors=int(logits.shape[-1]),
        temperature=tau,
        device=device,
    )
    gen.set_base_colors(base_colors_255)
    gen.initialize_color_prob_matrix()
    gen.color_prob_matrix.logits.data.copy_(logits.to(device))
    with torch.no_grad():
        patch = gen.generate_patch().detach().cpu()
    arr = (patch.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def color_assignment_stats(logits, tau):
    m = torch.sigmoid(logits.float())
    r = torch.softmax(torch.log(m + 1e-8) / tau, dim=-1)
    mx = r.max(dim=-1).values
    ent = -(r * torch.log(r + 1e-12)).sum(-1) / np.log(r.shape[-1])
    return {
        'm_mean': float(m.mean()),
        'm_std': float(m.std()),
        'r_max_mean': float(mx.mean()),
        'r_max_p95': float(torch.quantile(mx, 0.95)),
        'entropy_mean': float(ent.mean()),
    }


def make_initial_logits(patch_size, num_colors, init_std, seed):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return torch.randn(
        patch_size,
        patch_size,
        num_colors,
        generator=g,
        dtype=torch.float32,
    ) * float(init_std)


def train_t(name, base_colors, initial_logits, train_samples, args, device):
    out_dir = os.path.join(args.out_dir, name)
    ensure_dir(out_dir)

    save_color_prob(
        os.path.join(out_dir, 'initial_color_prob.pt'),
        initial_logits,
        base_colors,
        args.temperature,
        args.patch_size,
    )
    render_color_prob_png(
        os.path.join(out_dir, 'initial_patch.png'),
        initial_logits,
        base_colors,
        args.temperature,
        args.patch_size,
        device,
    )

    train_images = [sample[0] for sample in train_samples]
    train_bboxes = [sample[1] for sample in train_samples]

    trainer = CAPGenTrainer(
        patch_size=args.patch_size,
        num_base_colors=args.num_colors,
        temperature=args.temperature,
        temp_start=args.temperature,
        anneal_temperature=False,
        learning_rate=args.lr,
        device=device,
        detector_model=args.detector,
        target_class=0,
        use_tensorboard=False,
        image_size=(640, 640),
    )
    trainer.train(
        env_images=train_images,
        train_images=train_images,
        train_bboxes=train_bboxes,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        output_dir=out_dir,
        fixed_base_colors=base_colors,
        resize=args.resize,
        patch_frac=args.patch_frac,
        initial_logits=initial_logits,
    )
    return out_dir


def evaluate_rt(paths, args, device):
    test_samples = load_inria(args.dataset_dir, split='test', max_images=args.max_test_images)
    print(f"Loaded {len(test_samples)} INRIA test images")
    detector = YOLODetector(model_name=args.detector, device=device, conf_threshold=args.conf)
    applier = PatchApplier(args.patch_size)

    results = {}
    print("  eval clean ...")
    results['clean'] = eval_method(
        None,
        test_samples,
        detector,
        applier,
        device,
        resize=args.resize,
        frac=args.patch_frac,
        eot=None,
    )
    for method, path in paths:
        print(f"  eval {method} ...")
        patch = load_color_prob_patch(path, device)
        results[method] = eval_method(
            patch,
            test_samples,
            detector,
            applier,
            device,
            resize=args.resize,
            frac=args.patch_frac,
            eot=None,
        )
    return results, len(test_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='output_rt')
    ap.add_argument('--dataset_dir', default='./INRIAPerson')
    ap.add_argument('--detector', default=Config.DETECTOR_MODEL)
    ap.add_argument('--patch_size', type=int, default=Config.PATCH_SIZE)
    ap.add_argument('--num_colors', type=int, default=Config.NUM_BASE_COLORS)
    ap.add_argument('--temperature', type=float, default=Config.TEMPERATURE)
    ap.add_argument('--init_std', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_iterations', type=int, default=Config.NUM_ITERATIONS)
    ap.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    ap.add_argument('--save_interval', type=int, default=Config.SAVE_INTERVAL)
    ap.add_argument('--resize', choices=['squash', 'letterbox'], default='squash')
    ap.add_argument('--patch_frac', type=float, default=0.25)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--max_train_images', type=int, default=None)
    ap.add_argument('--max_test_images', type=int, default=None)
    ap.add_argument('--skip_train', action='store_true')
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    initial_logits = make_initial_logits(
        args.patch_size,
        args.num_colors,
        args.init_std,
        args.seed,
    )

    r_dir = os.path.join(args.out_dir, 'capgen_r')
    ensure_dir(r_dir)
    r_paths = [
        ('CAPGen-R1', os.path.join(r_dir, 'capgen_r1_color_prob.pt'), BC1),
        ('CAPGen-R2', os.path.join(r_dir, 'capgen_r2_color_prob.pt'), BC2),
    ]
    for method, path, colors in r_paths:
        stem = os.path.splitext(os.path.basename(path))[0].replace('_color_prob', '')
        save_color_prob(path, initial_logits, colors, args.temperature, args.patch_size)
        render_color_prob_png(
            os.path.join(r_dir, f'{stem}.png'),
            initial_logits,
            colors,
            args.temperature,
            args.patch_size,
            device,
        )

    if args.skip_train:
        t1_dir = os.path.join(args.out_dir, 'capgen_t1')
        t2_dir = os.path.join(args.out_dir, 'capgen_t2')
    else:
        train_samples = load_inria(
            args.dataset_dir,
            split='train',
            max_images=args.max_train_images,
        )
        print(f"Loaded {len(train_samples)} INRIA train images")
        t1_dir = train_t('capgen_t1', BC1, initial_logits, train_samples, args, device)
        t2_dir = train_t('capgen_t2', BC2, initial_logits, train_samples, args, device)

    eval_paths = [
        ('CAPGen-R1', os.path.join(r_dir, 'capgen_r1_color_prob.pt')),
        ('CAPGen-R2', os.path.join(r_dir, 'capgen_r2_color_prob.pt')),
        ('CAPGen-T1', os.path.join(t1_dir, 'best_color_prob.pt')),
        ('CAPGen-T2', os.path.join(t2_dir, 'best_color_prob.pt')),
    ]
    for method, path in eval_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{method} checkpoint not found: {path}")

    results, num_test = evaluate_rt(eval_paths, args, device)
    out_json = os.path.join(args.out_dir, 'table_rt.json')
    payload = {
        'command': ' '.join([sys.executable] + sys.argv),
        'protocol': 'shared initial m_r for R1/R2/T1/T2; eval EOT off',
        'resize': args.resize,
        'patch_frac': args.patch_frac,
        'init_std': args.init_std,
        'seed': args.seed,
        'temperature': args.temperature,
        'mAP50_backend': mAP50_backend(),
        'detector': args.detector,
        'num_train_images': None if args.skip_train else len(train_samples),
        'num_test_images': num_test,
        'initial_stats': color_assignment_stats(initial_logits, args.temperature),
        'results': results,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    clean_map = results.get('clean', {}).get('mAP50')
    print("\nR/T-only Table (lower mAP50 = stronger)")
    print("%-11s %9s %9s %8s %7s" % ('Method', 'mAP50', 'drop', 'detRate', 'conf'))
    print("-" * 52)
    for method in ['clean', 'CAPGen-R1', 'CAPGen-R2', 'CAPGen-T1', 'CAPGen-T2']:
        r = results[method]
        drop = '' if clean_map is None else "%.1f" % (clean_map - r['mAP50'])
        print("%-11s %9.2f %9s %8.3f %7.3f"
              % (method, r['mAP50'], drop, r['det_rate'], r['conf']))
    print(f"\nsaved {out_json}")


if __name__ == '__main__':
    main()
