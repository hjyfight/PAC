"""End-to-end letterbox training + YOLOv5-official mAP evaluation.

Pipeline:
  1. Train AdvPatch with --resize_mode letterbox.
  2. Build CAPGen-P linear patches from the trained AdvPatch.
  3. Build CAPGen-R from one random continuous m_r.
  4. Train CAPGen-T1/T2 from the same m_r with fixed tau=0.1.
  5. Render clean/patched INRIA test sets as 640x640 letterbox images.
  6. Evaluate each set with yolov5/val.py and save official mAP results.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from capgen_trainer import CAPGenTrainer
from config import Config
from eot_transforms import PatchApplier
from inria_dataset import load_inria
from patch_generator import CAPGenGenerator
from resize_modes import build_bbox_patch_positions, image_to_training_tensor, transform_bboxes


BC1 = np.array([[119, 49, 72], [2, 204, 1], [134, 2, 182]], dtype=np.float32)
BC2 = np.array([[199, 21, 131], [40, 165, 4], [16, 69, 120]], dtype=np.float32)

COCO80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def run(cmd, cwd=None):
    print('\n$ ' + ' '.join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding='utf-8',
        errors='replace',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
    )
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}: {' '.join(str(x) for x in cmd)}")
    return proc.stdout


def np_logit(x, eps=1e-5):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def make_initial_logits(patch_size, num_colors, seed):
    rng = np.random.default_rng(seed)
    m_r = rng.uniform(1e-3, 1.0 - 1e-3, size=(patch_size, patch_size, num_colors)).astype(np.float32)
    return torch.tensor(np_logit(m_r), dtype=torch.float32)


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


def train_advpatch(args):
    adv_dir = args.out_dir
    best = adv_dir / 'best_advpatch.pt'
    if best.exists() and not args.force:
        print(f"[skip] AdvPatch exists: {best}")
        return best
    cmd = [
        sys.executable, 'advpatch_trainer.py',
        '--dataset_dir', args.dataset_dir,
        '--patch_size', str(args.patch_size),
        '--num_iterations', str(args.adv_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.adv_lr),
        '--detector', args.detector,
        '--resize_mode', 'letterbox',
        '--save_interval', str(args.save_interval),
        '--output_dir', str(adv_dir),
    ]
    if args.max_train_images is not None:
        cmd += ['--max_train_images', str(args.max_train_images)]
    run(cmd, cwd=args.repo_dir)
    if not best.exists():
        raise FileNotFoundError(best)
    return best


def build_p_series(args, advpatch_path):
    p_dir = args.out_dir / 'capgen_p_linear'
    p0 = p_dir / 'capgen_p_orig_linear.pt'
    p1 = p_dir / 'capgen_p1_linear.pt'
    p2 = p_dir / 'capgen_p2_linear.pt'
    if p0.exists() and p1.exists() and p2.exists() and not args.force:
        print(f"[skip] P series exists: {p_dir}")
        return p_dir
    p_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, 'make_capgen_p.py',
        '--method', 'advpatch-rgb-linear',
        '--advpatch', str(advpatch_path),
        '--out_dir', str(p_dir),
        '--num_colors', str(args.num_colors),
        '--tau', str(args.tau),
    ]
    run(cmd, cwd=args.repo_dir)
    return p_dir


def build_r_series(args, initial_logits, device):
    r_dir = args.out_dir / 'capgen_r'
    r_dir.mkdir(parents=True, exist_ok=True)
    items = [('capgen_r1', BC1), ('capgen_r2', BC2)]
    for stem, colors in items:
        pt = r_dir / f'{stem}_color_prob.pt'
        png = r_dir / f'{stem}.png'
        if pt.exists() and png.exists() and not args.force:
            print(f"[skip] {stem} exists: {pt}")
            continue
        save_color_prob(pt, initial_logits, colors, args.tau, args.patch_size)
        render_color_prob_png(png, initial_logits, colors, args.tau, args.patch_size, device)
        print(f"saved {pt}")
    return r_dir


def train_t(args, name, base_colors, initial_logits, train_samples, device):
    out_dir = args.out_dir / name
    final = out_dir / 'final_color_prob.pt'
    best = out_dir / 'best_color_prob.pt'
    if final.exists() and best.exists() and not args.force:
        print(f"[skip] {name} exists: {final}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_images = [x[0] for x in train_samples]
    train_bboxes = [x[1] for x in train_samples]
    trainer = CAPGenTrainer(
        patch_size=args.patch_size,
        num_base_colors=args.num_colors,
        temperature=args.tau,
        temp_start=args.tau,
        anneal_temperature=False,
        learning_rate=args.t_lr,
        device=device,
        detector_model=args.detector,
        target_class=0,
        use_tensorboard=False,
        image_size=(640, 640),
        resize_mode='letterbox',
    )
    trainer.train(
        env_images=train_images,
        train_images=train_images,
        train_bboxes=train_bboxes,
        num_iterations=args.t_epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        output_dir=str(out_dir),
        fixed_base_colors=base_colors,
        initial_logits=initial_logits,
    )
    if not best.exists():
        raise FileNotFoundError(best)
    return out_dir


def load_raw_patch(path, device):
    ck = torch.load(path, map_location=device)
    return torch.sigmoid(ck['patch_logits']).to(device).detach()


def load_color_prob_patch(path, device):
    ck = torch.load(path, map_location='cpu')
    patch_size = int(ck.get('patch_size', 300))
    num_colors = int(ck.get('num_base_colors', ck['logits'].shape[-1]))
    tau = float(ck.get('temperature', 0.1))
    gen = CAPGenGenerator(patch_size=patch_size, num_base_colors=num_colors, temperature=tau, device=device)
    gen.load_color_prob_matrix(str(path))
    with torch.no_grad():
        return gen.generate_patch().detach()



def yolo_hash(paths):
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.md5(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()


def write_yolo_cache(dataset_dir):
    img_dir = dataset_dir / 'images' / 'test'
    lab_dir = dataset_dir / 'labels' / 'test'
    im_files = sorted(str(p).replace('/', os.sep) for p in img_dir.glob('*.jpg'))
    label_files = [str((lab_dir / (Path(p).stem + '.txt'))).replace('/', os.sep) for p in im_files]
    cache = {}
    nf = nm = ne = nc = 0
    for im_file, lb_file in zip(im_files, label_files):
        if not os.path.exists(lb_file):
            nm += 1
            labels = np.zeros((0, 5), dtype=np.float32)
        else:
            rows = []
            with open(lb_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        rows.append([float(x) for x in parts])
            labels = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            if len(labels):
                nf += 1
            else:
                ne += 1
        cache[im_file] = [labels, (640, 640), []]
    cache['hash'] = yolo_hash(label_files + im_files)
    cache['results'] = (nf, nm, ne, nc, len(im_files))
    cache['msgs'] = []
    cache['version'] = 0.6
    cache_path = lab_dir.with_suffix('.cache')
    with open(cache_path, 'wb') as f:
        np.save(f, cache)
    return cache_path

def write_yolo_yaml(dataset_dir):
    yaml_path = dataset_dir / 'inria_person.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {dataset_dir.as_posix()}\n")
        f.write("train: images/test\nval: images/test\ntest: images/test\n\n")
        f.write("names:\n")
        for i, name in enumerate(COCO80):
            f.write(f"  {i}: {name}\n")
    return yaml_path


def write_label(path, bboxes, width=640, height=640):
    lines = []
    for x1, y1, x2, y2 in bboxes:
        x1 = max(0.0, min(float(width), float(x1)))
        x2 = max(0.0, min(float(width), float(x2)))
        y1 = max(0.0, min(float(height), float(y1)))
        y2 = max(0.0, min(float(height), float(y2)))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 1.0 or bh <= 1.0:
            continue
        xc = (x1 + x2) * 0.5 / width
        yc = (y1 + y2) * 0.5 / height
        lines.append(f"0 {xc:.8f} {yc:.8f} {bw / width:.8f} {bh / height:.8f}")
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def render_dataset(args, method, patch, samples, device):
    dataset_dir = args.out_dir / 'yolo_official_datasets' / method
    done = dataset_dir / '.complete'
    if done.exists() and not args.force:
        print(f"[skip] YOLO dataset exists: {dataset_dir}")
        yaml_path = write_yolo_yaml(dataset_dir)
        write_yolo_cache(dataset_dir)
        return yaml_path
    if dataset_dir.exists() and args.force:
        shutil.rmtree(dataset_dir)
    img_dir = dataset_dir / 'images' / 'test'
    lab_dir = dataset_dir / 'labels' / 'test'
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    applier = PatchApplier(args.patch_size)
    for i, (img, bboxes) in enumerate(tqdm(samples, desc=f'render {method}')):
        tensor = image_to_training_tensor(img, (640, 640), 'letterbox').to(device)
        label_boxes = transform_bboxes(img, bboxes, (640, 640), 'letterbox')
        if patch is not None:
            positions = build_bbox_patch_positions(img, bboxes, args.patch_size, (640, 640), 'letterbox')
            with torch.no_grad():
                for x, y, scale in positions:
                    tensor = applier.apply_patch(tensor, patch, x, y, scale)
        arr = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(img_dir / f'{i:06d}.jpg', quality=95)
        write_label(lab_dir / f'{i:06d}.txt', label_boxes)
    yaml_path = write_yolo_yaml(dataset_dir)
    write_yolo_cache(dataset_dir)
    done.write_text('ok\n', encoding='utf-8')
    return yaml_path


def parse_val_output(text):
    result = {'raw_output_tail': '\n'.join(text.splitlines()[-80:])}
    pat = re.compile(r'^\s*all\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', re.MULTILINE)
    m = pat.search(text)
    if not m:
        result['parse_error'] = True
        return result
    result.update({
        'images': int(m.group(1)),
        'instances': int(m.group(2)),
        'precision': float(m.group(3)),
        'recall': float(m.group(4)),
        'mAP50': float(m.group(5)),
        'mAP50_95': float(m.group(6)),
        'mAP50_percent': float(m.group(5)) * 100.0,
        'mAP50_95_percent': float(m.group(6)) * 100.0,
    })
    return result


def run_yolo_val(args, method, yaml_path):
    results_path = args.out_dir / 'official_yolo_results.json'
    existing = {}
    if results_path.exists() and not args.force_eval:
        existing = json.loads(results_path.read_text(encoding='utf-8'))
        if method in existing.get('results', {}) and 'mAP50' in existing['results'][method]:
            print(f"[skip] YOLO val exists: {method}")
            return existing['results'][method]
    cmd = [
        sys.executable, args.yolov5_dir / 'val.py',
        '--weights', args.yolov5_dir / 'yolov5s.pt',
        '--data', yaml_path,
        '--imgsz', '640',
        '--conf-thres', str(args.conf_thres),
        '--batch-size', str(args.val_batch_size),
        '--workers', '0',
        '--project', args.out_dir / 'yolo_official_runs',
        '--name', f'{method}_official_conf{str(args.conf_thres).replace(".", "")}',
        '--exist-ok',
    ]
    text = run(cmd, cwd=args.repo_dir)
    parsed = parse_val_output(text)
    return parsed


def save_results(args, results, meta):
    payload = {'meta': meta, 'results': results}
    json_path = args.out_dir / 'official_yolo_results.json'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    csv_path = args.out_dir / 'official_yolo_results.csv'
    keys = ['method', 'mAP50_percent', 'mAP50_95_percent', 'precision', 'recall', 'images', 'instances']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for method, r in results.items():
            row = {'method': method}
            row.update({k: r.get(k) for k in keys if k != 'method'})
            writer.writerow(row)
    print(f"\nSaved {json_path}")
    print(f"Saved {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='output_advpatch_retrain_20260622-letterbox_yolomap')
    ap.add_argument('--dataset_dir', default='./INRIAPerson')
    ap.add_argument('--detector', default=Config.DETECTOR_MODEL)
    ap.add_argument('--patch_size', type=int, default=Config.PATCH_SIZE)
    ap.add_argument('--num_colors', type=int, default=Config.NUM_BASE_COLORS)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--adv_epochs', type=int, default=Config.NUM_ITERATIONS)
    ap.add_argument('--t_epochs', type=int, default=Config.NUM_ITERATIONS)
    ap.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    ap.add_argument('--adv_lr', type=float, default=0.03)
    ap.add_argument('--t_lr', type=float, default=Config.LEARNING_RATE)
    ap.add_argument('--save_interval', type=int, default=Config.SAVE_INTERVAL)
    ap.add_argument('--max_train_images', type=int, default=None)
    ap.add_argument('--max_test_images', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None, help='Seed only for shared R/T initial m_r. Default: entropy.')
    ap.add_argument('--conf_thres', type=float, default=0.001)
    ap.add_argument('--val_batch_size', type=int, default=16)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--force_eval', action='store_true')
    args = ap.parse_args()

    args.repo_dir = Path(__file__).resolve().parent
    args.out_dir = (args.repo_dir / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    args.yolov5_dir = args.repo_dir / 'yolov5'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is None:
        args.seed = int(np.random.SeedSequence().entropy) % (2 ** 32 - 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Output: {args.out_dir}")
    print(f"Device: {device}")
    print(f"Resize mode: letterbox, tau={args.tau}, R/T seed={args.seed}")

    advpatch = train_advpatch(args)
    p_dir = build_p_series(args, advpatch)

    initial_logits = make_initial_logits(args.patch_size, args.num_colors, args.seed)
    r_dir = build_r_series(args, initial_logits, device)

    train_samples = load_inria(args.dataset_dir, split='train', max_images=args.max_train_images)
    print(f"Loaded {len(train_samples)} INRIA train images")
    t1_dir = train_t(args, 'capgen_t1', BC1, initial_logits, train_samples, device)
    t2_dir = train_t(args, 'capgen_t2', BC2, initial_logits, train_samples, device)

    test_samples = load_inria(args.dataset_dir, split='test', max_images=args.max_test_images)
    print(f"Loaded {len(test_samples)} INRIA test images")

    methods = [
        ('clean', None, None),
        ('advpatch', 'raw', advpatch),
        ('p0', 'raw', p_dir / 'capgen_p_orig_linear.pt'),
        ('p1', 'raw', p_dir / 'capgen_p1_linear.pt'),
        ('p2', 'raw', p_dir / 'capgen_p2_linear.pt'),
        ('r1', 'cp', r_dir / 'capgen_r1_color_prob.pt'),
        ('r2', 'cp', r_dir / 'capgen_r2_color_prob.pt'),
        ('t1', 'cp', t1_dir / 'best_color_prob.pt'),
        ('t2', 'cp', t2_dir / 'best_color_prob.pt'),
    ]

    results = {}
    for method, kind, path in methods:
        print(f"\n=== Official YOLO eval: {method} ===")
        if kind == 'raw':
            patch = load_raw_patch(path, device)
        elif kind == 'cp':
            patch = load_color_prob_patch(path, device)
        else:
            patch = None
        yaml_path = render_dataset(args, method, patch, test_samples, device)
        results[method] = run_yolo_val(args, method, yaml_path)
        meta = {
            'command': ' '.join(sys.argv),
            'out_dir': str(args.out_dir),
            'dataset_dir': args.dataset_dir,
            'resize_mode': 'letterbox',
            'eval_backend': 'yolov5/val.py official metrics',
            'conf_thres': args.conf_thres,
            'patch_frac': 0.25,
            'tau': args.tau,
            'seed_rt_initial_mr': args.seed,
            'advpatch': str(advpatch),
            'p_dir': str(p_dir),
            'r_dir': str(r_dir),
            't1_dir': str(t1_dir),
            't2_dir': str(t2_dir),
        }
        save_results(args, results, meta)

    clean = results.get('clean', {}).get('mAP50_percent')
    print('\nFinal YOLOv5 official mAP50')
    print('%-9s %9s %9s %9s %9s' % ('method', 'mAP50', 'drop', 'P', 'R'))
    print('-' * 50)
    for method, _, _ in methods:
        r = results.get(method, {})
        m = r.get('mAP50_percent')
        drop = '' if clean is None or m is None else f'{clean - m:.2f}'
        print('%-9s %9s %9s %9s %9s' % (
            method,
            '' if m is None else f'{m:.2f}',
            drop,
            '' if r.get('precision') is None else f"{r['precision']:.3f}",
            '' if r.get('recall') is None else f"{r['recall']:.3f}",
        ))


if __name__ == '__main__':
    main()