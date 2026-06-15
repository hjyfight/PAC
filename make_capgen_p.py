"""Build CAPGen-P (and CAPGen-R) patches.

Paper §3.3/§4.2 describes CAPGen-P as: keep a strong adversarial pattern (its color
probability matrix r_k) and replace only its base colors (Eq.4: t_ij = sum_k nc_k * r_k).

Three construction methods are provided:

  --method recolor-trained  (RECOMMENDED, faithful to the paper)
      Take a TRAINED 3-color CAPGen-T pattern (--source_matrix) and recolor its base
      colors to Bc1/Bc2. This is an EXACT Eq.4 recolor (capgen_p_orig === the source
      patch). The paper's "color-unrestricted AdvPatch" is approximated by a 3-color
      CAPGen trained with good (K-means env) colors -- the strongest faithful source
      pattern that actually has a (r_k, base-colors) decomposition.

  --method soft  /  --method legacy-kmeans  (CRITIQUE / comparison only)
      Decompose a free-pixel AdvPatch into (r_k, 3 source colors). This CANNOT reproduce
      the paper: 3 convex-weighted colors span only a 2-D triangle in RGB, so a free-pixel
      patch reconstructs at only ~13 dB PSNR and the attack is already lost before any
      recolor. Kept as evidence for the "free-pixel decomposition fails" analysis.

Usage (faithful):
    python make_capgen_p.py --method recolor-trained \
        --source_matrix output_new/capgen_t/best_color_prob.pt --out_dir output_new/capgen_p
Usage (critique artifact):
    python make_capgen_p.py --method soft \
        --advpatch output_new/advpatch/best_advpatch.pt --out_dir output_new/capgen_p_soft
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - only needed for optional init/legacy mode
    KMeans = None

# Paper §4.2 base color sets (RGB 0-255)
BC1 = np.array([[119, 49, 72], [2, 204, 1], [134, 2, 182]], dtype=np.float32)
BC2 = np.array([[199, 21, 131], [40, 165, 4], [16, 69, 120]], dtype=np.float32)


def load_advpatch(path):
    """Return (3, P, P) float patch in [0,1] from a *.pt (patch_logits) or PNG."""
    if path.lower().endswith('.pt'):
        ck = torch.load(path, map_location='cpu')
        patch = torch.sigmoid(ck['patch_logits']).numpy()        # (3, P, P)
    else:
        img = Image.open(path).convert('RGB')
        patch = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return patch


def save_patch_png(path, patch_chw):
    arr = (patch_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def logits_from_labels(labels, P, k, B=10.0):
    """One-hot-ish logits (P, P, k): +B at the assigned cluster, -B elsewhere, so
    sigmoid -> log -> softmax(/tau) reproduces a hard per-pixel base-color pick
    (exactly the path CAPGenGenerator.forward / eval_inria.py take)."""
    onehot = np.full((P * P, k), -B, dtype=np.float32)
    onehot[np.arange(P * P), labels] = B
    return onehot.reshape(P, P, k)


def logits_from_probs(probs_hwk, tau=0.1, eps=1e-6, m_max=0.99):
    """Invert CAPGen's sigmoid/log/softmax path for a desired soft assignment.

    CAPGen renders r = softmax(log(sigmoid(logits)) / tau). For any probability
    tensor r, choose m = m_max * r**tau. Then log(m)/tau = const + log(r), whose
    softmax is r. This lets us save continuous AdvPatch pattern information
    without forcing a hard one-hot K-means assignment.
    """
    probs = np.asarray(probs_hwk, dtype=np.float32)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    m = m_max * np.power(probs, tau)
    m = np.clip(m, eps, 1.0 - eps)
    return np.log(m / (1.0 - m)).astype(np.float32)


def save_color_prob(path, logits_hwk, base_colors_255, tau=0.1):
    ck = {
        'logits': torch.tensor(logits_hwk, dtype=torch.float32),
        'base_colors': torch.tensor(base_colors_255 / 255.0, dtype=torch.float32),
        'num_base_colors': int(logits_hwk.shape[-1]),
        'temperature': tau,
        'patch_size': int(logits_hwk.shape[0]),
    }
    torch.save(ck, path)


def render_array(logits_hwk, base_colors_255, tau=0.1):
    """Replicate CAPGenGenerator.forward in numpy, just for a visual PNG."""
    m = 1.0 / (1.0 + np.exp(-logits_hwk))
    logm = np.log(m + 1e-8) / tau
    logm = logm - logm.max(-1, keepdims=True)
    e = np.exp(logm)
    r = e / e.sum(-1, keepdims=True)                              # (P, P, k)
    t = r @ (base_colors_255 / 255.0)                            # (P, P, 3)
    return t.clip(0.0, 1.0)


def render_png(path, logits_hwk, base_colors_255, tau=0.1):
    arr = (render_array(logits_hwk, base_colors_255, tau) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def emit(out_dir, name, logits, palette, tau):
    cp = os.path.join(out_dir, f'{name}_color_prob.pt')
    png = os.path.join(out_dir, f'{name}.png')
    save_color_prob(cp, logits, palette, tau)
    render_png(png, logits, palette, tau)
    print(f"  saved {name:14s} -> {cp}  (+ {os.path.basename(png)})")


def _np_logit(x, eps=1e-5):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def emit_random_R(out_dir, P, k, tau, seed):
    """CAPGen-R (paper Eq.5): a random CONTINUOUS color-probability matrix
    m_r ~ Uniform(0,1), recolored with Bc1/Bc2.

    The render path computes r = softmax(log(sigmoid(logits))/tau), so encoding
    logits = logit(m_r) makes sigmoid(logits) == m_r and reproduces Eq.5 exactly:
    r_k = Softmax(log(m_r)_ijk / tau). One shared m_r for R1/R2 so only the base
    colors differ. (Previously this used random HARD one-hot labels, which is a
    random assignment, not the paper's random weights.)
    """
    rng = np.random.RandomState(seed)
    m_r = rng.uniform(1e-3, 1.0 - 1e-3, size=(P, P, k)).astype(np.float32)
    rlogits = _np_logit(m_r).astype(np.float32)
    print("CAPGen-R (RANDOM continuous weights + base colors, weak control):")
    emit(out_dir, 'capgen_r1', rlogits, BC1, tau)
    emit(out_dir, 'capgen_r2', rlogits, BC2, tau)


def build_recolor_trained(args):
    """
    
    Faithful CAPGen-P (paper Eq.4): keep a TRAINED CAPGen-T pattern (its color
    probability matrix) and replace only the base colors with Bc1/Bc2.

    Unlike --method soft (which lossily fits a free-pixel AdvPatch into 3 colors,
    PSNR ~13 dB), this recolor is EXACT: capgen_p_orig === the source patch, so any
    attack drop on P1/P2 is purely the color swap (Eq.4), as the paper intends.
    """
    ck = torch.load(args.source_matrix, map_location='cpu') # 读取之前训好的模型
    logits = ck['logits'].numpy().astype(np.float32)                 # (P,P,K) trained pattern
    src_colors = (ck['base_colors'].numpy() * 255.0).astype(np.float32)  # (K,3) in 0-255，提取训练时的原始颜色
    P, _, K = logits.shape
    tau = float(ck.get('temperature', args.tau))
    if K != BC1.shape[0]: # 颜色的种类保持一致
        raise SystemExit(f"source matrix has K={K} base colors but Bc1/Bc2 define "
                         f"{BC1.shape[0]}; recolor-trained needs matching K (paper uses 3).")
    print(f"Recoloring trained pattern from {args.source_matrix}: P={P}, K={K}, tau={tau}")
    print(f"Source base colors (RGB):\n{src_colors.round(1)}\n")

    print("CAPGen-P (trained CAPGen-T pattern + new colors, faithful Eq.4):")
    emit(args.out_dir, 'capgen_p1', logits, BC1, tau)               # paper Bc1
    emit(args.out_dir, 'capgen_p2', logits, BC2, tau)               # paper Bc2
    emit(args.out_dir, 'capgen_p_orig', logits, src_colors, tau)    # exact == source patch，不换色的

    emit_random_R(args.out_dir, P, K, tau, args.seed)

    summary = {
        'method': 'recolor-trained',
        'source_matrix': args.source_matrix,
        'num_colors': K,
        'tau': tau,
        'source_colors_rgb': src_colors.tolist(),
        'note': ('Faithful Eq.4 CAPGen-P: a trained CAPGen-T pattern recolored to Bc1/Bc2. '
                 'capgen_p_orig renders identically to the source patch; evaluate P1/P2 vs '
                 'P_orig to isolate the pure effect of the color swap.'),
    }
    summary_path = os.path.join(args.out_dir, 'capgen_p_decomposition.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone -> {args.out_dir}/")
    print(f"Summary -> {summary_path}")
    print("Evaluate on INRIA test, e.g.:")
    print(f"  python eval_inria.py --dataset_dir ./INRIAPerson "
          f"--color_prob_path {args.out_dir}/capgen_p1_color_prob.pt "
          f"--output_json {args.out_dir}/eval_p1.json")


def init_palette_luma(pixels, k):
    """Initialize source colors by luminance bins; no hard pattern is kept."""
    lum = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    edges = np.quantile(lum, np.linspace(0.0, 1.0, k + 1))
    centers = []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        if i == k - 1:
            mask = (lum >= lo) & (lum <= hi)
        else:
            mask = (lum >= lo) & (lum < hi)
        if not np.any(mask):
            mid = 0.5 * (lo + hi)
            mask[np.argmin(np.abs(lum - mid))] = True
        centers.append(pixels[mask].mean(axis=0))
    centers = np.asarray(centers, dtype=np.float32)
    order = np.argsort(centers @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return centers[order]


def init_palette_kmeans(pixels, k, seed):
    if KMeans is None:
        raise SystemExit("scikit-learn is required for --init_palette kmeans or --method legacy-kmeans")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(pixels)
    centers = km.cluster_centers_.astype(np.float32)
    order = np.argsort(centers @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return centers[order]


def init_palette_random(pixels, k, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pixels), size=k, replace=False)
    return pixels[idx].astype(np.float32)


def initial_assignment_scores(pixels, palette, P):
    dist2 = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1)
    scale = max(float(np.std(dist2)), 1e-4)
    return (-dist2 / scale).reshape(P, P, palette.shape[0]).astype(np.float32)


def reconstruction_metrics(patch_chw, logits_hwk, palette_255, tau):
    target = patch_chw.transpose(1, 2, 0)
    recon = render_array(logits_hwk, palette_255, tau)
    err = recon - target
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))
    return {'mse': mse, 'mae': mae, 'max_abs': max_abs, 'psnr': psnr}


def soft_decompose_advpatch(patch_chw, k, steps, lr, seed, init_palette, device):
    """Find soft r_k and source colors whose rendering reconstructs AdvPatch."""
    _, P, _ = patch_chw.shape
    pixels = patch_chw.transpose(1, 2, 0).reshape(-1, 3).astype(np.float32)

    if init_palette == 'luma':
        palette0 = init_palette_luma(pixels, k)
    elif init_palette == 'kmeans':
        palette0 = init_palette_kmeans(pixels, k, seed)
    elif init_palette == 'random':
        palette0 = init_palette_random(pixels, k, seed)
    else:
        raise ValueError(f"unknown init_palette: {init_palette}")

    torch.manual_seed(seed)
    target = torch.tensor(pixels.reshape(P, P, 3), dtype=torch.float32, device=device)
    assign_logits = torch.nn.Parameter(
        torch.tensor(initial_assignment_scores(pixels, palette0, P), device=device)
    )
    palette_logits = torch.nn.Parameter(
        torch.tensor(_np_logit(palette0), dtype=torch.float32, device=device)
    )
    opt = torch.optim.Adam([assign_logits, palette_logits], lr=lr)

    best_loss = float('inf')
    best_r = None
    best_palette = None
    log_every = max(1, steps // 5)

    for step in range(steps):
        r = torch.softmax(assign_logits, dim=-1)
        palette = torch.sigmoid(palette_logits)
        recon = torch.einsum('hwk,kc->hwc', r, palette)
        loss = ((recon - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_r = r.detach().cpu().numpy()
            best_palette = palette.detach().cpu().numpy()
        if step == 0 or (step + 1) % log_every == 0 or step == steps - 1:
            print(f"  soft-decompose step {step + 1:4d}/{steps}: mse={loss_value:.8f}")

    return best_r.astype(np.float32), (best_palette * 255.0).astype(np.float32), best_loss


def legacy_kmeans_decompose(patch_chw, k, seed, logit_strength):
    """Old lossy baseline: K-means centers plus hard one-hot labels."""
    if KMeans is None:
        raise SystemExit("scikit-learn is required for --method legacy-kmeans")
    _, P, _ = patch_chw.shape
    pixels = patch_chw.reshape(3, -1).T
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(pixels)
    logits = logits_from_labels(km.labels_, P, k, B=logit_strength)
    centers = km.cluster_centers_ * 255.0
    return logits, centers.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--advpatch', default='output_new/advpatch/best_advpatch.pt',
                    help='(soft/legacy-kmeans) trained free-pixel AdvPatch (*.pt with patch_logits, or PNG)')
    ap.add_argument('--source_matrix', default='output_new/capgen_t/best_color_prob.pt',
                    help='(recolor-trained) trained CAPGen-T color_prob.pt whose pattern is '
                         'recolored to Bc1/Bc2 -- the faithful Eq.4 CAPGen-P source')
    ap.add_argument('--out_dir', default='output_new/capgen_p')
    ap.add_argument('--num_colors', type=int, default=3)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--method', choices=['recolor-trained', 'soft', 'legacy-kmeans'], default='recolor-trained',
                    help='recolor-trained = faithful Eq.4 (recolor a trained CAPGen-T pattern, RECOMMENDED); '
                         'soft/legacy-kmeans = decompose a free-pixel AdvPatch (critique artifact, cannot reproduce)')
    ap.add_argument('--init_palette', choices=['luma', 'kmeans', 'random'], default='luma',
                    help='initial source palette for the soft reconstruction optimizer')
    ap.add_argument('--recon_steps', type=int, default=1200)
    ap.add_argument('--recon_lr', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--legacy_logit_strength', type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.method == 'recolor-trained':
        build_recolor_trained(args)
        return

    patch = load_advpatch(args.advpatch)                          # (3, P, P) in [0,1]
    _, P, _ = patch.shape
    save_patch_png(os.path.join(args.out_dir, 'advpatch_raw.png'), patch)

    print(f"AdvPatch loaded: P={P}, k={args.num_colors}, method={args.method}")
    if args.method == 'soft':
        device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
        if device == 'auto':
            device = 'cpu'
        print(f"Soft decomposition on {device} (init={args.init_palette})")
        probs, centers, best_loss = soft_decompose_advpatch(
            patch, args.num_colors, args.recon_steps, args.recon_lr,
            args.seed, args.init_palette, device,
        )
        logits = logits_from_probs(probs, tau=args.tau)
    else:
        logits, centers = legacy_kmeans_decompose(
            patch, args.num_colors, args.seed, args.legacy_logit_strength,
        )
        best_loss = None

    metrics = reconstruction_metrics(patch, logits, centers, args.tau)
    print(f"AdvPatch source colors (RGB):\n{centers.round(1)}\n")
    print("P_orig reconstruction vs raw AdvPatch:")
    print(f"  mse={metrics['mse']:.8f}  mae={metrics['mae']:.6f}  "
          f"max_abs={metrics['max_abs']:.6f}  psnr={metrics['psnr']:.2f} dB\n")

    print("CAPGen-P (AdvPatch pattern + new colors):")
    emit(args.out_dir, 'capgen_p1', logits, BC1, args.tau)        # paper Bc1
    emit(args.out_dir, 'capgen_p2', logits, BC2, args.tau)        # paper Bc2
    emit(args.out_dir, 'capgen_p_orig', logits, centers, args.tau)  # sanity: should match raw AdvPatch

    emit_random_R(args.out_dir, P, args.num_colors, args.tau, args.seed)

    summary = {
        'advpatch': args.advpatch,
        'method': args.method,
        'num_colors': args.num_colors,
        'tau': args.tau,
        'init_palette': args.init_palette if args.method == 'soft' else None,
        'recon_steps': args.recon_steps if args.method == 'soft' else None,
        'recon_lr': args.recon_lr if args.method == 'soft' else None,
        'best_optimizer_mse': best_loss,
        'source_colors_rgb': centers.tolist(),
        'p_orig_reconstruction': metrics,
        'note': (
            'Evaluate raw AdvPatch and capgen_p_orig. If P_orig is much weaker, '
            'CAPGen-P1/P2 cannot be interpreted as the paper CAPGen-P.'
        ),
    }
    summary_path = os.path.join(args.out_dir, 'capgen_p_decomposition.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone -> {args.out_dir}/")
    print(f"Decomposition summary -> {summary_path}")
    print("Evaluate each on INRIA test, e.g.:")
    print(f"  python eval_inria.py --dataset_dir ./INRIAPerson "
           f"--color_prob_path {args.out_dir}/capgen_p1_color_prob.pt "
          f"--output_json {args.out_dir}/eval_p1.json")
    print(f"  python eval_inria.py --dataset_dir ./INRIAPerson "
           f"--raw_patch_pt {args.advpatch} "
           f"--output_json {args.out_dir}/eval_advpatch.json   # the raw AdvPatch baseline")
    print(f"  python eval_inria.py --dataset_dir ./INRIAPerson "
          f"--color_prob_path {args.out_dir}/capgen_p_orig_color_prob.pt "
          f"--output_json {args.out_dir}/eval_p_orig.json      # P_orig sanity check")


if __name__ == '__main__':
    main()
