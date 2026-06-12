"""
EOT (Expectation Over Transformation) Module
Implements data augmentation for adversarial patch robustness.

Per paper supplementary (Appendix D), EOT applies:
- Smoothing   m' = m * s  (Eq.6, uniform kernel, reduces printing error)
- Contrast    D ~ Uniform(0.8, 1.2)
- Brightness  B ~ Uniform(0.9, 1.1)
- Noise       N ~ Uniform(0, 0.1)
- Rotation    θ ~ Uniform(-20°, 20°)
- Scale       S ~ Uniform(0.9, 1.1)

All transformations in `DifferentiableEOT` are implemented with
torch primitives so that gradients flow back to the patch parameters.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import random


class DifferentiableEOT(nn.Module):
    """
    Fully differentiable EOT pipeline used during training.

    Applies (in order): smoothing, rotation, scale, brightness, contrast,
    additive noise. Rotation/scale use F.affine_grid + F.grid_sample (bilinear)
    so gradients propagate through the spatial transform.
    """

    def __init__(
        self,
        rotation_range=(-20, 20),
        brightness_range=(0.9, 1.1),
        contrast_range=(0.8, 1.2),
        noise_range=(0.0, 0.1),
        scale_range=(0.9, 1.1),
        smooth_kernel_size=3,
    ):
        super().__init__()
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_range = noise_range
        self.scale_range = scale_range
        # Paper Eq.6: m' = m * s, a uniform smoothing kernel applied first "for
        # reducing the printing error". Odd kernel size keeps the patch size via
        # symmetric padding; set <=1 to disable.
        self.smooth_kernel_size = smooth_kernel_size

    @staticmethod
    def _sample(rng, device):
        lo, hi = rng
        return torch.empty(1, device=device).uniform_(lo, hi).item()

    def forward(self, patch):
        """
        Args:
            patch: (B, C, H, W) tensor in [0, 1]

        Returns:
            Transformed patch of same shape.
        """
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        B, C, H, W = patch.shape
        device = patch.device

        # ----- Smoothing m' = m * s (paper Eq.6, applied first) -----
        # Differentiable depthwise averaging conv; constant kernel so gradients
        # flow to the patch. Training-time only (forward is called by the trainer).
        k = self.smooth_kernel_size
        if k and k > 1:
            kernel = torch.ones(C, 1, k, k, device=device, dtype=patch.dtype) / float(k * k)
            patch = F.conv2d(patch, kernel, padding=k // 2, groups=C)

        # ----- Rotation + scale via affine_grid / grid_sample -----
        theta_deg = self._sample(self.rotation_range, device)
        theta_rad = math.radians(theta_deg)
        s = self._sample(self.scale_range, device)

        cos_t = math.cos(theta_rad) / s
        sin_t = math.sin(theta_rad) / s
        # 2x3 affine matrix in normalized coordinates
        theta_mat = torch.tensor(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0]],
            dtype=patch.dtype, device=device
        ).unsqueeze(0).expand(B, -1, -1)

        grid = F.affine_grid(theta_mat, patch.shape, align_corners=False)
        patch = F.grid_sample(
            patch, grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )

        # ----- Brightness (multiplicative) -----
        # NOTE: paper Eq.7 writes brightness as ADDITIVE (mp = D*m' + B) operating on
        # the color-probability matrix m. We apply it multiplicatively on the rendered
        # patch because additive B~U(0.9,1.1) on a [0,1] RGB image would saturate it to
        # white. Disclosed deviation; preserves the intended brightness augmentation.
        b = self._sample(self.brightness_range, device)
        patch = patch * b

        # ----- Contrast around per-image mean -----
        c = self._sample(self.contrast_range, device)
        mean = patch.mean(dim=[2, 3], keepdim=True)
        patch = (patch - mean) * c + mean

        # ----- Additive UNIFORM noise (paper Eq.7: N ~ Uniform(0, 0.1)) -----
        # Per-pixel symmetric-uniform perturbation scaled by an amplitude n ~ U(0, n_max).
        # Symmetric (zero-mean) so noise does not bias brightness; uniform (not Gaussian)
        # to match the paper's distribution family.
        if self.training:
            n_amp = self._sample(self.noise_range, device)
            if n_amp > 0:
                patch = patch + (torch.rand_like(patch) * 2.0 - 1.0) * n_amp

        patch = torch.clamp(patch, 0.0, 1.0)
        return patch


class EOTTransform:
    """Non-differentiable EOT (kept for evaluation / inspection)."""

    def __init__(
        self,
        rotation_range=(-20, 20),
        brightness_range=(0.9, 1.1),
        contrast_range=(0.8, 1.2),
        noise_std=0.1,
        scale_range=(0.9, 1.1),
    ):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, patch):
        angle = random.uniform(*self.rotation_range)
        patch = TF.rotate(patch, angle)

        scale = random.uniform(*self.scale_range)
        _, h, w = patch.shape if patch.dim() == 3 else patch.shape[1:]
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        patch = TF.resize(patch, [nh, nw])
        patch = TF.resize(patch, [h, w])  # back to original size

        b = random.uniform(*self.brightness_range)
        patch = TF.adjust_brightness(patch, b)

        c = random.uniform(*self.contrast_range)
        patch = TF.adjust_contrast(patch, c)

        if self.noise_std > 0:
            patch = patch + torch.randn_like(patch) * self.noise_std

        return torch.clamp(patch, 0, 1)


class PatchApplier:
    """
    Differentiable patch application onto images.

    Uses mask-based blending so gradients flow through the patch region
    while leaving the rest of the image as a constant background:

        img_out = (1 - mask) * image + mask * patch_placed
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    @staticmethod
    def _place(image_shape, patch, x, y):
        """Place `patch` inside a zero canvas of `image_shape` at (y, x)."""
        C, H, W = image_shape
        ph, pw = patch.shape[-2:]

        # Clip placement to image bounds
        x = max(0, min(int(x), W - pw))
        y = max(0, min(int(y), H - ph))

        canvas = torch.zeros(C, H, W, dtype=patch.dtype, device=patch.device)
        mask = torch.zeros(1, H, W, dtype=patch.dtype, device=patch.device)
        canvas[:, y:y + ph, x:x + pw] = patch
        mask[:, y:y + ph, x:x + pw] = 1.0
        return canvas, mask

    def apply_patch(self, image, patch, x, y, scale=1.0):
        """
        Apply (resized) patch to image at (x, y) via differentiable blending.

        Args:
            image: (C, H, W)
            patch: (C, ph, pw) — current adversarial patch (with grad)
            x, y, scale: placement & resize

        Returns:
            Image with patch blended in. Same shape as `image`. Gradients
            flow through the patch region.
        """
        if image.dim() == 4:
            image = image[0]
        C, H, W = image.shape

        ph = max(1, int(patch.shape[-2] * scale))
        pw = max(1, int(patch.shape[-1] * scale))

        # Differentiable resize
        patch_resized = F.interpolate(
            patch.unsqueeze(0), size=(ph, pw),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        canvas, mask = self._place(image.shape, patch_resized, x, y)
        return (1.0 - mask) * image + mask * canvas

    def apply_patch_random(self, image, patch, scale_range=(0.3, 0.6)):
        if image.dim() == 4:
            image = image[0]
        _, h, w = image.shape
        scale = random.uniform(*scale_range)
        ph = max(1, int(patch.shape[-2] * scale))
        pw = max(1, int(patch.shape[-1] * scale))
        x = random.randint(0, max(0, w - pw))
        y = random.randint(0, max(0, h - ph))
        return self.apply_patch(image, patch, x, y, scale)


if __name__ == "__main__":
    eot = DifferentiableEOT().train()
    p = torch.rand(1, 3, 300, 300, requires_grad=True)
    out = eot(p)
    out.sum().backward()
    print(f"EOT differentiable OK. grad norm = {p.grad.norm().item():.4f}")

    applier = PatchApplier(300)
    img = torch.rand(3, 640, 640)
    patch = torch.rand(3, 300, 300, requires_grad=True)
    out = applier.apply_patch(img, patch, 100, 100, 0.5)
    out.sum().backward()
    print(f"PatchApplier differentiable OK. grad norm = {patch.grad.norm().item():.4f}")
