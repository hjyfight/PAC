"""
CAPGen Patch Generator
Core module implementing the Camouflaged Adversarial Patch Generator

Key components:
1. Color probability matrix optimization
2. Pattern-color decomposition
3. Fast color transfer for new environments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class ColorProbabilityMatrix(nn.Module):
    """
    Color probability matrix as described in the paper.

    The matrix m ∈ (0,1)^(W×H×3) determines the probability of each pixel
    belonging to each base color. The color at position (i,j) is computed as:

    t_ij = Σ(c_k · r_k) where r_k = Softmax(log(m_ijk) / τ)
    """

    def __init__(self, width, height, num_colors=3):
        super().__init__()
        self.width = width
        self.height = height
        self.num_colors = num_colors

        # Initialise logits with unit-variance noise so that, after
        #   m = sigmoid(logits)  ->  values in ~[0.27, 0.73]
        #   r = softmax(log(m)/τ) with τ=0.1
        # each pixel has a clear (different) dominant base-color at init.
        # This gives a richer pattern than near-uniform initialization and
        # therefore stronger gradient signal in early epochs.
        self.logits = nn.Parameter(torch.randn(height, width, num_colors))

    def forward(self, base_colors, temperature=0.1):
        """
        Generate patch from base colors and color probability matrix.

        Based on Eq.(3) from the paper:
        t_ij = Σ(c_k · r_k)
        r_k = Softmax(log(m_ijk) / τ)

        where m ∈ (0,1)^(W×H×3) is the color probability matrix.

        Args:
            base_colors: torch tensor of shape (num_colors, 3) - RGB values in [0,1]
            temperature: Temperature coefficient τ for Softmax

        Returns:
            patch: torch tensor of shape (3, H, W) - the generated patch
        """
        # Compute m = sigmoid(logits) to ensure m ∈ (0,1)
        m = torch.sigmoid(self.logits)  # (H, W, num_colors)

        # Compute r_k = Softmax(log(m_ijk) / τ) - Eq.(3)
        log_m = torch.log(m + 1e-8)  # Add epsilon for numerical stability
        r = F.softmax(log_m / temperature, dim=-1)  # (H, W, num_colors)

        # Compute patch colors: t_ij = Σ(c_k · r_k)
        # base_colors: (num_colors, 3)
        # r: (H, W, num_colors)
        patch = torch.einsum('hwk,kc->hwc', r, base_colors)  # (H, W, 3)

        # Clamp to [0, 1]
        patch = torch.clamp(patch, 0, 1)

        # Reshape to (3, H, W) for PyTorch format
        patch = patch.permute(2, 0, 1)

        return patch


class CAPGenGenerator:
    """
    CAPGen: Camouflaged Adversarial Patch Generator

    Main class for generating environment-adaptive adversarial patches.
    """

    def __init__(
        self,
        patch_size=300,
        num_base_colors=3,
        temperature=0.1,
        device='cpu'
    ):
        """
        Args:
            patch_size: Size of the adversarial patch (W=H)
            num_base_colors: Number of base colors (k in K-means)
            temperature: Temperature coefficient τ
            device: Device to use (cpu/cuda)
        """
        self.patch_size = patch_size
        self.num_base_colors = num_base_colors
        self.temperature = temperature
        self.device = device

        # Color probability matrix
        self.color_prob_matrix = None

        # Base colors from environment
        self.base_colors = None

        # Generated patch
        self.patch = None

    def set_base_colors(self, base_colors):
        """
        Set base colors extracted from environment.

        Args:
            base_colors: numpy array of shape (num_colors, 3) with RGB values [0, 255]
        """
        self.base_colors = torch.tensor(
            base_colors / 255.0,
            dtype=torch.float32,
            device=self.device
        )

    def initialize_color_prob_matrix(self):
        """
        Initialize the color probability matrix.
        """
        self.color_prob_matrix = ColorProbabilityMatrix(
            self.patch_size,
            self.patch_size,
            self.num_base_colors
        ).to(self.device)

    def generate_patch(self, temperature=None):
        """
        Generate adversarial patch using current base colors and color probability matrix.

        Args:
            temperature: Override temperature coefficient

        Returns:
            patch: torch tensor of shape (3, H, W)
        """
        if self.color_prob_matrix is None:
            self.initialize_color_prob_matrix()

        if self.base_colors is None:
            raise ValueError("Base colors not set. Call set_base_colors first.")

        temp = temperature if temperature is not None else self.temperature
        self.patch = self.color_prob_matrix(self.base_colors, temp)

        return self.patch

    def get_color_assignment(self, hard=False):
        """
        Return the paper's "pattern": the color-probability tensor r_k.

        Per paper rebuttal §6 (Reviewer 7Kmv reply), the *pattern* is defined
        as the relative pixel arrangement (i.e. which base color each pixel
        is assigned to). This is exactly the softmax r_k from Eq.(3), not a
        grayscale image.

        Args:
            hard: If True, return a one-hot argmax map (H, W) indicating the
                  dominant base-color index at each pixel. If False, return
                  the soft (H, W, K) probability tensor.

        Returns:
            Tensor — (H, W) long if hard else (H, W, K) float.
        """
        if self.color_prob_matrix is None:
            raise ValueError("No color probability matrix. Generate a patch first.")

        m = torch.sigmoid(self.color_prob_matrix.logits)
        log_m = torch.log(m + 1e-8)
        r = F.softmax(log_m / self.temperature, dim=-1)  # (H, W, K)
        if hard:
            return r.argmax(dim=-1)
        return r

    def get_pattern(self):
        """
        Backward-compatible "pattern" view: returns the grayscale of the
        current patch normalised to [0, 1]. NOTE: this is *not* the paper's
        definition of pattern; use `get_color_assignment()` for that.
        """
        if self.patch is None:
            raise ValueError("No patch generated. Call generate_patch first.")

        gray = 0.299 * self.patch[0] + 0.587 * self.patch[1] + 0.114 * self.patch[2]
        pattern = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        return pattern

    def transfer_colors(self, new_base_colors, temperature=None):
        """
        Fast color transfer: Replace base colors while keeping the pattern.

        This implements the fast adversarial patch generation strategy:
        "replacing the colors of high-performance patches with colors that match
        the surroundings, maintaining their effectiveness while ensuring they blend in."

        Args:
            new_base_colors: numpy array of shape (num_colors, 3) - new RGB values [0, 255]
            temperature: Override temperature coefficient

        Returns:
            new_patch: torch tensor of shape (3, H, W) with new colors
        """
        if self.color_prob_matrix is None:
            raise ValueError("No color probability matrix. Generate a patch first.")

        # Convert new base colors to tensor
        new_colors_tensor = torch.tensor(
            new_base_colors / 255.0,
            dtype=torch.float32,
            device=self.device
        )

        # Generate new patch with same color probability matrix but new colors
        temp = temperature if temperature is not None else self.temperature
        new_patch = self.color_prob_matrix(new_colors_tensor, temp)

        return new_patch

    def decompose(self):
        """
        Decompose patch into pattern (paper definition) and base colors.

        Per paper rebuttal §6 (Reviewer 7Kmv reply), the *pattern* is the
        color-probability tensor r_k from Eq.(3). The colors are simply the
        k base colors stored in `self.base_colors`.

        Returns:
            color_prob: (H, W, K) tensor — the soft color assignment r_k
                        (this IS the paper's "pattern")
            base_colors: (K, 3) tensor in [0, 1]
            grayscale: optional grayscale view of the rendered patch
        """
        if self.color_prob_matrix is None:
            color_prob = None
        else:
            color_prob = self.get_color_assignment(hard=False)

        grayscale = self.get_pattern() if self.patch is not None else None
        return color_prob, self.base_colors, grayscale

    def save_patch(self, path):
        """
        Save generated patch as image.

        Args:
            path: File path to save
        """
        if self.patch is None:
            raise ValueError("No patch generated.")

        # Convert to numpy and save
        patch_np = self.patch.detach().cpu().permute(1, 2, 0).numpy()
        patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(patch_np)
        img.save(path)

    def load_color_prob_matrix(self, path):
        """
        Load a saved color probability matrix and base colors.

        Tensors are moved to the generator's current device so that the
        generator can be used immediately afterwards on CPU or GPU.
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Infer K from checkpoint shape so we can rebuild the right module
        logits = checkpoint['logits']
        K = logits.shape[-1]
        if K != self.num_base_colors:
            self.num_base_colors = K

        # Adopt the checkpoint's geometry/temperature so rendering matches what
        # was saved (all current checkpoints are 300px / tau=0.1, so this is a
        # no-op for them -- it only guards against silent mismatches).
        if logits.shape[0] != self.patch_size:
            self.patch_size = int(logits.shape[0])
        if 'temperature' in checkpoint:
            self.temperature = float(checkpoint['temperature'])

        self.initialize_color_prob_matrix()
        self.color_prob_matrix.logits.data = logits.to(self.device)

        base_colors = checkpoint['base_colors']
        if isinstance(base_colors, torch.Tensor):
            self.base_colors = base_colors.to(self.device).float()
        else:
            self.base_colors = torch.tensor(
                np.asarray(base_colors, dtype=np.float32),
                device=self.device,
            )

    def save_color_prob_matrix(self, path):
        """Save color probability matrix and base colors for later reuse."""
        if self.color_prob_matrix is None:
            raise ValueError("No color probability matrix to save.")
        checkpoint = {
            'logits': self.color_prob_matrix.logits.detach().cpu(),
            'base_colors': (
                self.base_colors.detach().cpu()
                if isinstance(self.base_colors, torch.Tensor)
                else self.base_colors
            ),
            'num_base_colors': self.num_base_colors,
            'temperature': self.temperature,
            'patch_size': self.patch_size,
        }
        torch.save(checkpoint, path)


if __name__ == "__main__":
    # Test CAPGen generator
    generator = CAPGenGenerator(patch_size=100, num_base_colors=3)

    # Set sample base colors (R, G, B)
    base_colors = np.array([
        [34, 139, 34],   # Forest green
        [139, 119, 101], # Brown
        [85, 107, 47]    # Dark olive green
    ])
    generator.set_base_colors(base_colors)

    # Generate patch
    patch = generator.generate_patch()
    print(f"Generated patch shape: {patch.shape}")

    # Extract pattern
    pattern = generator.get_pattern()
    print(f"Pattern shape: {pattern.shape}")

    # Transfer to new colors (snow environment)
    new_colors = np.array([
        [255, 250, 250],  # Snow white
        [220, 220, 220],  # Light gray
        [192, 192, 192]   # Silver
    ])
    new_patch = generator.transfer_colors(new_colors)
    print(f"Transferred patch shape: {new_patch.shape}")
