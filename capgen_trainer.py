"""
CAPGen Trainer
Optimizes the color probability matrix `m` so that the generated patch
attacks the chosen YOLO detector.

Implements the optimization framework from the paper:
    max_P  E_φ[ L( M(I + δ(P, O, φ; ε)), O, θ) ]

Per paper rebuttal §4 (Reviewer zes9 reply), the loss is back-propagated
through the entire pipeline (detector → EOT → patch generation) to update
the color probability matrix. R() is realised via EOT; S() is realised by
constraining the patch to k base colors — no explicit regularisation term.
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from patch_generator import CAPGenGenerator
from color_extractor import ColorExtractor
from eot_transforms import DifferentiableEOT, PatchApplier
from detector import YOLODetector
from resize_modes import build_bbox_patch_positions, image_to_training_tensor


class CAPGenTrainer:
    """
    Trainer for CAPGen adversarial patches.

    The optimisation target is the color probability matrix `m`
    inside `CAPGenGenerator.color_prob_matrix`. The detector weights are
    frozen.
    """

    def __init__(
        self,
        patch_size=300,
        num_base_colors=3,
        temperature=0.1,
        temp_start=0.1,
        anneal_temperature=False,
        learning_rate=0.03,
        device='cpu',
        detector_model='yolov5s',
        target_class=0,
        use_tensorboard=True,
        image_size=(640, 640),
        resize_mode='squash',
    ):
        self.patch_size = patch_size
        self.num_base_colors = num_base_colors
        self.temperature = temperature
        self.temp_start = temp_start
        self.anneal_temperature = anneal_temperature
        self.lr = learning_rate
        self.device = device
        self.target_class = target_class
        self.image_size = image_size
        self.resize_mode = resize_mode

        # Components
        self.color_extractor = ColorExtractor(num_colors=num_base_colors)
        self.generator = CAPGenGenerator(
            patch_size=patch_size,
            num_base_colors=num_base_colors,
            temperature=temperature,
            device=device,
        )
        self.eot = DifferentiableEOT(
            rotation_range=(-20, 20),
            brightness_range=(0.9, 1.1),
            contrast_range=(0.8, 1.2),
            noise_range=(0.0, 0.1),
            scale_range=(0.9, 1.1),
        ).to(device)
        self.eot.train()

        self.patch_applier = PatchApplier(patch_size)

        print(f"Loading detector: {detector_model}")
        self.detector = YOLODetector(
            model_name=detector_model,
            device=device,
            conf_threshold=0.3,
        )

        self.optimizer = None

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=os.path.join('./runs', 'capgen'))
                print("TensorBoard logging enabled")
            except ImportError:
                print("TensorBoard not available, console logging only")

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------
    def extract_environment_colors(self, env_images):
        print("Extracting base colors from environment images...")
        base_colors = self.color_extractor.extract_from_images(env_images)
        print(f"Extracted base colors:\n{base_colors}")
        return base_colors

    def prepare_training_data(self, images, patch_positions=None, bboxes_per_image=None):
        """
        Build a list of (img, positions) for each training image.

        Per paper §4.1: "we resize all images to 640×640 pixels and set the
        patch size to 25% of the object's bounding box." No image cropping.
        positions is a list of (x, y, scale) - one per person bbox in the
        image, all expressed in the resized 640×640 coordinate system.

        If `bboxes_per_image[i]` lists multiple person bboxes, ALL of them
        receive a patch placement (matches AdvPatch / CAPGen training: every
        target instance is attacked simultaneously). This is what the
        E_φ[L(M(I + δ), O)] expectation in §3.1 requires.

        `patch_positions`, if given, overrides everything and is interpreted
        as a single (x, y, scale) per image (legacy path for env-only runs).
        """
        H, W = self.image_size
        training_data = []
        for i, img in enumerate(images):
            if patch_positions is not None:
                # Legacy: single fixed position per image
                training_data.append((img, [patch_positions[i]]))
                continue

            if bboxes_per_image is not None and bboxes_per_image[i]:
                positions = build_bbox_patch_positions(
                    img,
                    bboxes_per_image[i],
                    self.patch_size,
                    image_size=self.image_size,
                    resize_mode=self.resize_mode,
                )
                training_data.append((img, positions))
            else:
                scale = float(np.random.uniform(0.3, 0.6))
                ph = max(1, int(self.patch_size * scale))
                pw = max(1, int(self.patch_size * scale))
                x = int(np.random.randint(0, max(1, W - pw)))
                y = int(np.random.randint(0, max(1, H - ph)))
                training_data.append((img, [(x, y, scale)]))
        return training_data

    # ------------------------------------------------------------------
    # Differentiable attack loss
    # ------------------------------------------------------------------
    def compute_attack_loss(self, image_tensor, patch, positions):
        """
        Differentiable objectness-attack loss.

        Pipeline (every step is differentiable wrt color_prob_matrix.logits):
          patch -> EOT (independent sample per bbox) -> placed onto image
                -> YOLOv5 raw forward
                -> max(obj * cls[target_class]) per image -> mean.

        Args:
            image_tensor: (3, H, W) image tensor already resized to
                          self.image_size and in [0, 1].
            patch:        (3, patch_size, patch_size) current patch
            positions:    list of (x, y, scale) - one per target person bbox,
                          all expressed in image_tensor's coordinate system.
                          ALL positions are patched on the SAME image before
                          a single detector forward (matches CAPGen §3.1:
                          L(M(I + δ), O) where every target object is hidden).
        """
        image_with_patch = image_tensor
        for (x, y, scale) in positions:
            # Independent EOT sample per bbox - different rotation/brightness
            # per instance, like wearing the patch at slightly different
            # angles. Gradients flow through patch in all of them.
            patch_eot = self.eot(patch.unsqueeze(0)).squeeze(0) # 先进行EOT
            image_with_patch = self.patch_applier.apply_patch( # 把补丁给掩码贴上去
                image_with_patch, patch_eot, x, y, scale
            )

        # Detector forward (raw, differentiable). image_tensor is already
        # at self.image_size, so no extra resize is needed.
        det_input = image_with_patch.unsqueeze(0)
        loss = self.detector.objectness_attack_loss( # 计算损失
            det_input, target_class=self.target_class
        )
        return loss

    def _annealed_temperature(self, epoch, total_epochs):
        """τ schedule: exponential decay from temp_start -> temperature.

        τ=0.1 makes Softmax(log(m)/τ) almost one-hot, so its Jacobian w.r.t.
        the logits is tiny (saturated) and the patch barely learns. Starting
        from a softer τ gives a strong early gradient, then we sharpen toward
        the paper's deployment temperature. Returns self.temperature when
        annealing is disabled.
        """
        t0, t1 = self.temp_start, self.temperature
        if not self.anneal_temperature or total_epochs <= 1 or t0 <= t1:
            return t1
        frac = epoch / (total_epochs - 1)
        return float(t0 * (t1 / t0) ** frac)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        env_images,
        train_images=None,
        train_bboxes=None,
        num_iterations=200,
        batch_size=8,
        save_interval=50,
        output_dir='./output',
        fixed_base_colors=None,
        initial_logits=None,
    ):
        os.makedirs(output_dir, exist_ok=True)

        # Step 1-3: get base colors (fixed for paper T1/T2, else K-means), set
        # them, initialise color prob matrix
        if fixed_base_colors is not None:
            base_colors = np.asarray(fixed_base_colors, dtype=np.float32)
            print(f"Using FIXED base colors (skipping K-means):\n{base_colors}")
        else:
            base_colors = self.extract_environment_colors(env_images)
        self.generator.set_base_colors(base_colors)
        self.generator.initialize_color_prob_matrix()
        if initial_logits is not None:
            logits = torch.as_tensor(initial_logits, dtype=torch.float32, device=self.device)
            if tuple(logits.shape) != tuple(self.generator.color_prob_matrix.logits.shape):
                raise ValueError(
                    f'initial_logits shape {tuple(logits.shape)} does not match '
                    f'{tuple(self.generator.color_prob_matrix.logits.shape)}'
                )
            self.generator.color_prob_matrix.logits.data.copy_(logits)

        # Step 4: Adam optimiser on the color probability matrix only
        self.optimizer = optim.Adam(
            self.generator.color_prob_matrix.parameters(),
            lr=self.lr,
        )

        # Step 5: training data
        if train_images is None:
            train_images = env_images
        training_data = self.prepare_training_data(
            train_images, bboxes_per_image=train_bboxes,
        )

        # One epoch = a FULL pass over all training images in mini-batches of
        # `batch_size` (paper §4.2: batch=8, 200 epochs over the 614 INRIA
        # images). The optimizer steps once per mini-batch, so the total number
        # of updates is num_iterations * ceil(N / batch_size) — NOT just
        # num_iterations. (The previous version did a single 8-image batch per
        # epoch => ~200 updates total, which severely under-trained the patch.)
        n_train = len(training_data)
        steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
        print(f"\nStarting training for {num_iterations} epochs "
              f"({steps_per_epoch} steps/epoch, "
              f"{num_iterations * steps_per_epoch} total updates)...")
        print(f"Device: {self.device} | patch={self.patch_size} | K={self.num_base_colors}"
              f" | batch={batch_size} | lr={self.lr} | target_class={self.target_class}"
              f" | input={self.image_size} ({self.resize_mode})"
              f" | tau {self.temp_start}->{self.temperature}"
              f" {'(anneal)' if self.anneal_temperature else '(fixed)'}")

        best_loss = float('inf')

        for epoch in tqdm(range(num_iterations), desc="Training"):
            # τ for this epoch (annealed high->low to avoid gradient saturation)
            cur_temp = self._annealed_temperature(epoch, num_iterations)

            perm = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_seen = 0

            for b in range(0, n_train, batch_size):
                batch_indices = perm[b:b + batch_size]
                self.optimizer.zero_grad()

                for idx in batch_indices:
                    img, positions = training_data[int(idx)]
                    img_tensor = image_to_training_tensor(
                        img,
                        image_size=self.image_size,
                        resize_mode=self.resize_mode,
                    ).to(self.device)

                    # Generate patch at the current (annealed) temperature so
                    # gradients flow to color_prob_matrix.logits.
                    patch = self.generator.generate_patch(temperature=cur_temp)

                    loss = self.compute_attack_loss(img_tensor, patch, positions)
                    # average over this mini-batch, accumulate grads
                    (loss / len(batch_indices)).backward()
                    epoch_loss += loss.item()
                    n_seen += 1

                self.optimizer.step()

            avg_loss = epoch_loss / max(1, n_seen)

            # Render the patch at the DEPLOYMENT temperature (τ=0.1) for saving
            # and logging, so on-disk artifacts match what eval reconstructs.
            with torch.no_grad():
                self.generator.generate_patch(temperature=self.temperature)

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{num_iterations}, "
                           f"Attack Loss: {avg_loss:.4f}, tau={cur_temp:.3f}")

            if self.writer is not None:
                self.writer.add_scalar('Loss/attack', avg_loss, epoch)
                self.writer.add_scalar('Train/temperature', cur_temp, epoch)

            # Save best (lowest detector score) patch
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.generator.save_patch(os.path.join(output_dir, 'best_patch.png'))
                self.generator.save_color_prob_matrix(
                    os.path.join(output_dir, 'best_color_prob.pt')
                )

            # Periodic save
            if (epoch + 1) % save_interval == 0:
                self.generator.save_patch(
                    os.path.join(output_dir, f'patch_epoch_{epoch+1}.png')
                )
                self.generator.save_color_prob_matrix(
                    os.path.join(output_dir, f'color_prob_epoch_{epoch+1}.pt')
                )
                if self.writer is not None:
                    self.writer.add_image(
                        'Patch', self.generator.patch.detach().cpu(), epoch
                    )

        print(f"\nTraining completed. Best attack loss: {best_loss:.4f}")
        self.generator.save_patch(os.path.join(output_dir, 'final_patch.png'))
        self.generator.save_color_prob_matrix(
            os.path.join(output_dir, 'final_color_prob.pt')
        )

        if self.writer is not None:
            self.writer.close()

        return self.generator.patch

    # ------------------------------------------------------------------
    # Fast color transfer (paper §3.3: "fast adversarial patch generation").
    # Re-color an already-trained pattern with base colors from a NEW
    # environment instead of re-optimizing from scratch.
    # ------------------------------------------------------------------
    def fast_color_transfer(self, new_env_images, output_path=None):
        print("\nPerforming fast color transfer...")
        new_base_colors = self.color_extractor.extract_from_images(new_env_images)
        print(f"New base colors:\n{new_base_colors}")
        new_patch = self.generator.transfer_colors(new_base_colors)

        if output_path:
            patch_np = new_patch.detach().cpu().permute(1, 2, 0).numpy()
            patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(patch_np).save(output_path)
            print(f"Transferred patch saved to: {output_path}")
        return new_patch


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    trainer = CAPGenTrainer(
        patch_size=300,
        num_base_colors=3,
        temperature=0.1,
        learning_rate=0.03,
        device=device,
        detector_model='yolov5s',
    )

    env_images = [
        Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    colors = trainer.extract_environment_colors(env_images)
    print(f"Extracted colors: {colors}")
