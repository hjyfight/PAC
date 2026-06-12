"""
CAPGen Evaluator Module
Provides quantitative evaluation metrics for adversarial patches.

Metrics:
- Attack Success Rate (ASR): Percentage of images where detection is evaded
- Mean Average Precision (mAP): Detection accuracy before/after attack
- Confidence Reduction: Average reduction in detection confidence
- Pattern-Color Decomposition Analysis
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import json


class PatchEvaluator:
    """
    Evaluator for adversarial patch effectiveness.

    Computes various metrics to quantify attack performance.
    """

    def __init__(self, detector, device='cpu', conf_threshold=0.5):
        """
        Args:
            detector: YOLODetector instance
            device: Device to use
            conf_threshold: Confidence threshold for detection
        """
        self.detector = detector
        self.device = device
        self.conf_threshold = conf_threshold
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def compute_asr(self, images, patch, patch_applier, eot=None, num_trials=1):
        """
        Compute Attack Success Rate (ASR).

        ASR = (Number of images where detection fails) / (Total images)

        Args:
            images: List of PIL Images
            patch: Adversarial patch tensor (C, H, W)
            patch_applier: PatchApplier instance
            eot: Optional EOT transform for robustness testing
            num_trials: Number of trials per image with random positions

        Returns:
            asr: Attack success rate [0, 1]
            details: Dict with detailed results
        """
        total_trials = 0
        successful_attacks = 0
        confidence_reductions = []

        print(f"Computing ASR over {len(images)} images, {num_trials} trials each...")

        for img in tqdm(images, desc="Evaluating ASR"):
            # Get baseline detections (without patch)
            baseline_scores = self.detector.get_confidence_scores(img)
            has_baseline_detection = len(baseline_scores) > 0

            for trial in range(num_trials):
                # Convert image to tensor
                img_tensor = self.transform(img).to(self.device)

                # Random patch position
                _, h, w = img_tensor.shape
                _, ph, pw = patch.shape
                scale = np.random.uniform(0.3, 0.6)
                scaled_h = int(ph * scale)
                scaled_w = int(pw * scale)
                x = np.random.randint(0, max(1, w - scaled_w))
                y = np.random.randint(0, max(1, h - scaled_h))

                # Apply EOT if provided
                if eot is not None:
                    patch_transformed = eot(patch.unsqueeze(0)).squeeze(0)
                else:
                    patch_transformed = patch

                # Apply patch to image
                img_with_patch = patch_applier.apply_patch(
                    img_tensor, patch_transformed, x, y, scale
                )

                # Convert to PIL for detection
                img_np = img_with_patch.cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                # Get detection scores with patch
                patched_scores = self.detector.get_confidence_scores(pil_img)
                has_patched_detection = len(patched_scores) > 0

                total_trials += 1

                # Attack successful if:
                # 1. Image had detections before but not after, OR
                # 2. Confidence significantly reduced
                if has_baseline_detection and not has_patched_detection:
                    successful_attacks += 1
                elif has_baseline_detection and has_patched_detection:
                    # Check confidence reduction
                    baseline_max = max(baseline_scores)
                    patched_max = max(patched_scores)
                    reduction = (baseline_max - patched_max) / baseline_max
                    confidence_reductions.append(reduction)

                    if reduction > 0.5:  # 50% confidence reduction threshold
                        successful_attacks += 1

        asr = successful_attacks / total_trials if total_trials > 0 else 0.0
        avg_confidence_reduction = np.mean(confidence_reductions) if confidence_reductions else 0.0

        details = {
            'total_trials': total_trials,
            'successful_attacks': successful_attacks,
            'asr': asr,
            'avg_confidence_reduction': avg_confidence_reduction,
            'num_images': len(images)
        }

        return asr, details

    def evaluate_patch_transferability(self, original_patch, transferred_patch, images, patch_applier):
        """
        Evaluate how well the transferred patch maintains attack performance.

        This tests the key finding: patterns are more important than colors.

        Args:
            original_patch: Original adversarial patch
            transferred_patch: Patch with transferred colors
            images: List of test images
            patch_applier: PatchApplier instance

        Returns:
            results: Dict with comparison metrics
        """
        print("Evaluating patch transferability...")

        # Evaluate original patch
        asr_original, details_original = self.compute_asr(
            images, original_patch, patch_applier, num_trials=1
        )

        # Evaluate transferred patch
        asr_transferred, details_transferred = self.compute_asr(
            images, transferred_patch, patch_applier, num_trials=1
        )

        results = {
            'original_asr': asr_original,
            'transferred_asr': asr_transferred,
            'asr_difference': abs(asr_original - asr_transferred),
            'original_details': details_original,
            'transferred_details': details_transferred,
            'transferability_preserved': abs(asr_original - asr_transferred) < 0.1
        }

        print(f"\nTransferability Results:")
        print(f"  Original ASR: {asr_original:.4f}")
        print(f"  Transferred ASR: {asr_transferred:.4f}")
        print(f"  ASR Difference: {abs(asr_original - asr_transferred):.4f}")
        print(f"  Transferability Preserved: {results['transferability_preserved']}")

        return results

    def analyze_pattern_color_importance(self, generator, images, patch_applier,
                                          color_sets=None):
        """
        Validate the paper's key claim: pattern matters more than color.

        Per paper rebuttal §6 (Reviewer 7Kmv reply), the experiment design is:
        take ONE high-performance adversarial *pattern* (i.e. a trained
        color_prob_matrix), then apply MULTIPLE different sets of base colors
        to it, and verify that attack performance (ASR) stays consistently high
        across all color sets.

        This function requires `generator` to already have a trained
        `color_prob_matrix` (the pattern). It freezes the pattern, sweeps the
        provided `color_sets`, and measures ASR for each.

        Args:
            generator: CAPGenGenerator with an ALREADY-TRAINED color_prob_matrix
            images: list of test PIL Images
            patch_applier: PatchApplier instance
            color_sets: list of (K, 3) numpy arrays of RGB values in [0, 255].
                        If None, a default set of {forest, snow, desert, urban}
                        palettes is used.

        Returns:
            analysis: dict with per-color-set ASR and the variance across sets.
        """
        if generator.color_prob_matrix is None:
            raise ValueError(
                "Generator has no trained pattern. Train or load a "
                "color_prob_matrix before running this analysis."
            )

        if color_sets is None:
            color_sets = {
                'forest': np.array([[34, 139, 34], [139, 119, 101], [85, 107, 47]],
                                   dtype=np.float32),
                'snow':   np.array([[255, 250, 250], [220, 220, 220], [192, 192, 192]],
                                   dtype=np.float32),
                'desert': np.array([[210, 180, 140], [188, 143, 107], [139, 119, 101]],
                                   dtype=np.float32),
                'urban':  np.array([[128, 128, 128], [105, 105, 105], [169, 169, 169]],
                                   dtype=np.float32),
            }
        elif isinstance(color_sets, list):
            color_sets = {f'set_{i}': c for i, c in enumerate(color_sets)}

        print("\nAnalysing pattern vs color importance "
              "(SAME pattern, DIFFERENT colors)...")

        # Snapshot the trained logits — we will NOT touch them
        original_logits = generator.color_prob_matrix.logits.data.clone()
        original_base_colors = (
            generator.base_colors.clone() if generator.base_colors is not None else None
        )

        results = {}
        for name, colors in color_sets.items():
            print(f"\n>>> Evaluating with '{name}' colors:\n{colors}")
            # transfer_colors keeps logits intact and just renders with new colors
            patch = generator.transfer_colors(colors)
            asr, details = self.compute_asr(images, patch, patch_applier, num_trials=1)
            results[name] = {'asr': asr, 'colors': colors.tolist(), 'details': details}
            print(f"    -> ASR = {asr:.4f}")

        # Sanity-restore original colors (logits were never modified)
        if original_base_colors is not None:
            generator.base_colors = original_base_colors
        generator.color_prob_matrix.logits.data = original_logits

        asrs = [r['asr'] for r in results.values()]
        analysis = {
            'per_color_set': results,
            'asr_mean': float(np.mean(asrs)),
            'asr_std':  float(np.std(asrs)),
            'observation': (
                "Low std across color sets indicates that the pattern, not the "
                "colors, dominates attack effectiveness — confirming the paper's "
                "main empirical claim."
            ),
        }
        print(f"\nASR mean = {analysis['asr_mean']:.4f}, "
              f"std = {analysis['asr_std']:.4f}")
        return analysis

    def compute_visual_quality_metrics(self, patch, environment_image=None):
        """
        Compute visual quality metrics for the patch.

        Args:
            patch: Adversarial patch tensor (C, H, W)
            environment_image: Optional environment image to compare against

        Returns:
            metrics: Dict with quality metrics
        """
        patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()

        metrics = {}

        # Color statistics
        metrics['mean_color'] = patch_np.mean(axis=(0, 1)).tolist()
        metrics['std_color'] = patch_np.std(axis=(0, 1)).tolist()

        # Contrast (standard deviation of grayscale)
        gray = 0.299 * patch_np[:, :, 0] + 0.587 * patch_np[:, :, 1] + 0.114 * patch_np[:, :, 2]
        metrics['contrast'] = float(gray.std())

        # Total variation (smoothness)
        tv_h = np.mean(np.abs(patch_np[1:, :, :] - patch_np[:-1, :, :]))
        tv_w = np.mean(np.abs(patch_np[:, 1:, :] - patch_np[:, :-1, :]))
        metrics['total_variation'] = float(tv_h + tv_w)

        # Color diversity (unique colors in quantized patch)
        quantized = (patch_np * 10).astype(int)
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        metrics['unique_colors'] = unique_colors

        # Compare with environment if provided
        if environment_image is not None:
            if isinstance(environment_image, Image.Image):
                env_np = np.array(environment_image) / 255.0
            else:
                env_np = environment_image

            # Color similarity (mean squared error)
            patch_resized = np.array(Image.fromarray((patch_np * 255).astype(np.uint8)).resize(
                (env_np.shape[1], env_np.shape[0])
            )) / 255.0
            mse = np.mean((patch_resized - env_np) ** 2)
            metrics['env_color_mse'] = float(mse)

            # Lower MSE means better camouflage
            metrics['camouflage_score'] = float(1.0 - mse)

        return metrics

    def save_evaluation_results(self, results, output_path):
        """
        Save evaluation results to JSON file.

        Args:
            results: Dict with evaluation results
            output_path: Path to save JSON file
        """
        # Convert numpy/torch types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return obj

        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    # Test evaluator
    from detector import YOLODetector
    from eot_transforms import PatchApplier
    from patch_generator import CAPGenGenerator

    print("Testing PatchEvaluator...")

    # Create components
    detector = YOLODetector(model_name='yolov5s', device='cpu')
    evaluator = PatchEvaluator(detector, device='cpu')
    patch_applier = PatchApplier(300)

    # Create sample patch
    generator = CAPGenGenerator(patch_size=300, num_base_colors=3)
    base_colors = np.array([
        [34, 139, 34],
        [139, 119, 101],
        [85, 107, 47]
    ])
    generator.set_base_colors(base_colors)
    patch = generator.generate_patch()

    # Create sample images
    sample_images = [
        Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        for _ in range(3)
    ]

    # Test ASR computation
    asr, details = evaluator.compute_asr(sample_images, patch, patch_applier, num_trials=1)
    print(f"\nASR: {asr:.4f}")
    print(f"Details: {details}")

    # Test visual quality metrics
    metrics = evaluator.compute_visual_quality_metrics(patch)
    print(f"\nVisual Quality Metrics: {metrics}")

    print("\nEvaluator tests completed!")
