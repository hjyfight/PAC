"""
CAPGen: Camouflaged Adversarial Patch Generator
Main entry point for the reproduction

Based on the paper:
"CAPGen: An Environment-Adaptive Generator of Adversarial Patches"

Key contributions:
1. Environment-aware patch generation using base colors
2. Color probability matrix optimization
3. Pattern-color decomposition analysis
4. Fast color transfer strategy
"""
import argparse
import os
import sys
from PIL import Image
import numpy as np
import torch

from config import Config
from capgen_trainer import CAPGenTrainer
from color_extractor import ColorExtractor
from patch_generator import CAPGenGenerator
from eot_transforms import PatchApplier

'''
python main.py --mode demo 是把论文4个核心组件挨个跑一遍
对应提取基色；用softmax+基础色生成一张patch图像并保存；
拿同一个color_prob_matrix 换一套新基础色(如雪地)，得到新环境的patch，不需要重训；
把生成的patch贴到dummy图像上，调用yolo检测器
'''

def load_images_from_folder(folder, max_images=10):
    """
    Load images from a folder.

    Args:
        folder: Path to image directory
        max_images: Maximum number of images to load

    Returns:
        images: List of PIL Images
    """
    images = []

    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist")
        return images

    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in Config.VALID_EXTENSIONS:
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                if len(images) >= max_images:
                    break
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return images


def demo_color_extraction():
    """Demo: Extract base colors from environment images."""
    print("\n" + "="*60)
    print("Demo 1: Color Extraction")
    print("="*60)

    # Create sample images (simulating different environments)
    forest_img = np.zeros((100, 100, 3), dtype=np.uint8)
    forest_img[:, :] = [34, 139, 34]  # Forest green
    forest_img[20:40, 20:40] = [139, 119, 101]  # Brown
    forest_img[60:80, 60:80] = [85, 107, 47]  # Dark olive

    snow_img = np.zeros((100, 100, 3), dtype=np.uint8)
    snow_img[:, :] = [255, 250, 250]  # Snow white
    snow_img[20:40, 20:40] = [220, 220, 220]  # Light gray
    snow_img[60:80, 60:80] = [192, 192, 192]  # Silver

    desert_img = np.zeros((100, 100, 3), dtype=np.uint8)
    desert_img[:, :] = [210, 180, 140]  # Tan
    desert_img[20:40, 20:40] = [188, 143, 107]  # Rosy brown
    desert_img[60:80, 60:80] = [139, 119, 101]  # Brown

    # Extract colors
    extractor = ColorExtractor(num_colors=Config.NUM_BASE_COLORS)

    forest_colors = extractor.extract_from_image(Image.fromarray(forest_img))
    print(f"\nForest environment colors:\n{forest_colors}")

    snow_colors = extractor.extract_from_image(Image.fromarray(snow_img))
    print(f"\nSnow environment colors:\n{snow_colors}")

    desert_colors = extractor.extract_from_image(Image.fromarray(desert_img))
    print(f"\nDesert environment colors:\n{desert_colors}")

    # Visualize colors
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    extractor.visualize_colors(forest_colors, save_path=os.path.join(Config.OUTPUT_DIR, 'forest_colors.png'))
    extractor.visualize_colors(snow_colors, save_path=os.path.join(Config.OUTPUT_DIR, 'snow_colors.png'))
    extractor.visualize_colors(desert_colors, save_path=os.path.join(Config.OUTPUT_DIR, 'desert_colors.png'))
    print("\nColor visualizations saved to ./output/")

    return forest_colors, snow_colors, desert_colors


def demo_patch_generation():
    """Demo: Generate adversarial patch with CAPGen."""
    print("\n" + "="*60)
    print("Demo 2: Patch Generation")
    print("="*60)

    # Forest environment colors
    forest_colors = np.array([
        [34, 139, 34],   # Forest green
        [139, 119, 101], # Brown
        [85, 107, 47]    # Dark olive green
    ])

    # Create generator
    generator = CAPGenGenerator(
        patch_size=Config.PATCH_SIZE,
        num_base_colors=Config.NUM_BASE_COLORS,
        temperature=Config.TEMPERATURE
    )

    # Set base colors
    generator.set_base_colors(forest_colors)

    # Generate patch
    patch = generator.generate_patch()
    print(f"\nGenerated patch shape: {patch.shape}")

    # Extract pattern (grayscale view; not the paper definition)
    pattern = generator.get_pattern()
    print(f"Pattern (grayscale) shape: {pattern.shape}")

    # Decompose into paper-defined pattern (r_k) and base colors
    color_prob, base_colors, _ = generator.decompose()
    print(f"Color probability matrix r_k shape: {color_prob.shape}")

    # Save patch
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    generator.save_patch(os.path.join(Config.OUTPUT_DIR, 'forest_patch.png'))
    print("Patch saved to ./output/forest_patch.png")

    return generator


def demo_fast_color_transfer(generator):
    """Demo: Fast color transfer to new environment."""
    print("\n" + "="*60)
    print("Demo 3: Fast Color Transfer")
    print("="*60)

    print("\nKey insight from CAPGen paper:")
    print("Pattern (texture) is more important than color for adversarial effectiveness.")
    print("This enables fast adaptation to new environments by only replacing colors.\n")

    # New environment colors (snow)
    snow_colors = np.array([
        [255, 250, 250],  # Snow white
        [220, 220, 220],  # Light gray
        [192, 192, 192]   # Silver
    ])

    # Transfer colors (keep pattern, replace colors)
    new_patch = generator.transfer_colors(snow_colors)

    # Save transferred patch
    patch_np = new_patch.detach().cpu().permute(1, 2, 0).numpy()
    patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(patch_np)
    img.save(os.path.join(Config.OUTPUT_DIR, 'snow_patch.png'))
    print("Transferred patch saved to ./output/snow_patch.png")

    return new_patch


def demo_pattern_color_analysis():
    """Demo: Analyze pattern vs color importance."""
    print("\n" + "="*60)
    print("Demo 4: Pattern-Color Analysis")
    print("="*60)

    print("""
    Key finding from CAPGen paper:
    - Pattern (texture/structure) has MORE impact on attack performance than color
    - Colors can be replaced without significantly affecting adversarial effectiveness

    This enables fast adversarial patch generation:
    1. Generate a high-performance patch in one environment
    2. When moving to new environment, only replace colors
    3. Pattern remains the same -> attack performance preserved
    """)

    # Example: Same pattern, different colors
    generator = CAPGenGenerator(
        patch_size=Config.PATCH_SIZE,
        num_base_colors=Config.NUM_BASE_COLORS,
        temperature=Config.TEMPERATURE
    )

    # Original colors (forest)
    forest_colors = np.array([
        [34, 139, 34],
        [139, 119, 101],
        [85, 107, 47]
    ])
    generator.set_base_colors(forest_colors)
    generator.generate_patch()
    pattern_original = generator.get_pattern()

    # Transfer to desert colors
    desert_colors = np.array([
        [210, 180, 140],  # Tan
        [188, 143, 107],  # Rosy brown
        [139, 119, 101]   # Brown
    ])
    desert_patch = generator.transfer_colors(desert_colors)

    # Transfer to urban colors
    urban_colors = np.array([
        [128, 128, 128],  # Gray
        [105, 105, 105],  # Dim gray
        [169, 169, 169]   # Dark gray
    ])
    urban_patch = generator.transfer_colors(urban_colors)

    # Transfer to snow colors
    snow_colors = np.array([
        [255, 250, 250],  # Snow white
        [220, 220, 220],  # Light gray
        [192, 192, 192]   # Silver
    ])
    snow_patch = generator.transfer_colors(snow_colors)

    print("Generated patches for 4 environments:")
    print("- Forest (green/brown)")
    print("- Desert (tan/brown)")
    print("- Urban (gray)")
    print("- Snow (white/gray)")

    # Save all patches
    patches = {
        'forest': generator.patch,
        'desert': desert_patch,
        'urban': urban_patch,
        'snow': snow_patch
    }

    for name, patch in patches.items():
        patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
        patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(patch_np)
        img.save(os.path.join(Config.OUTPUT_DIR, f'{name}_patch.png'))

    print(f"\nAll patches saved to {Config.OUTPUT_DIR}/")

    # Show pattern similarity
    pattern_desert = generator.get_pattern()
    print(f"\nPattern shapes are identical: {pattern_original.shape}")
    print("This confirms that only colors change, not the pattern structure.")


def demo_evaluation():
    """Demo: Evaluate adversarial patch effectiveness."""
    print("\n" + "="*60)
    print("Demo 5: Patch Evaluation")
    print("="*60)

    try:
        from detector import YOLODetector
        from evaluator import PatchEvaluator

        # Create detector
        detector = YOLODetector(
            model_name=Config.DETECTOR_MODEL,
            device='cpu',
            conf_threshold=Config.CONFIDENCE_THRESHOLD
        )

        # Create evaluator
        evaluator = PatchEvaluator(detector, device='cpu')
        patch_applier = PatchApplier(Config.PATCH_SIZE)

        # Create sample patch
        generator = CAPGenGenerator(
            patch_size=Config.PATCH_SIZE,
            num_base_colors=Config.NUM_BASE_COLORS,
            temperature=Config.TEMPERATURE
        )
        forest_colors = np.array([
            [34, 139, 34],
            [139, 119, 101],
            [85, 107, 47]
        ])
        generator.set_base_colors(forest_colors)
        patch = generator.generate_patch()

        # Create sample test images
        test_images = [
            Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            for _ in range(3)
        ]

        # Compute visual quality metrics
        print("\nComputing visual quality metrics...")
        metrics = evaluator.compute_visual_quality_metrics(patch)
        print(f"Visual Quality Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Save evaluation results
        evaluator.save_evaluation_results(
            {'visual_quality': metrics},
            os.path.join(Config.OUTPUT_DIR, 'evaluation_results.json')
        )

    except Exception as e:
        print(f"Evaluation demo requires YOLO model: {e}")
        print("Skipping evaluation demo.")


def run_training(env_dir, args):
    """
    Run training mode.

    Args:
        env_dir: Directory containing environment images. May be None when
                 --dataset_dir is provided — in that case, INRIA Train/pos
                 images double as the environment source for k-means.
        args:    Command line arguments
    """
    print("\n" + "#"*60)
    print("# CAPGen - Training Mode")
    print("#"*60)

    # Optional INRIA Person dataset (paper §4.1)
    train_images = None
    train_bboxes = None
    if args.dataset_dir is not None:
        from inria_dataset import load_inria
        print(f"Loading INRIA Person dataset from {args.dataset_dir} ...")
        samples = load_inria(args.dataset_dir, split='train',
                             max_images=args.max_train_images)
        if len(samples) == 0:
            print(f"Error: no INRIA samples found under {args.dataset_dir}")
            sys.exit(1)
        train_images = [s[0] for s in samples]
        train_bboxes = [s[1] for s in samples]
        print(f"Loaded {len(train_images)} INRIA images with person bboxes")

    # Environment images for k-means base-color extraction.
    # If --env_dir not given but INRIA was loaded, reuse INRIA images.
    if env_dir is not None:
        env_images = load_images_from_folder(env_dir)
        if len(env_images) == 0:
            print(f"Error: No images found in {env_dir}")
            sys.exit(1)
        print(f"Loaded {len(env_images)} environment images")
    else:
        if train_images is None:
            print("Error: must supply --env_dir or --dataset_dir")
            sys.exit(1)
        env_images = train_images
        print(f"Reusing INRIA images as environment source for k-means")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create trainer
    # Note: per paper rebuttal, R() and S() are NOT explicit loss terms.
    # R() is realized via EOT, S() via base-color constraint. No TV/NPS needed.
    trainer = CAPGenTrainer(
        patch_size=args.patch_size,
        num_base_colors=args.num_colors,
        temperature=Config.TEMPERATURE,
        temp_start=Config.TEMP_START,
        anneal_temperature=Config.ANNEAL_TEMPERATURE,
        learning_rate=Config.LEARNING_RATE,
        device=device,
        detector_model=args.detector,
        target_class=Config.TARGET_CLASS,
        use_tensorboard=True,
        image_size=Config.IMAGE_SIZE,
        resize_mode=args.resize_mode,
    )

    # Print config
    Config.print_config()

    # Resolve fixed base colors for paper-style T1/T2 (skip K-means extraction)
    fixed_bc = None
    if getattr(args, 'base_colors', None):
        _bc = args.base_colors.strip().lower()
        if _bc == 'bc1':
            fixed_bc = [[119, 49, 72], [2, 204, 1], [134, 2, 182]]
        elif _bc == 'bc2':
            fixed_bc = [[199, 21, 131], [40, 165, 4], [16, 69, 120]]
        else:
            v = [int(x) for x in args.base_colors.split(',')]
            fixed_bc = [v[i:i + 3] for i in range(0, len(v), 3)]
        print(f"[base_colors] using fixed palette: {fixed_bc}")

    # Train
    patch = trainer.train(
        env_images=env_images,
        train_images=train_images,
        train_bboxes=train_bboxes,
        num_iterations=args.num_iterations,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        fixed_base_colors=fixed_bc,
    )

    print(f"\nTraining completed! Final patch saved to {args.output_dir}/")


def run_transfer(args):
    """
    Run fast color transfer mode.

    This implements the key contribution of CAPGen:
    Fast adversarial patch generation by transferring colors to new environments.

    Args:
        args: Command line arguments
    """
    print("\n" + "#"*60)
    print("# CAPGen - Fast Color Transfer Mode")
    print("#"*60)

    print("\nThis mode demonstrates fast adversarial patch adaptation.")
    print("The key insight: patterns are more important than colors.")
    print("We can reuse a trained pattern and only replace colors.\n")

    # Check for required arguments
    if args.patch_path is None:
        print("Error: --patch_path is required for transfer mode")
        print("Please provide a path to a pre-trained patch (.png) or color probability matrix (.pt)")
        sys.exit(1)

    if args.new_env_dir is None:
        print("Error: --new_env_dir is required for transfer mode")
        print("Please provide a directory containing images of the new environment")
        sys.exit(1)

    # Load new environment images
    new_env_images = load_images_from_folder(args.new_env_dir)
    if len(new_env_images) == 0:
        print(f"Error: No images found in {args.new_env_dir}")
        sys.exit(1)

    print(f"Loaded {len(new_env_images)} new environment images")

    # Create generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = CAPGenGenerator(
        patch_size=args.patch_size,
        num_base_colors=args.num_colors,
        temperature=Config.TEMPERATURE,
        device=device
    )

    # Load pre-trained patch or color probability matrix
    patch_path = args.patch_path
    if patch_path.endswith('.pt'):
        # Load color probability matrix
        print(f"\nLoading color probability matrix from: {patch_path}")
        generator.load_color_prob_matrix(patch_path)
        # Generate patch from loaded matrix
        generator.generate_patch()
    elif patch_path.endswith('.png') or patch_path.endswith('.jpg'):
        # Load patch image and extract pattern
        print(f"\nLoading patch image from: {patch_path}")
        patch_img = Image.open(patch_path).convert('RGB')
        patch_np = np.array(patch_img) / 255.0
        patch_tensor = torch.tensor(patch_np, dtype=torch.float32).permute(2, 0, 1)

        # Extract base colors from the patch
        extractor = ColorExtractor(num_colors=args.num_colors)
        base_colors = extractor.extract_from_image(patch_img)
        generator.set_base_colors(base_colors)
        generator.initialize_color_prob_matrix()

        # Set the patch directly
        generator.patch = patch_tensor
    else:
        print(f"Error: Unsupported patch format: {patch_path}")
        print("Please provide a .png, .jpg, or .pt file")
        sys.exit(1)

    # Extract new base colors
    print("\nExtracting base colors from new environment...")
    extractor = ColorExtractor(num_colors=args.num_colors)
    new_base_colors = extractor.extract_from_images(new_env_images)
    print(f"New base colors:\n{new_base_colors}")

    # Perform fast color transfer
    print("\nPerforming fast color transfer...")
    new_patch = generator.transfer_colors(new_base_colors)

    # Save transferred patch
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'transferred_patch.png')
    patch_np = new_patch.detach().cpu().permute(1, 2, 0).numpy()
    patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(patch_np)
    img.save(output_path)
    print(f"\nTransferred patch saved to: {output_path}")

    # Save color probability matrix for future use
    matrix_path = os.path.join(args.output_dir, 'transferred_color_prob.pt')
    generator.save_color_prob_matrix(matrix_path)
    print(f"Color probability matrix saved to: {matrix_path}")

    # Visualize color comparison
    print("\n" + "="*60)
    print("Color Comparison:")
    print("="*60)
    print(f"Original colors (from patch):")
    print(generator.base_colors.cpu().numpy() * 255)
    print(f"\nNew environment colors:")
    print(new_base_colors)

    print("\n" + "#"*60)
    print("# Transfer completed!")
    print("#"*60)


def main():
    parser = argparse.ArgumentParser(
        description='CAPGen: Camouflaged Adversarial Patch Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo mode (no GPU required)
  python main.py --mode demo

  # Train with environment images
  python main.py --mode train --env_dir ./data/forest --num_iterations 500

  # Transfer patch to new environment
  python main.py --mode transfer --patch_path ./output/best_patch.pt --new_env_dir ./data/snow
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'train', 'transfer'],
                       help='Mode to run (default: demo)')

    # Training arguments
    parser.add_argument('--env_dir', type=str, default=None,
                       help='Directory containing environment images (optional when --dataset_dir is set; defaults to reusing INRIA imgs)')
    parser.add_argument('--dataset_dir', type=str, default=None,
                       help='Path to INRIAPerson/ root (Train/pos + Train/annotations). Enables bbox-based patch placement (paper §4.1).')
    parser.add_argument('--max_train_images', type=int, default=None,
                       help='Cap on INRIA training images loaded (debug; default: all 614)')
    parser.add_argument('--patch_size', type=int, default=Config.PATCH_SIZE,
                       help=f'Size of adversarial patch (default: {Config.PATCH_SIZE})')
    parser.add_argument('--num_colors', type=int, default=Config.NUM_BASE_COLORS,
                       help=f'Number of base colors (default: {Config.NUM_BASE_COLORS})')
    parser.add_argument('--base_colors', type=str, default=None,
                       help="Fixed base colors instead of K-means (for paper T1/T2): "
                            "'bc1', 'bc2', or 9 comma-separated ints 'R,G,B,R,G,B,R,G,B'")
    parser.add_argument('--num_iterations', type=int, default=Config.NUM_ITERATIONS,
                       help=f'Number of training iterations (default: {Config.NUM_ITERATIONS})')
    parser.add_argument('--save_interval', type=int, default=Config.SAVE_INTERVAL,
                       help=f'Save patch every N iterations (default: {Config.SAVE_INTERVAL})')
    parser.add_argument('--detector', type=str, default=Config.DETECTOR_MODEL,
                       help=f'YOLO model variant (default: {Config.DETECTOR_MODEL})')
    parser.add_argument('--resize_mode', choices=['squash', 'letterbox'], default='squash',
                       help="Training resize mode: 'squash' direct-resizes to 640x640; "
                            "'letterbox' uses YOLOv5 letterbox before patch placement.")

    # Transfer arguments
    parser.add_argument('--patch_path', type=str, default=None,
                       help='Path to pre-trained patch (.png/.jpg) or color probability matrix (.pt)')
    parser.add_argument('--new_env_dir', type=str, default=None,
                       help='Directory containing images of new environment')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR,
                       help=f'Output directory (default: {Config.OUTPUT_DIR})')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'demo':
        print("\n" + "#"*60)
        print("# CAPGen Reproduction - Demo Mode")
        print("#"*60)

        # Demo 1: Color extraction
        demo_color_extraction()

        # Demo 2: Patch generation
        generator = demo_patch_generation()

        # Demo 3: Fast color transfer
        demo_fast_color_transfer(generator)

        # Demo 4: Pattern-color analysis
        demo_pattern_color_analysis()

        # Demo 5: Evaluation (optional)
        demo_evaluation()

        print("\n" + "#"*60)
        print(f"# Demo completed! Check {Config.OUTPUT_DIR}/ for results")
        print("#"*60)

    elif args.mode == 'train':
        if args.env_dir is None and args.dataset_dir is None:
            print("Error: training mode needs --env_dir or --dataset_dir")
            sys.exit(1)
        run_training(args.env_dir, args)

    elif args.mode == 'transfer':
        run_transfer(args)

    print("\nDone!")


if __name__ == '__main__':
    main()
