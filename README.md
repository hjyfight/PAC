# CAPGen Reproduction

## Paper
**CAPGen: An Environment-Adaptive Generator of Adversarial Patches**

This is a reproduction of the CAPGen method for generating camouflaged adversarial patches that blend with the environment.

## Key Contributions

1. **Environment-Aware Patch Generation**: Extracts base colors from environment using K-means clustering
2. **Color Probability Matrix**: Optimizes a matrix to determine pixel-color assignments
3. **Pattern-Color Decomposition**: Discovers that patterns (texture) matter more than colors for attack effectiveness
4. **Fast Color Transfer**: Enables quick adaptation to new environments by only replacing colors

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Demo Mode (No GPU required)
```bash
python main.py --mode demo
```

This will demonstrate:
- Color extraction from environment images
- Patch generation with CAPGen
- Fast color transfer to new environments
- Pattern vs color analysis

### Training Mode (Requires GPU and YOLOv5)
```bash
python main.py --mode train --env_dir /path/to/environment/images --output_dir ./output
```

## Project Structure

```
CAPGen_Reproduction/
├── main.py              # Main entry point
├── config.py            # Configuration parameters
├── color_extractor.py   # K-means color extraction
├── patch_generator.py   # Core CAPGen generator
├── eot_transforms.py    # EOT (Expectation Over Transformation)
├── detector.py          # YOLO detector wrapper
├── capgen_trainer.py    # Training loop
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Core Algorithm

### 1. Base Color Extraction
Using K-means clustering to extract dominant colors from environment:
```python
extractor = ColorExtractor(num_colors=3)
base_colors = extractor.extract_from_images(env_images)
```

### 2. Patch Generation
Generate patch using color probability matrix:
```python
generator = CAPGenGenerator(patch_size=300)
generator.set_base_colors(base_colors)
patch = generator.generate_patch()
```

### 3. Fast Color Transfer
Transfer patch to new environment by replacing colors:
```python
new_patch = generator.transfer_colors(new_env_colors)
```

## Key Findings from Paper

| Component | Impact on Attack Performance |
|-----------|------------------------------|
| Pattern (texture) | HIGH |
| Color | LOW |

This enables fast adversarial patch generation:
1. Generate high-performance patch once
2. For new environments, only replace colors
3. Attack performance is preserved

## Output

The generated patches will be saved in the `./output/` directory:
- `forest_patch.png` - Patch with forest colors
- `snow_patch.png` - Patch with snow colors
- `desert_patch.png` - Patch with desert colors
- `urban_patch.png` - Patch with urban colors

## References

- Original paper: CAPGen: An Environment-Adaptive Generator of Adversarial Patches
- YOLO: https://github.com/ultralytics/yolov5
