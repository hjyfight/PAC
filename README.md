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



方法	            是否训练	        入口文件	                                    说明
AdvPatch	        是	        advpatch_trainer.py	                                自由 RGB 像素补丁
CAPGen-T / T1 / T2	是	        main.py --mode train，内部调用 capgen_trainer.py	训练颜色概率矩阵
CAPGen-P / P1 / P2	不是重新训练	make_capgen_p.py	                        用已有 pattern 换 Bc1/Bc2
CAPGen-R1 / R2	    不是训练	    make_capgen_p.py	                          随机 pattern + Bc1/Bc2



目前主要问题集中在 CAPGen-P 这个实验。

复现现状大概是：

方法	        论文 YOLOv5s mAP50	         当前较新结果 mAP50	        差距
AdvPatch	        31.6	                        31.75	         基本复现成功
CAPGen-T1/T2	    71.5 / 71.7	                45.61 / 47.07	    比论文更强，方向对但数值不一致
CAPGen-P1/P2	    32.7 / 36.8	                47.85 / 51.39	    明显偏弱，未复现成功
CAPGen-R1/R2	    86.3 / 85.4	                43.00 / 41.92	    mAP 失真，det_rate 仍很高，实际无效
最大差距是：论文里 CAPGen-P1/P2 ≈ AdvPatch，说明“换色后仍保留 AdvPatch 强攻击 pattern”；但当前结果里 P1/P2 明显弱于 AdvPatch。尤其 det_rate 仍然很高，说明人多数还能被检测出来。

代码是否按论文写：大部分训练框架是按论文描述写的，但 CAPGen-P 这一步存在论文描述不清导致的实现偏差。

已经基本按论文实现的部分：

INRIA Person，输入 resize 到 640x640
patch 放在 person bbox 上，面积约为 bbox 的 25%
batch size 8，Adam lr 0.03，epoch 200
K=3 base colors
Eq.(3)：r_k = Softmax(log(m_ijk) / tau)，tau=0.1
EOT：亮度、对比度、噪声、旋转、缩放
CAPGen-T/T1/T2：训练 color probability matrix
CAPGen-R：随机 color probability matrix
AdvPatch：自由像素补丁训练
没有完全解决的部分：

论文说 CAPGen-P1/P2 是“先用 AdvPatch 生成强补丁，再把 AdvPatch 的颜色替换成 Bc1/Bc2”。但论文没有给出 raw AdvPatch 如何精确分解成 pattern + colors / r_k 的代码级算法。

当前代码尝试过两类做法：

从 raw AdvPatch 做 K-means / soft decomposition，再换 Bc1/Bc2
问题：P0 都不能很好重构 raw AdvPatch，pattern 已经损坏，所以 P1/P2 必然弱。

用训练好的 CAPGen-T matrix 作为 source pattern，再按 Eq.(4) 换色
优点：更符合 Eq.(4) 的“固定 r_k，只换 base colors”；P0 能和 source patch 对齐。
问题：严格说它不是论文原文里的“AdvPatch pattern + Bc1/Bc2”，因为 source pattern 来自 CAPGen-T，而不是 raw AdvPatch。