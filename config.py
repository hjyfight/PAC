"""
CAPGen Configuration File

Central configuration for all CAPGen modules.
Based on the paper: "CAPGen: An Environment-Adaptive Generator of Adversarial Patches"

All hyperparameters are from the paper and its supplementary material.
"""
import torch
import os


class Config:
    """
    Configuration class for CAPGen.

    All hyperparameters match the paper's specifications.
    """

    # ==================== Patch Settings ====================
    # Internal generation resolution of the color-probability matrix (W = H).
    # NOTE: the paper does NOT specify an absolute pixel size. It only defines
    # the *applied* patch size as 25% of the target's bounding-box AREA
    # (Sec 4.2 / Sec 4.5). 300 is simply our chosen generation resolution; the
    # patch is resized to the bbox-relative size when pasted onto objects.
    PATCH_SIZE = 300

    # ==================== Color Extraction Settings ====================
    # Paper: "utilize K-Means algorithm to cluster the pixel points in the image"
    # "setting K = 3 serves as a solid baseline"
    NUM_BASE_COLORS = 3
    KMEANS_MAX_ITER = 100
    KMEANS_N_INIT = 10
    MAX_PIXELS_PER_IMAGE = 10000

    # ==================== Color Probability Matrix Settings ====================
    # Paper Eq.(3): "r_k = Softmax(log(m_ijk) / τ)"
    # "τ = 0.1 serves as the optimal choice". Keep it fixed during training,
    # saving, and evaluation for strict paper-style CAPGen-T/R reproduction.
    TEMPERATURE = 0.1
    TEMP_START = 0.1
    ANNEAL_TEMPERATURE = False

    # ==================== Training Settings ====================
    # Paper Table 1: "Batch size = 8, Learning rate = 0.03, Epochs = 200"
    LEARNING_RATE = 0.03
    NUM_ITERATIONS = 200  # Epochs
    BATCH_SIZE = 8
    SAVE_INTERVAL = 50

    # ==================== EOT Settings ====================
    # Paper Appendix D:
    # "Smoothing   m' = m * s"            (Eq.6, uniform kernel; reduces printing error)
    # "Contrast    D ~ Uniform(0.8, 1.2)"
    # "Brightness  B ~ Uniform(0.9, 1.1)" (paper: additive on m; we apply multiplicative
    #                                      on the rendered patch -- additive on [0,1] RGB
    #                                      would saturate. Disclosed deviation.)
    # "Noise       N ~ Uniform(0, 0.1)"   (additive symmetric-uniform per-pixel noise)
    # "Rotation    θ ~ Uniform(-20, 20) degrees"
    # "Scale       S ~ Uniform(0.9, 1.1)"
    EOT_ENABLED = True
    ROTATION_RANGE = (-20, 20)
    BRIGHTNESS_RANGE = (0.9, 1.1)
    CONTRAST_RANGE = (0.8, 1.2)
    NOISE_RANGE = (0.0, 0.1)
    SCALE_RANGE = (0.9, 1.1)
    SMOOTH_KERNEL_SIZE = 3  # Eq.6 uniform smoothing kernel (odd; <=1 disables)

    # ==================== Device Settings ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== Path Settings ====================
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output"
    MODEL_PATH = "./models"
    RUNS_DIR = "./runs"

    # ==================== Attack Settings ====================
    # Paper: "we use Yolov5s as the default substitute model"
    # Paper Sec 4.4: target object is a person wearing CAPGen coat
    # COCO class 0 = 'person'
    CONFIDENCE_THRESHOLD = 0.3
    TARGET_CLASS = 0  # Untargeted detection-evasion on person class
    DETECTOR_MODEL = "yolov5s"

    # ==================== Patch Position Settings ====================
    # Paper Supplementary: patch placement on objects
    SCALE_MIN = 0.3
    SCALE_MAX = 0.6

    # ==================== Image Settings ====================
    # Paper Table 1: "Input size = 640"
    IMAGE_SIZE = (640, 640)
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # ==================== Loss Settings ====================
    # Paper Rebuttal: R(P;φ) and S(P; ε) are NOT explicit loss terms
    # R() is achieved through EOT framework
    # S() is achieved through color constraint design
    # The optimization only minimizes the detection loss L

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

    @classmethod
    def print_config(cls):
        """Print all configuration values."""
        print("\n" + "="*60)
        print("CAPGen Configuration (from paper)")
        print("="*60)
        for key, value in cls.to_dict().items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
