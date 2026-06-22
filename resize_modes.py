"""Training resize helpers for squash vs YOLOv5 letterbox modes."""
import os
import sys

import numpy as np
import torch
from PIL import Image


YOLOV5_DIR = os.path.join(os.path.dirname(__file__), "yolov5")
if YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)

try:
    from utils.augmentations import letterbox as yolov5_letterbox
except Exception:  # pragma: no cover - letterbox mode reports a clear error.
    yolov5_letterbox = None


def image_to_training_tensor(img_pil, image_size=(640, 640), resize_mode="squash"):
    """Convert a PIL image to a CHW float tensor using the selected resize mode."""
    w, h = image_size
    if resize_mode == "squash":
        arr = np.asarray(img_pil.resize((w, h)).convert("RGB"), dtype=np.float32)
    elif resize_mode == "letterbox":
        if yolov5_letterbox is None:
            raise RuntimeError("Could not import yolov5.utils.augmentations.letterbox")
        arr = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
        arr, _, _ = yolov5_letterbox(
            arr,
            new_shape=(h, w),
            auto=False,
            scaleFill=False,
            scaleup=False,
            stride=32,
        )
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")

    return torch.from_numpy(arr / 255.0).permute(2, 0, 1).contiguous()


def transform_bboxes(img_pil, bboxes, image_size=(640, 640), resize_mode="squash"):
    """Map original INRIA xyxy boxes into the resized training image frame."""
    w, h = image_size
    iw, ih = img_pil.size
    transformed = []

    if resize_mode == "squash":
        sx, sy = w / iw, h / ih
        for x1, y1, x2, y2 in bboxes:
            transformed.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy))
        return transformed

    if resize_mode == "letterbox":
        r = min(h / ih, w / iw)
        r = min(r, 1.0)
        new_unpad = (int(round(iw * r)), int(round(ih * r)))
        dw = (w - new_unpad[0]) / 2.0
        dh = (h - new_unpad[1]) / 2.0
        for x1, y1, x2, y2 in bboxes:
            transformed.append((x1 * r + dw, y1 * r + dh,
                                x2 * r + dw, y2 * r + dh))
        return transformed

    raise ValueError(f"Unknown resize_mode: {resize_mode}")


def build_bbox_patch_positions(img_pil, bboxes, patch_size,
                               image_size=(640, 640), resize_mode="squash"):
    """Build one centered patch placement per bbox in resized-image coords."""
    w, h = image_size
    positions = []
    for bx1, by1, bx2, by2 in transform_bboxes(
        img_pil, bboxes, image_size=image_size, resize_mode=resize_mode,
    ):
        bw, bh = bx2 - bx1, by2 - by1
        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        placed = max(1.0, (0.25 * bw * bh) ** 0.5)
        scale = float(max(1e-3, placed / patch_size))
        x = int(round(bcx - placed * 0.5))
        y = int(round(bcy - placed * 0.5))
        x = max(0, min(w - int(placed), x))
        y = max(0, min(h - int(placed), y))
        positions.append((x, y, scale))
    return positions
