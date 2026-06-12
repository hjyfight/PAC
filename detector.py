"""
Object Detection Model Wrapper
Provides interface to YOLOv5 detector for adversarial patch optimization.

Per paper rebuttal §4 (Reviewer zes9 reply):
    "The loss is backpropagated through the entire differentiable pipeline
     (detector, EOT, patch generation) all the way back to the color
     probability matrix m."

Therefore we expose two interfaces:
- `YOLODetector.detect(...)`: post-processed (numpy) detections for evaluation.
- `YOLODetector.forward_raw(image_tensor)`: returns the raw (B, N, 5+C) tensor
  before NMS, which keeps gradients and is the entry point used by the trainer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms 

'''
检测器接口与梯度链路
确认 YOLOv5 包装器 YOLODetector 这个组件本身是好的、可微的、能反传梯度
'''
class YOLODetector(nn.Module):
    """
    YOLOv5 wrapper that exposes a differentiable forward pass.

    Loads the underlying nn.Module from torch.hub (ultralytics/yolov5) so that
    we can run the model directly on a tensor and obtain raw predictions.
    """

    def __init__(self, model_name='yolov5s', device='cpu', conf_threshold=0.3):
        super().__init__()
        self.device = device
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """
        Load YOLOv5 model and extract the inner nn.Module for differentiable
        forward passes.

        torch.hub.load returns an `AutoShape` wrapper that performs PIL/numpy
        preprocessing (letterbox + normalisation) and post-processing (NMS).
        For differentiable training we need to call the underlying
        `DetectMultiBackend` directly on a pre-normalised tensor.
        """
        import os
        hub_dir = os.path.expanduser('~/.cache/torch/hub/ultralytics_yolov5_master')
        local_weights = os.path.join(hub_dir, f'{self.model_name}.pt')
        if os.path.isdir(hub_dir) and os.path.isfile(local_weights):
            hub_model = torch.hub.load(
                hub_dir, 'custom', path=local_weights,
                source='local', verbose=False,
            )
        else:
            try:
                hub_model = torch.hub.load(
                    'ultralytics/yolov5', self.model_name,
                    pretrained=True, trust_repo=True, verbose=False,
                )
            except TypeError:
                hub_model = torch.hub.load(
                    'ultralytics/yolov5', self.model_name, pretrained=True
                )
        hub_model.conf = self.conf_threshold

        # Keep the AutoShape wrapper for high-level (non-differentiable) detect()
        self.hub_model = hub_model

        # AutoShape.model -> DetectMultiBackend, which is itself an nn.Module
        # and accepts a (B, 3, H, W) float tensor in [0, 1].
        raw_model = getattr(hub_model, 'model', hub_model)
        self.raw_model = raw_model.to(self.device)
        self.raw_model.eval()

        # Freeze detector weights (only the patch / color_prob_matrix is updated)
        for p in self.raw_model.parameters():
            p.requires_grad_(False)

        print(f"Loaded {self.model_name} (differentiable interface enabled)")

    # ----- Differentiable interface (used by trainer) -----
    def forward_raw(self, image_tensor):
        """
        Run the YOLOv5 forward pass on a batched tensor and return raw
        predictions BEFORE NMS.

        Args:
            image_tensor: (B, 3, H, W) float tensor in [0, 1]

        Returns:
            pred: (B, N, 5 + num_classes) tensor where last dim is
                  [x, y, w, h, objectness, cls_0, cls_1, ...]
                  (sigmoid already applied by YOLOv5 head).
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        out = self.raw_model(image_tensor)
        # YOLOv5 returns (pred, aux) in train mode and pred in eval mode.
        if isinstance(out, (tuple, list)):
            pred = out[0]
        else:
            pred = out
        return pred

    def objectness_attack_loss(self, image_tensor, target_class=0):
        """
        Compute the objectness-based attack loss used in the paper.

        Following common YOLO patch-attack practice (used by AdvPatch and CAPGen
        baseline), we minimize the maximum (objectness * class_prob) score for
        the target class across all anchors. Minimizing this score forces the
        detector to drop the detection.

        Args:
            image_tensor: (B, 3, H, W) in [0, 1]
            target_class: COCO class index to suppress (0 = person)

        Returns:
            Scalar attack loss (differentiable wrt image_tensor)
        """
        pred = self.forward_raw(image_tensor)        # (B, N, 5+C)
        obj = pred[..., 4]                            # (B, N) objectness
        cls = pred[..., 5 + target_class]             # (B, N) target-class prob
        score = obj * cls                             # (B, N)

        # Use a soft-max approximation for stable gradient on "max over anchors"
        # max ≈ logsumexp / temperature
        # but a simple max also has subgradient; we take per-image max then mean.
        per_image_max, _ = score.max(dim=1)           # (B,)
        loss = per_image_max.mean()
        return loss

    # ----- High-level (non-differentiable) detection for evaluation -----
    def detect(self, image):
        """
        Run object detection on image (returns post-processed results).

        Args:
            image: PIL Image / numpy array / torch tensor (C,H,W) or (1,C,H,W)

        Returns:
            detections: list of dicts {'bbox', 'class', 'confidence'}
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image.detach().cpu().permute(1, 2, 0).numpy()
            image = (image * 255).clip(0, 255).astype(np.uint8)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        results = self.hub_model(image)
        detections = []
        pred = results.xyxy[0].cpu().numpy()
        for det in pred:
            detections.append({
                'bbox': det[:4],
                'class': int(det[5]),
                'confidence': float(det[4])
            })
        return detections

    def get_confidence_scores(self, image, target_class=None):
        """
        Get confidence scores for detections (post-processed).
        """
        detections = self.detect(image)
        if target_class is not None:
            return [d['confidence'] for d in detections if d['class'] == target_class]
        return [d['confidence'] for d in detections]


class DetectionLoss:
    """
    Legacy non-differentiable loss kept for backward compatibility / evaluation.
    Training should use `YOLODetector.objectness_attack_loss` directly.
    """

    def __init__(self, target_class=None, loss_type='obj'):
        self.target_class = target_class
        self.loss_type = loss_type

    def compute_loss(self, conf_scores, detections=None):
        if len(conf_scores) == 0:
            return torch.tensor(0.0)
        conf_tensor = torch.tensor(conf_scores)
        return torch.mean(conf_tensor)


if __name__ == "__main__":
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector = YOLODetector(model_name='yolov5s', device=device)

        # Test raw differentiable forward
        x = torch.rand(1, 3, 640, 640, device=device, requires_grad=True)
        loss = detector.objectness_attack_loss(x, target_class=0)
        loss.backward()
        print(f"Raw forward OK. Loss={loss.item():.4f}, x.grad norm={x.grad.norm().item():.4f}")

        # Test post-processed detect
        sample_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = detector.detect(Image.fromarray(sample_image))
        print(f"Number of detections: {len(detections)}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please install ultralytics / yolov5 dependencies.")
