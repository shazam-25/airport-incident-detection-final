import os 
import cv2
import time
import torch
import numpy as np
from torchvision.ops import nms
from typing import Dict, List, Tuple, Optional, Union

from src.models.config import MultiTaskModelConfig
from src.models.network import MultiTaskYOLO

class AirportStreamProcessor:
    """Multi-stream video inference engine for Airport Incident Detection.
    Processes video feeds for 'turnaround', 'ppe', and 'fod' camera positions simultaneously."""

    # Distinct color palette (BGR) for stream categories & tasks
    STREAM_COLORS = {
        "turnaround": (255, 165, 0),    # Bright Orange
        "ppe": (0, 255, 255),           # Yellow / Cyan
        "fod": (0, 0 , 255)             # Bright Yellow
    }

    # CLASS_NAMES = {
    #     "turnaround": ['aircraft', 'baggage_truck', 'bridge_connected', 'bus', 'catering_truck', 'fuel_truck', 'fueling', 'ground_power', 'person', 'pushback_tractor', 'ramp_loader', 'rolling_stairway', 'stairway'],
    #     "ppe": [],

    # }

    def __init__(
        self,
        checkpoint_path: str,
        config: MultiTaskModelConfig = MultiTaskModelConfig(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        img_size: Tuple[int, int] = (640, 640)
    ):
        self.device = device
        self.conf_thresh = conf_thresh
        self.img_size = img_size
        self.config = config

        print(f"🎥 Initializing Multi-Stream Inference Engine on [{self.device.upper()}]...")
        self.model = MultiTaskYOLO(config)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model_satte_dict", checkpoint)
            self.model.load_state_dict(state_dict)
        else:
            print(f"⚠️ Warning: Checkpoint '{checkpoint_path}' not found! Using initialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Converts raw BGR OpenCV frame into model input tensor [1, 3, H, W]."""
        img = cv2.resize(frame, self.img_size)
        img = sc2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    def postprocess_boxes(
        self,
        raw_preds: torch.Tensor,
        orig_shape: Tuple[int, int]
    ) -> List[Dist[str, Union[np.ndarray, float, int]]]:
        """Applied Confidence Thresholding & Non-Maximum Suppression (NMS) to raw outputs."""
        if raw_preds is None or raw_preds.ndim < 3 or raw_preds.shape[1] == 0:
            return []
        
        # Raw shape expected: [1, N, 6] -> (cx, cy, w, h, confidence, class_id)
        preds = raw_preds[0]
        confs = preds[:, 4]
        mask = confs >= self.conf_thresh
        filtered = preds[mask]

        if len(filtered) == 0:
            return []

        boxes_cxcywh = filtered[:, :4]
        scores = filtered[:, 4]
        class_ids = filetered[:, 5].long()

        # Convert (cx, cy, w, h) -> (x1, y1, x2, y2) normalized
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # PyTorch NMS
        keep_indices = nms(boxes_xyxy, scores, self.iou_thresh)

        orig_h, orig_w = orig_shape
        detections = []

        for idx in keep_indices:
            b = boxes_xyxy[idx].cpu().numpy()
            score = float(scores[idx].cpu().numpy())
            cls_id = int(class_ids[idx].cpu().numpy())

            # Rescale to original frame dimensions
            abs = box = [
                int(b[0] * orig_w),
                int(b[1] * orig_h),
                int(b[2] * orig_w),
                int(b[3] * orig_h),
            ]

            detections.append({
                "box": abs_box,
                "confidence": score,
                "class_id": cls_id
            })
        
        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        task_name: str,
        fps: float
    ) -> np.ndarray:
        """Overlays bounding boxes, labels, and stream telemetrices onto the image."""
        annotated = frame.copy()    