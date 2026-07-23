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

    CLASS_NAMES = {
        "turnaround": ['aircraft', 'baggage_truck', 'bridge_connected', 'bus', 'catering_truck', 'fuel_truck', 'fueling', 'ground_power', 'person', 'pushback_tractor', 'ramp_loader', 'rolling_stairway', 'stairway'],
        "ppe": ["ear_protector", "safety_vest"],
        "fod": ["debris_object"]
    }

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
        class_ids = filtered[:, 5].long()

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
        color = self.STREAM_COLORS.get(task_name, (0, 255, 0))
        classes = self.CLASS_NAMES.get(task_name, [])

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cls_id = det["class_id"]

            cls_name = classes[cls_id] if cls_id < len(classes) else f"Class-{cls_id}"
            label = f"{cls_name}: {conf:.2f}"

            # Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Text background badge
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Header overlay with Stream statistics
        header_str = f"STREM: [{task_name.upper()}] | FPS: {fps:.1f} | Detections: {len(detections)}"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            header_str,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return annotated
    
    @torch.no_grad()
    def process_triple_stream(
        self,
        stream_sources: Dict[str, Union[str, int]],
        output_video_path: Optional[str] = "output_multistream.mp4",
        max_frames: int = 300
    ):
        """Simulates concurrent stream ingestion for 'turnaround', 'ppe', and 'fod'.
        Generates a 2x2 multi-grid tile view of all video streams."""
        task_id_map = {"turnaround": 0, "ppe": 1, "fod": 2}
        caps = {}

        # Open video streams
        for task, source in stream_sources.items():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"❌ Failed to open video stearm for [{task}]: {source}")
                return
            cap[task] = cap
        
        print(f"\n🎬 Starting Multi-Camera Inference Stream Processing...")

        # Setup Video Writer for Multi-Tile Grid Output
        writer = None
        frame_count = 0

        try:
            while frame_count < max_frames:
                start_time = time.time()
                frames = {}

                # Read 1 frame from each camera stream
                for task, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video back to frame 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    frames[task] = frame

                if len(frames) < len(stream_sources):
                    break

                annotated_frames = {}

                # Execute Inference per Camera Stream
                for task_name, raw_frame in frames.items():
                    orig_h, orig_w = raw_frame.shape[:2]
                    input_tensor = self.preprocess_frame(raw_frame).to(self.device)
                    task_id_tensor = torch.tensor([task_id_map[task_name]], device=self.device)

                    # Model Forward Pass
                    preds_dict = self.model(input_tensor, task_id_tensor)
                    raw_preds = preds_dict[task_name]["predictions"]

                    # Post-process & Draw
                    dets = self.postprocess_boxes(raw_preds, (orig_h, orig_w))
                    elapsed = time.time() - start_time
                    fps = 1.0 / max(elapsed, 1e-5)

                    annotated = self.draw_detections(raw_frame, dets, task_name, fps)
                    annotated_frames[task_name] = cv2.resize(annotated, (640, 360))

                # Combine 3 streams into a 2x2 grid layout
                top_row = np.hstack([annotated_frames["turnaround"], annotated_frames["ppe"]])
                blank_tile = np.zeros_like(annotated_frames["fod"])
                bottom_row = np.hstack([annotated_frames["fod"], blank_tile])
                grid_frame = np.vstack([top_row, bottom_row])

                # Initialize Video Writer
                if writer is None and output_video_path:
                    grid_h, grid_w = grid_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (grid_w, grid_h))

                if writer:
                    writer.write(grid_frame)

                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"• Processed {frame_count}/{max_frames} video frames across all 3 streams.")

        finally:
            for cap in caps.values():
                cap.release()
            if writer:
                writer.release()
            print(f"\n✅ Multi-Stream Processing Complete! Saved tiled video feed to: {output_video_path}")