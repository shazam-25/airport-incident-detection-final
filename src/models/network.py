"""
Core PyTorch model class:
It uses a shared feature extractor (backbone + neck) and routes the extracted features into
separate detection heads on the input task ID.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from src.models.config import MultiTaskModelConfig

class PretrainedYOLOBackbone(nn.Module):
    """
    Extracts multi-scale feature maps [P3, P4, P5] from a pretrained YOLOv8 model 
    using forward hooks to preserve internal skip-connections (Concat layers).
    """
    def __init__(self, weights_path="weights/yolov8m.pt"):
        super().__init__()
        # Load native Ultralytics PyTorch model
        self.yolo_model = YOLO(weights_path).model
        self.extracted_features = {}

        # Layer indices corresponding to P3, P4, P5 feature outputs in YOLOv8
        # Layer 15: P3/8 (small objects - FOD)
        # Layer 18: P4/16 (medium objects - PPE)
        # Layer 21: P5/32 (large objects - Turnaround)
        self.target_layers = {15: "P3", 18: "P4", 21: "P5"}

        # Register forward hooks on target feature layers
        for layer_idx, name in self.target_layers.items():
            self.yolo_model.model[layer_idx].register_forward_hook(
                self._get_hook(name)
            )

    def _get_hook(self, name):
        def hook(module, input, output):
            self.extracted_features[name] = output
        return hook

    def forward(self, x):
        self.extracted_features.clear()
        # Execute forward pass through the native YOLO network structure
        _ = self.yolo_model(x)
        
        # Return feature maps list [P3, P4, P5]
        return [
            self.extracted_features["P3"],
            self.extracted_features["P4"],
            self.extracted_features["P5"]
        ]


class TaskDetectionHead(nn.Module):
    """
    Decoupled detection head for a specific task stream.
    Outputs box regression coordinates and localized class logits.
    """
    def __init__(self, in_channels: list, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # Using Ultralytics native Detect module structure for compatibility
        self.detect = Detect(nc=num_classes, ch=in_channels)

        # Attach default feature map strides required by YOLO loss calculations
        # P3/8 (8px), P4/16 (16px), P5/32 (32px)
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x):
        return self.detect(x)


class MultiTaskYOLO(nn.Module):
    """
    Shared-Backbone Multi-Task Neural Network with decoupled task heads 
    for Turnaround Monitoring, PPE Compliance, and FOD Detection.
    """
    def __init__(self, config: MultiTaskModelConfig = MultiTaskModelConfig()):
        super().__init__()
        self.config = config
        
        # Real YOLOv8m feature map output channels for [P3, P4, P5]
        self.feature_channels = [192, 384, 576] if "m" in config.backbone_type else [128, 256, 512]

        # Pretrained Backbone Extractor
        self.backbone = PretrainedYOLOBackbone(weights_path=config.weights_path)
        
        # Task-Specific Heads
        self.heads = nn.ModuleDict({
            head_name: TaskDetectionHead(
                in_channels=self.feature_channels,
                num_classes=head_cfg.num_classes
            )
            for head_name, head_cfg in config.heads.items()
        })
        
        # Map task index to head keys
        self.task_idx_to_name = {0: "turnaround", 1: "ppe", 2: "fod"}

    def forward(self, x, task_ids=None):
        """
        Args:
            x (Tensor): Input image tensor [Batch, 3, 640, 640]
            task_ids (Tensor, optional): Batch task identity flags [Batch]
        """
        # If task_ids is provided, route through respective task heads
        if task_ids is not None:
            # Group batch by task ID for efficient multi-head forwarding
            outputs = {}
            unique_tasks = torch.unique(task_ids)
            
            for t_id in unique_tasks:
                task_name = self.task_idx_to_name[int(t_id.item())]
                mask = (task_ids == t_id)
                task_input = x[mask]

                # Extract real multi-scale features [P3, P4, P5]
                feats = self.backbone(task_input)
                
                # Route through corresponding head
                head_out = self.heads[task_name](feats)
                outputs[task_name] = {
                    "mask": mask,
                    "predictions": head_out
                }
            return outputs
            
        # Inference mode: return predictions across all heads
        else:
            feats = self.backbone(x)
            return {
                head_name: head_module(feats) 
                for head_name, head_module in self.heads.items()
            }