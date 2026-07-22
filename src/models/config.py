"""
Centralized model configuration script that defines the network parameters, backbone
settings, and task head output specifications.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

# Automatically identify project root directory (two levels up from src/models/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class HeadConfig:
    name: str
    num_classes: int
    class_names: List[str]
    loss_weight: float = 1.0    # Weight for multi-task loss balancing

@dataclass
class MultiTaskModelConfig:
    # Backbone settings
    backbone_type: str = "yolov8m"    # Options: yolov8n, yolov8s, yolov8m, yolov8l
    weights_dir: str = "weights"
    pretrained: bool = True
    input_size: tuple = (640, 640)

    @property
    def weights_path(self) -> str:
        """Returns the absolute path to the weight file at project root."""
        weights_folder = PROJECT_ROOT / "model" / self.weights_dir
        
        # Automatically create the weights folder at project root if missing
        weights_folder.mkdir(parents=True, exist_ok=True)
        
        return str(weights_folder / f"{self.backbone_type}.pt")

    # Task Heads Configuration
    heads: Dict[str, HeadConfig] = field(default_factory=lambda: {
        "turnaround": HeadConfig(
            name="turnaround",
            num_classes=13,
            class_names=[
                "aircraft", "baggage_truck", "bridge_connected", "bus", "catering_truck", 
                "fuel_truck", "fueling", "ground_power", "person", "pushback_tractor", 
                "ramp_loader", "rolling_stairway", "stairway"
            ],
            loss_weight=1.0
        ),
        "ppe": HeadConfig(
            name="ppe",
            num_classes=2,
            class_names=["ear_protector", "safety_vest"],
            loss_weight=1.2  # Slight boost to prioritize safety critical detection
        ),
        "fod": HeadConfig(
            name="fod",
            num_classes=1,
            class_names=["foreign_object_debris"],
            loss_weight=1.5  # Higher weight due to extreme small-object scale difficulty
        )
    })

    def get_head_names(self) -> List[str]:
        return list(self.heads.keys())