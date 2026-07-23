import os
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import numpy as np

class TaskDataset(Dataset):
    """Standard Dataset wrapper for a single detection stream (e.g., FOD, PPE, or Turnaround).
    Expects images in 'img_dir' and YOLO-formatted txt labels in 'label_dir'."""
    def __init__(self, img_dir: str, label_dir: str, task_id: int, img_size: Tuple[int, int] = (640, 640)):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.task_id = task_id
        self.img_size = img_size

        # Gather image filenames
        valid_extensions = (".jpg", ".png", ".jpeg", ".bmp")
        self.image_files = [
            f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(valid_extensions)
        ] if os.path.exists(img_dir) else []

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
    
        # Load ans resize image (BGR -> RGB)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, self.img_size)

        # Convert to Tensor (CHW, normalized [0, 1])
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Load YOLO labels: [class_id, x_center, y_center, width, height]
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        boxes.append([cls_id] + coords)
        
        targets = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, targets, self.task_id


def multitask_collate_fn(batch):
    """
    Custom collate function that batches multi-task images, labels, and task IDs cleanly.
    Formated to match Ultralytics v8DetectionLoss expected target dictionary structure.
    """
    images, targets, task_ids = zip(*batch)

    # Stack image tensors [B, 3, H, W]
    images_stacked = torch.stack(images, dim=0)
    task_ids_stacked = torch.tensor(task_ids, dtype=torch.long)

    task_map = {0: "turnaround", 1: "ppe", 2: "fod"}
    task_targets = {}

    for t_id, task_name in task_map.items():
        task_boxes = []
        # Find batch indices matching this task
        batch_indices = (task_ids_stacked == t_id).nonzero(as_tuple=True)[0]

        for sub_b_idx, real_b_idx in enumerate(batch_indices):
            sample_boxes = targets[real_b_idx]
            
            # Ensure target tensor is 2D matrix [N, 5]
            if sample_boxes.ndim == 1 and sample_boxes.numel() > 0:
                sample_boxes = sample_boxes.unsqueeze(0)

            if sample_boxes.numel() > 0 and sample_boxes.ndim == 2:
                # Prepend batch index within task batch slice -> [sub_b_idx, cls_id, x, y, w, h]
                batch_col = torch.full((sample_boxes.shape[0], 1), sub_b_idx, dtype=torch.float32)
                task_boxes.append(torch.cat([batch_col, sample_boxes], dim=1))

        if len(task_boxes) > 0:
            task_targets[task_name] = torch.cat(task_boxes, dim=0)
        else:
            task_targets[task_name] = torch.zeros((0, 6), dtype=torch.float32)

    return images_stacked, task_targets, task_ids_stacked


def create_multitask_dataloader(
    data_dirs: Dict[str, Dict[str, str]],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (640, 640)
) -> DataLoader:
    """
    Factory function to construct a unified Multi-Task DataLoader across
    Turnaround, PPE, and FOD datasets.

    data_dirs format:
    {
        "turnaround": {"images": "data/processed/turnaround/images/train", "labels": "..."},
        "ppe": {"images": "data/processed/ppe/images/train", "labels": "..."},
        "fod": {"images": "data/processed/fod/images/train", "labels": "..."}
    }
    """
    task_mapping = {"turnaround": 0, "ppe": 1, "fod": 2}
    datasets = []

    for task_name, task_id in task_mapping.items():
        if task_name in data_dirs:
            img_dir = data_dirs[task_name].get("images", "")
            label_dir = data_dirs[task_name].get("labels", "")

            if os.path.exists(img_dir):
                ds = TaskDataset(img_dir=img_dir, label_dir=label_dir, task_id=task_id, img_size=img_size)
                print(f"📦 Loaded [{task_name.upper()}] dataset: {len(ds)} samples")
                datasets.append(ds)
    
    if not datasets:
        raise ValueError("No valid datasets found in provided data_dirs paths!")

    concat_dataset = ConcatDataset(datasets)
    print(f"✅ Combined Multi-Task Dataset total size: {len(concat_dataset)} samples\n")

    return DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multitask_collate_fn,
        pin_memory=True
    )

