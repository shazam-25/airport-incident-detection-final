import os
import cv2
import torch
import numpy as numpy
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultiTaskDataset(Dataset):
    def __init__(self, processed_dir="data/processed", split="train", transform=None):
        """Custom multi-task dataset wrapper mapping three independent domain streams."""
        self.processed_dir = os.path.abspath(processed_dir)
        self.split = split
        self.streams = ["turnaround", "ppe", "fod"]

        # Internal storage containers
        self.image_paths = []
        self.label_paths = []
        self.stream_indices =[] # Track which task head each sample belongs to

        # Build index maps
        for stream_idx, stream in enumerate(self.streams):
            img_dir = os.path.join(self.processed_dir, split, stream, "images")
            if not os.path.exists(img_dir):
                continue

            for filename in sorted(os.listdir(img_dir)):
                if filename.endswith(".jpg"):
                    base_name = os.path.splitext(filename)[0]
                    lbl_path = os.path.join(self.processed_dir, split, stream, "labels", f"{base_name}.txt")

                    if os.path.exists(lbl_path):
                        self.image_paths.append(os.path.join(img_dir, filename))
                        self.label_paths.append(lbl_path)
                        self.stream_indices.append(stream_idx)

            # Default Albumentations Pipeline COnfiguration
            if transform is not None:
                self.transform = transform
            else:
                self.transform = self._get_default_transforms()
    
    def _get_default_transforms(self):
        """Constructs a weather-robust transformation pipeline tailored for airport tarmacs."""
        if self.split == "train":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                # Weather variations simulating external airport environment shifts
                A.OneOf([
                    A.RandomRain(p=0.2, rain_type='heavy'),
                    A.RandomFog(p=0.2, fog_coef_lower=0.3, fog_coef_upper=0.5),
                    A.MotionBlur(p=0.2)
                ], p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # Clean normalization pass for evaluation paths (Val/Test)
            return A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        task_id = self.stream_indices[idx]

        # Read image asset
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load custom local YOLO targets
        bboxes = []
        class_labels = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if parts:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
        
        # Apply transformation matrices
        if self.transform:
            try:
                augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                image = augmented['image']
                bboxes = augmented['bboxes']
                class_labels = augmented['class_labels']
            except Exception:
                # Safe fallback if augmentations reject coordinates
                pass

        # Convert labels to PyTorch arrays
        if len(bboxes) > 0:
            target_boxes = torch.tensor(bboxes, dtype=torch.float32)
            target_classes = torch.tensor(class_labels, dtype=torch.long)
        else:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_classes = torch.zeros((0,), dtype=torch.long)
        
        return image, target_boxes, target_classes, torch.tensor(task_id, dtype=torch.long)

def multi_task_collate_fn(batch):
    """Custom collate tool that groups variant bounding boxes
    without running into size shape mismatch errors."""
    images, targets, target_classes, task_ids = zip(*batch)

    # Staandard batch stack for matching 640x640 tensor frames
    images = torch.stack(images, 0)
    task_ids = torch.stack(task_ids, 0)

    # Process variant target limits safely using indexed assignment matrices
    batch_targets = []
    for i in range(len(targets)):
        num_boxes = targets[i].shape[0]
        if num_boxes > 0:
            # Prepend image batch index matrix onto targets for backprop tracking
            box_indices = torch.full((num_boxes, 1), i, dtype=torch.float32)
            cls_indices = target_classes[i].unsqueeze(1).float()
            combined = torch.cat([box_indices, cls_indices, targets[i]], dim=1)
            batch_targets.append(combined)
        
    if batch_targets:
        batch_targets = torch.cat(batch_targets, dim=0)
    else:
        batch_targets = torch.zeros((0, 6), dtype=torch.float32)

    return images, batch_targets, task_ids