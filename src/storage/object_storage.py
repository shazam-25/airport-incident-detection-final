import os
from pathlib import Path
import cv2
import time
import numpy as np
from typing import Optional

# Automatically identify project root directory (two levels up from src/storage/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class LocalObjectStorage:
    """
    Unstructured Blob Storage Tier.
    Saves raw high-resolution violation frames and cropped evidence snapshots
    to local disk/MinIO without bloating relational DB indexes.
    """
    def __init__(self, base_dir: str = "storage/bobs"):
        self.base_dir = PROJECT_ROOT / base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        # Subdirectories per stream
        for stream in ["turnaround", "ppe", "fod"]:
            os.makedirs(os.path.join(self.base_dir, stream), exist_ok=True)
    
    def save_violation_image(
        self,
        frame: np.ndarray,
        stream_name: str,
        event_type: str,
        crop_box: Optional[list] = None
    ) -> str:
        """
        Saves full frame or cropped violation image to storage and returns relative path. 
        """
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        filename = f"{event_type.lower()}_{timestamp_str}_{millis:03d}.png"

        stream_folder = os.path.join(self.base_dir, stream_name.lower())
        file_path = os.path.join(stream_folder, filename)

        image_to_save = frame.copy()

        # Crop region if box provided (x1, y1, x2, y2)
        if crop_box is not None and len(crop_box) == 4:
            x1, y1, x2, y2 = crop_box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                image_to_save = frame[y1:y2, x1:x2]

        cv2.imwrite(file_path, image_to_save)
        return file_path