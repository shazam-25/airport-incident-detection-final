import os
import cv2
import time
import numpy as np
from typing import Optional

class LocalObjectStorage:
    """
    Unstructured Blob Storage Tier.
    Saves raw high-resolution violation frames and cropped evidence snapshots
    to local disk/MinIO without bloating relational DB indexes.
    """
    def __init__(self, base_dir: str = "storage/bobs"):
        