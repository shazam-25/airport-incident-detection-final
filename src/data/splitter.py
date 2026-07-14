import os
import glob
import shutil
import random
from sklearn.model_selection import train_test_split

class MultiTaskDataSplitter:
    def __init__(self, interim_dir="data/interim", processed_dir="data/processed", ratios=(0.70, 0.15, 0.15)):
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.ratios = ratios    # (train, val, test)

    def execute_splits(self):
        """Partitions interim files into structured train/val/test splits."""
        splits = ["train", "val", "test"]
        streams = ["turnaround", "ppe", "fod"]

        # Build production structure
        for s in splits:
            for stream in streams:
                os.makedirs(os.path.join(self.processed_dir, s, stream, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.processed_dir, s, stream, "labels"), exist_ok=True)

        summary = {}

        for stream in streams:
            img_dir = os.path.join(self.interim_dir, stream, "images")
            images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

            # Shuffle deterministically
            random.seed(42)
            random.shuffle(images)

            # 70% Train, 30% Temp (Val + Test)
            train_files, temp_files = train_test_split(images, train_size=self.ratios[0], random_state=42)

            # Split Temp 50/50 into Val and Test (making them 15% and 15% of total)
            val_files, test_files = train_test_split(temp_files, train_size=0.5, random_state=42)

            file_groups = {
                "train": train_files,
                "val": val_files,
                "test": test_files
            }

            for split, files in file_groups.items():
                for f_path in files:
                    base_name = os.path.basename(f_path)
                    lbl_name = os.path.splitext(base_name)[0] + ".txt"

                    # Copy images
                    shutil.copy(f_path, os.path.join(self.processed_dir, split, stream, "images", base_name))
                    # Copy labels
                    lbl_src = os.path.join(self.interim_dir, stream, "labels", lbl_name)
                    if os.path.exists(lbl_src):
                        shutil.copy(lbl_src, os.path.join(self.processed_dir, split, stream, "labels", lbl_name))

            summary[stream] = {
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files)
            }
        return summary