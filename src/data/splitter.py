import os
import glob
import shutil
import random
from collections import Counter
from sklearn.model_selection import train_test_split

class MultiTaskDataSplitter:
    def __init__(self, interim_dir="data/interim", processed_dir="data/processed", ratios=(0.70, 0.15, 0.15)):
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.ratios = ratios    # (train, val, test)

    def _profile_dataset_labels(self, stream):
        """Scans the interim text annotations to calculate the primary/rarest class
        associated with each individual image frame."""
        lbl_dir = os.path.join(self.interim_dir, stream, "labels")
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

        file_to_primary_class = {}

        for lbl_path in lbl_files:
            base_name = os.path.splitext(os.path.basename(lbl_path))[0]
            img_path = os.path.join(self.interim_dir, stream, "images", "f{base_name}.jpg")

            if not os.path.exists(img_path):
                continue

            classes_in_file = []
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts: classes_in_file.append(int(parts[0]))

                if classes_in_file:
                    # Stratification Anchor: Assign this file to its rarest present class
                    # to guarantee rare class instances are distributed evenly across splits
                    file_to_primary_class[img_path] = classes_in_file
                else:
                    # Background-only images or empty frames assigned to an arbitrary background class marker
                    file_to_primary_class[img_path] = [-1]
        
        return file_to_primary_class

    def execute_splits(self):
        """Executes a true multi-label stratification split across all three streams
        individually using iterative allocation matching."""
        splits = ["train", "val", "test"]
        streams = ["turnaround", "ppe", "fod"]

        # Build production structure directories
        for s in splits:
            for stream in streams:
                os.makedirs(os.path.join(self.processed_dir, s, stream, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.processed_dir, s, stream, "labels"), exist_ok=True)

        summary = {}
        random.seed(42)

        for stream in streams:
            file_label_map = self._profile_dataset_labels(stream)
            images = list(file_label_map.keys())

            # Count global class distributions for this specific stream
            global_counts = Counter([cls for classes in file_label_map.values() for cls in classes])
            # Sort classes from rarest to most frequent to prioritize allocating critical low-sample data first
            sorted_classes = [k for k, v in sorted(global_counts.items(), key=lambda item: item[1])]

            train_pool, val_pool, test_pool = [], [], []

            # Map out clear numeric distribution buckets based on target rations
            target_train_pct, target_val_pct, target_test_pct = self.ratios

            # Iteratively distribute samples starting with the rarest classes
            for target_cls in sorted_classes:
                matching_files = [f for f in images if target_cls in file_label_map[f] and f not in train_pool and f not in val_pool and f not in test_pool]
                random.shuffle(matching_files)

                for f in matching_files:
                    total_allocated = len(train_pool) + len(val_pool) + len(test_pool)
                    if total_allocated == 0:
                        train_pool.append(f)
                        continue

                    current_train_ratio = len(train_pool) / total_allocated
                    current_val_ratio = len(val_pool) / total_allocated

                    if current_train_ratio < target_train_pct:
                        train_pool.append(f)
                    elif current_val_ratio < target_val_pct:
                        val_pool.append(f)
                    else:
                        test_pool.append(f)
                    
            # Catch-all loop for any unallocated files (e.g., background samples)
            remaining_files = [f for f in images if f not in train_pool and f not in val_pool and f not in test_pool]
            for f in remaining_files:
                r = random.random()
                if r < target_train_pct:
                    train_pool.append(f)
                elif r < (target_train_pct + target_val_pct):
                    val_pool.append(f)
                else:
                    test_pool.append(f)
            
            # Physically deploy the files to the final data directories
            file_groups = {"train": train_pool, "val": val_pool, "test": test_pool}
            for split, files in file_groups.items():
                for f_path in files:
                    base_name = os.path.basename(f_path)
                    lbl_name = os.path.splitext(base_name)[0] + ".txt"

                    shutil.copy(f_path, os.path.join(self.processed_dir, split, stream, "images", base_name))

                    lbl_src = os.path.join(self.interim_dir, stream, "labels", lbl_name)
                    if os.path.exists(lbl_src):
                        shutil.copy(lbl_src, os.path.join(self.processed_dir, split, stream, "labels", lbl_name))

            summary[stream] = {
                "train": len(train_pool), 
                "val": len(val_pool),
                "test": len(test_pool)
            }
        
        return summary