import os
import glob

class DatasetSanityChecker:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = os.path.abspath(processed_dir)
        # Expected max localized classes based on reduction profiling step
        self.stream_class_limits = {
            "turnaround": 13, # 0 to 12 native classes
            "ppe": 2,         # 0: Ear Protector, 1: Safety Vest
            "fod": 1          # 0: debris_object    
        }

    def check_integrity(self):
        """Scans processed directory splits to validate label pairings and bounding box compliance."""
        errors = []
        splits = ["train", "val", "test"]
        streams = ["turnaround", "ppe", "fod"]

        total_checked = 0

        for split in splits:
            for stream in streams:
                img_dir = os.path.join(self.processed_dir, split, stream, "images")
                lbl_dir = os.path.join(self.processed_dir, split, stream, "labels")

                if not os.path.exists(img_dir):
                    continue

                img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
                max_allowed_cls = self.stream_class_limits[stream]

                for img_path in img_files:
                    total_checked += 1
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

                    # 1. Check Pairing Integrity
                    if not os.path.exists(lbl_path):
                        errors.append(f"❌ [{split.upper()} - {stream.upper()}] Missing label file for image: {base_name}.jpg")
                        continue

                    # 2. Check Annotation Coordinates & Classes
                    with open(lbl_path, 'r') as f:
                        for line_idx, line in enumerate(f.readlines(), 1):
                            parts = line.strip().split()
                            if not parts:
                                continue

                            if len(parts) != 5:
                                errors.append(f"❌ [{split.upper()} - {stream.upper()}] Format error in {base_name}.txt line {line_idx}: Expected 5 elements, got {len(parts)}")
                                continue

                            try:
                                cls_id = int(parts[0])
                                coords = [float(x) for x in parts[1:]]
                            except ValueError:
                                errors.append(f"❌ [{split.upper()} - {stream.upper()}] Non-numeric values in {base_name}.txt line {line_idx}")
                                continue
                        
                            # Validate class index limits
                            if cls_id < 0 or cls_id >= max_allowed_cls:
                                errors.append(f"❌ [{split.upper()} - {stream.upper()}] Class Out-of-Bounds in {base_name}.txt line {line_idx}: Local Class {cls_id} invalid (Limit: {max_allowed_cls})")

                            # Validate normalized coordinates bounds [0, 1]
                            for coord_idx, coord in enumerate(coords):
                                coord_names = ["x_center", "y_center", "width", "height"]
                                if coord < 0.0 or coord > 1.0:
                                    errors.append(f"⚠️ [{split.upper()} - {stream.upper()}] Unnormalized leak in {base_name}.txt line {line_idx}: {coord_names[coord_idx]} value is {coord}")

        return total_checked, errors