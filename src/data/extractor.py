import os
import glob
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDatasetExtractor:
    def __init__(self, download_root="data/raw"):
        """Initializes Kaggle API connection and target raw folders."""
        self.api = KaggleApi()
        self.api.authenticate()
        self.download_root = download_root

        # Explicit download configurations for 3 specific tasks
        self.datasets = {
            "turnaround": "shazam0k/airport-turnaround-dataset",
            "ppe": "shazam0k/airport-ppe-dataset",
            "fod": "kilogrand/foreign-object-debris-in-airports-fod-a-dataset"
        }
    
    def download_all(self):
        """Programmatically downloads and unzips all 3 distinct sets cleanly."""
        for task_name, kaggle_slug in self.datasets.items():
            target_dir = os.path.join(self.download_root, task_name)
            os.makedirs(target_dir, exist_ok=True)

            print(f"\n📥Downloading {task_name.upper()} dataset from Kaggle...")

            # Download files directly into target raw directory and unzip
            self.api.dataset_download_files(
                kaggle_slug,
                path=target_dir,
                unzip=True,
                quiet=False
            )
            print(f"✅Saved raw {task_name} files to: {target_dir}")
        return "All extractions completed."

    def reconstruct_all(self):
        """Restructure the raw directory of all streams."""
        # ----------------------------------------
        # Process "Turnaround" & "PPE" streams.
        # ----------------------------------------
        for task_name in ["turnaround", "ppe"]:
            stream_dir = Path(self.download_root) / task_name
            if not stream_dir.exists():
                print(f"❌No dataset found for '{task_name}'")
            print(f"Consolidating Multi-Stream Splits -> [{task_name.upper()}]")

            all_found_images = []
            all_found_labels = []
            garbage_to_delete = []

            # Traverse the nested layout to extract pairs
            for root, dirs, files in os.walk(stream_dir):
                root_path = Path(root)
                if root_path.name in ["train", "valid", "test", "images", "labels"]:
                    if root_path not in garbage_to_delete:
                        garbage_to_delete.append(root_path)
                for file in files:
                    file_path = root_path / file
                    ext = file_path.suffix.lower()
                    # Check for redundant setup parameters and metadata
                    if file in ["README.dataset.txt", "README.roboflow.txt"] or ext == ".html":
                        if file_path not in garbage_to_delete:
                            garbage_to_delete.append(root_path)
                        continue
                    if ext in [".jpg", "jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                        all_found_images.append(file_path)
                    elif ext == ".txt": all_found_labels.append(file_path)
            
            # Standardize target directory structure
            target_images_dir = stream_dir / "images"
            target_labels_dir = stream_dir / "labels"
            target_images_dir.mkdir(parents=True, exist_ok=True)
            target_labels_dir.mkdir(parents=True, exist_ok=True)

            # Consolidate raw data shards into a flat directory layer
            moved_imgs, moved_lbls = 0, 0
            for img_file in all_found_images:
                if img_file.parent != target_images_dir:
                    shutil.move(str(img_file), str(target_images_dir / img_file.name))
                    moved_imgs += 1
            for lbl_file in all_found_labels:
                if lbl_file != target_labels_dir:
                    shutil.move(str(lbl_file), str(target_labels_dir / lbl_file.name))
                    moved_lbls += 1
            print(f" ├── Migrated {moved_imgs} images into unified flat folder layout.")
            print(f" └── Migrated {moved_lbls} text label annotations into unified flat folder layout.")

            # Purge empty directories and unwanted configuration clutter files
            for trash in sorted(garbage_to_delete, key=lambda p: len(str(p)), reverse=True):
                if trash.exists():
                    if trash.is_file():
                        trash.unlink()
                    elif trash.is_dir() and not os.listdir(trash):
                        trash.rmdir()

        # ----------------------------------------
        # Process "FOD" stream
        # ----------------------------------------
        fod_root = Path(self.download_root) / "fod"
        if os.path.exists(fod_root):
            print("Consolidating Nested Structured layouts -> [FOD]")

            target_images_dir = fod_root / "images"
            target_labels_dir = fod_root / "xml_labels"
            target_images_dir.mkdir(parents=True, exist_ok=True)
            target_labels_dir.mkdir(parents=True, exist_ok=True)

            all_images = list(fod_root.glob("**/JPEGImages/*.*")) + list(fod_root.glob("**/*.jpg"))
            all_labels = list(fod_root.glob("**/Annotations/*.xml")) + list(fod_root.glob("**/*.xml"))

            moved_imgs, moved_lbls = 0, 0
            for img in all_images:
                if img.exists() and img.parent!= target_images_dir:
                    shutil.move(str(img), str(target_images_dir / img.name))
                    moved_imgs += 1
            for label in all_labels:
                if label.exists() and label.parent != target_labels_dir:
                    shutil.move(str(label), str(target_labels_dir / label.name))
                    moved_lbls += 1
            print(f" ├── Flattened {moved_imgs} source images into fod/images/")
            print(f" └── Flattened {moved_lbls} Pascal VOC XML frames into fod/labels/")

            # Clear out empty residual subdirectories
            for item in fod_root.iterdir():
                if item.is_dir() and item.name not in ["images", "labels"]:
                    shutil.rmtree(item)

        print("✅Raw directory reconstruction and garbage collection purge complete.")


