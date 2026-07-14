import os
import glob
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt

class MultiTaskDatasetProfiler:
    def __init__(self, raw_root="data/raw"):
        self.raw_root = raw_root
        self.tasks = ["turnaround", "ppe", "fod"]

    def run_initial_eda(self):
        """Profiles image metrics and extracts raw bounding box/class distributions."""
        report = {}

        for task in self.tasks:
            task_path = os.path.join(self.raw_root, task)

            # Gather all image assets recursively
            images = glob.glob(os.path.join(task_path, "images", "**/*.[jJ][pP][gG]"), recursive=True) + \
                glob.glob(os.path.join(task_path, "images", "**/*.[pP][nN][gG]"), recursive=True)

            # Gather all matching label text tracking documents
            labels = glob.glob(os.path.join(task_path, "labels", "**/*.txt"), recursive=True)

            # Parse individual class distributions within annotations
            class_counter = Counter()
            for label in labels:
                if os.path.basename(label) in ['classes.txt', 'notes.txt']:
                    continue
                with open(label, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if parts:
                            class_counter[int(parts[0])] += 1
            
            report[task] = {
                "total_images": len(images),
                "total_label_files": len(labels),
                "raw_class_frequencies": dict(sorted(class_counter.items())),
                "mismatch_warning": len(images) != len(labels)
            }

        return report

    def parse_turnaround_classes(self):
        """Counts native classes in the Turnaround dataset."""
        lbl_dir = os.path.join(self.raw_root, "turnaround", "labels", "**", "*.txt")
        counter = Counter()
        for lbl in glob.glob(lbl_dir, recursive=True):
            if os.path.basename(lbl) in ['classes.txt', 'notes.txt']: continue
            with open(lbl, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts:
                        counter[int(parts[0])] += 1
        return dict(sorted(counter.items()))

    def parse_ppe_classes(self):
        """Counts native classes in the PPE dataset."""
        lbl_dir = os.path.join(self.raw_root, "ppe", "labels", "**", "*.txt")
        counter = Counter()
        for lbl in glob.glob(lbl_dir, recursive=True):
            if os.path.basename(lbl) in ['classes.txt', 'notes.txt']: continue
            with open(lbl, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts:
                        counter[int(parts[0])] += 1
        return dict(sorted(counter.items()))
    
    def profile_and_convert_fod_classes(self):
        """Parses Pascal VOC XMLs for FOD, maps to a local YOLO class counter, 
        and returns the distribution"""
        xml_dir = os.path.join(self.raw_root, "fod", "labels", "**", "*.xml")
        xml_files = glob.glob(xml_dir, recursive=True)

        counter = Counter()
        local_class_map = {} # Maps string names to local 0, 1, 2 indices

        for xml_path in xml_files:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text.strip().lower()

                    if cls_name not in local_class_map:
                        local_class_map[cls_name] = len(local_class_map)

                    local_id = local_class_map[cls_name]
                    counter[local_id] += 1
            except Exception: continue

        return dict(sorted(counter.items())), local_class_map

    def plot_distributions(self, t_dist, p_dist, f_dist, f_map, t_names=None, p_names=None):
        """Generates clear, distinct bar charts for each head's target dataset."""
        fig, axes = plt.subplots(1, 3, figsize=(20,5))

        # 1. Turnaround Plot
        t_labels = t_names if t_names else [f"Class {k}" for k in t_dist.keys()]
        axes[0].bar([str(x) for x in t_dist.keys()], t_dist.values(), color='steelblue')
        axes[0].set_title("Head 1: Turnaround Native Distribution")
        axes[0].set_xticklabels(t_labels, rotation=45, ha='right')

        # 2. PPE Plot
        p_labels = p_names if p_names else [f"Class {k}" for k in p_dist.keys()]
        axes[1].bar([str(x) for x in p_dist.keys()], p_dist.values(), color='darkorange')
        axes[1].set_title("Head 2: PPE Native Distribution")
        axes[1].set_xticklabels(p_labels, rotation=45, ha='right')

        # 3. FOD Plot
        inverse_f_map = {v: k for k, v in f_map.items()}
        f_labels = [inverse_f_map[k] for k in f_dist.keys()]
        axes[2].bar([str(x) for x in f_dist.keys()], f_dist.values(), color='forestgreen')
        axes[2].set_title("Head 3: FOD Localized Distribution")
        axes[2].set_xticklabels(f_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
