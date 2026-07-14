import os
import cv2
import glob
import xml.etree.ElementTree as ET

class MultiTaskPreprocessor:
    def __init__(self, raw_dir="data/raw", interim_dir="data/interim", target_size=(640, 640)):
        self.raw_dir = raw_dir
        self.interim_dir = interim_dir
        self.target_size = target_size

        # Explicit whitelists based on Initial EDA analysis
        self.ppe_whitelist = {
            0: 0,    # Ear Protector -> Head 2 Local Class 0
            8: 1     # Safety Vest -> Head 2 Local Class 1
            # Note: 'Person' comes from Turnaround
        }

    def letterbox_image(self, img):
        """Resizes image using padding to preserve aspect ration for exact geometric logic."""
        h, w = img.shape[:2]
        th, tw = self.target_size
        scale = min(th / h, tw / w)
        nh, nw = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded

    def process_all_streams(self):
        """Process Turnaround, PPE, and FOD into intermediate representations."""
        summary = {}
        for stream in ["turnaround", "ppe", "fod"]:
            out_img_dir = os.path.join(self.interim_dir, stream, "images")
            out_lbl_dir = os.path.join(self.interim_dir, stream, "labels")
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)

            processed_count = 0

            # Find all image paths
            raw_img_dir = os.path.join(self.raw_dir, stream)
            img_paths = glob.glob(os.path.join(raw_img_dir, "**", "*.[jJ][pP][gG]"), recursive=True) + \
                        glob.glob(os.path.join(raw_img_dir, "**", "*.[pP][nN][gG]"), recursive=True)
            
            for img_path in img_paths:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                img = cv2.imread(img_path)
                if img is None: continue

                # Apply letterboxing
                padded_img = self.letterbox_image(img)

                # --- Stream-Specific Annotations Processing ---
                has_valid_annotations = False

                if stream == "turnaround":
                    lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                    if os.path.exists(lbl_path):
                        with open(lbl_path, 'r') as rf, open(os.path.join(out_lbl_dir, f"{base_name}.txt"), 'w') as wf:
                            wf.write(rf.read())
                        has_valid_annotations = True
                
                elif stream == "ppe":
                    lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                    if os.path.exists(lbl_path):
                        filtered_lines = []
                        with open(lbl_path, 'r') as f:
                            for line in f.readlines():
                                parts = line.strip().split()
                                if parts and int(parts[0]) in self.ppe_whitelist:
                                    new_cls = self.ppe_whitelist[int(parts[0])]
                                    filtered_lines.append(f"{new_cls} " + " ".join(parts[1:]) + "\n")
                        if filtered_lines:
                            with open(os.path.join(out_lbl_dir, f"{base_name}.txt"), 'w') as wf:
                                wf.writelines(filtered_lines)
                            has_valid_annotations = True
                
                elif stream == "fod":
                    xml_path = img_path.replace("images", "labels").replace(".jpg", ".xml").replace(".png", ".xml")
                    if os.path.exists(xml_path):
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            size = root.find('size')
                            w = int(size.find('width').text)
                            h = int(size.find('height').text)

                            yolo_lines = []
                            for obj in root.iter('object'):
                                xmlbox = obj.find('bndbox')
                                xmin = float(xmlbox.find('xmin').text)
                                xmax = float(xmlbox.find('xmax').text)
                                ymin = float(xmlbox.find('ymin').text)
                                ymax = float(xmlbox.find('ymax').text)

                                xc = (xmin + xmax) / (2.0 * w)
                                yc = (ymin + ymax) / (2.0 * h)
                                nw = (xmax - xmin) / w
                                nh = (ymax - ymin) / h
                                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

                            if yolo_lines:
                                with open(os.path.join(out_lbl_dir, f"{base_name}.txt"), 'w') as wf:
                                    wf.writelines(yolo_lines)
                                has_valid_annotations = True
                        except Exception: continue
                
                # Save processed image if valid labels existed
                if has_valid_annotations:
                    cv2.imwrite(os.path.join(out_img_dir, f"{base_name}.jpg"), padded_img)
                    processed_count += 1
            
            summary[stream] = processed_count
        
        return summary
