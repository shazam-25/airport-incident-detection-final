import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter, defaultdict
import yaml
from pathlib import Path

def convert_voc_xml_to_yolo(xml_path, output_txt_path, target_class_name="debris"):
    """Parses a Pascal VOC XML file and outputs a normalized YOLO text file."""
    if not os.path.exists(xml_path):
        return False
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions to normalize coordinates
    size = root.find('size')
    if size is None:
        return False
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # Avoid division by zero on corrupt headers
    if w == 0 or h == 0:
        return False
    
    yolo_boxes = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text

        # Only extract the debris/anomaly class for Head 3
        if cls_name == target_class_name or target_class_name == 'all':
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / (2.0 * w)
            y_center = (ymin + ymax) / (2.0 * h)
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            # Map strictly to Class 20 (FOD Head Target)
            yolo_boxes.append(f"20 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    if yolo_boxes:
        with open(output_txt_path, 'w') as f:
            f.writelines(yolo_boxes)
        return True
    return False

def class_remap():
    return {
        'turnaround': {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            
        }
    }