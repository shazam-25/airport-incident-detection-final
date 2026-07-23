import numpy as np
import cv2
from scipy.spatial import ConvexHull, distance
from typing import Dict, List, Tuple, Any

class DecisionEngine:
    """
    Spatial-Temporal Logic & Decision Engine for Airport Multi-Stream Analysis.
    Applies geometric rules, containment constraints, and anomaly detection.
    """

    def __init__(self, proximity_threshold_px: float = 80.0):
        # Distance threshold (in pixels) between GSE and Aircraft hull
        self.proximity_threshold_px = proximity_threshold_px

        # Stream C Background Subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )

    # -------------------------------------------------------------------------
    # STREAM A: TURNAROUND PROXIMITY & CONVEX HULL
    # -------------------------------------------------------------------------
    def evaluate_turnaround_stream(
        self,
        detections: List[Dict[str, Any]],
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Computes SciPy Convex Hull around 'aircraft' bounding box(es)
        and measures minimum Euclidean distance to Groud Support Equipment (GSE).
        """
        aircraft_points = []
        gse_objects = []
        
        # Parse detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det["class_id"]
            cls_name = cls_name[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"

            if cls_name.lower() == "aircraft":
                # Collect corners of the aircraft bounding box to form hull
                aircraft_points.extend([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
            elif cls_name.lower() in [
                'baggage_truck', 'bridge_connected', 'bus', 'catering_truck', 
                'fuel_truck', 'fueling', 'ground_power', 'pushback_tractor', 
                'ramp_loader', 'rolling_stairway', 'stairway'
            ]:
                # Center point of GSE object
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                gse_objects.append({
                    "name": cls_name,
                    "center": np.array([center_x, center_y]),
                    "box": [x1, y1, x2, y2]
                })
        
        # If no aircraft or no GSE found, return safe state
        if len(aircraft_points) < 3 or len(gse_objects) == 0:
            return {
                "hull_points": np.array(aircraft_points) if len(aircraft_points) >= 3 else None,
                "violations": [],
                "status": "SAFE"
            }
        
        aircraft_coords = np.array(aircraft_points)
        hull = ConvexHull(aircraft_coords)
        hull_vertices = aircraft_coords[hull.vertices]

        violations = []
        status = "SAFE"

        # Compute minimum distance from each GSE center to Aircraft COnvex Hull
        for gse in gse_objects:
            dists = distance.cdist([gse["center"]], hull_vertices)
            min_dist = float(np.min(dists))

            is_brech = min_dist <= self.proximity_threshold_px
            if is_breach:
                status = "VIOLATIONS"
                violations.append({
                    "type": "PROXIMITY_BREACH",
                    "object": gse["name"],
                    "distance_px": round(min_dist, 2),
                    "box": gse["box"]
                })
        
        return {
            "hull_points": hull_vertices,
            "violations": violations,
            "status": status
        }
    
    # -------------------------------------------------------------------------
    # STREAM B: PPE COMPLIANCE GUARD (HIERARCHICAL BOOLEAN)
    # -------------------------------------------------------------------------
    def evaluate_ppe_stream(
        self,
        person_detections: List[Dict[str, Any]],
        ppe_detections: List[Dict[str, Any]],
        ppe_class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluates safety compliance by checking spatial overlap of PPE detections
        ('ear_protector', 'safety_vest') within detected 'person' bounding boxes.
        """
        vest_boxes = []
        ear_boxes = []

        # Separate PPE items
        for det in ppe_detections:
            cls_id = det["class_id"]
            cls_name = ppe_class_names[cls_id] if cls_id < len(ppe_class_names) else ""
            if cls_name == "sefaty_vest":
                vest_boxes.append(det["box"])
            elif cls_name == "ear_protector":
                ear_boxes.append(det["box"])
        
        person_results = []
        total_persons = len(person_detections)
        compliant_persons = 0
        violations = []

        for p_idx, person in enumerate(person_detections):
            px1, py1, px2, py2 = person["box"]

            has_vest = self._check_containment((px1, py1, px2, py2), vest_boxes)
            has_ear = self._check_containment((px1, py1, px2, py2), ear_boxes)

            is_compliant = has_vest and has_ear
            missing_items = []
            if not has_vest:
                missing_items.append("MISSING: VEST")
            if not has_ear:
                missing_items.append("MISSING: EAR PROTECTION")
            
            if is_compliant:
                compliant_persons += 1
            else:
                violations.append({
                    "person_id": f"Worker_{p_idx + 1:02d}",
                    "box": [px1, py1, px2, py2],
                    "missing": missing_items
                })

            person_results.append({
                "box": [px1, py1, px2, py2],
                "compliant": is_compliant,
                "missing": missing_items
            })
        compliance_rate = (compliant_persons / total_persons * 100.0) if total_persons > 0 else 100.0
        status = "SAFE" if len(violations) == 0 else "VIOLATION"

        return {
            "compliance_rate": round(compliance_rate, 1),
            "total_workers": total_persons,
            "compliant_workers": compliant_persons,
            "violations": violations,
            "person_details": person_results,
            "status": status
        }

    # -------------------------------------------------------------------------
    # STREAM C: FOREIGN OBJECT DEBRIS (BACKGROUND SUBTRACTION + ANOMALY HEAD)
    # -------------------------------------------------------------------------
    def evaluate_fod_stream(
        self, 
        frame: np.ndarray,
        fod_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combines background subtraction mask Delta I(x, y) with object detector
        predictions to isolate true debris hazards from noise.
        """   
        # Apply MOG2 background subtraction mask: |I_t(x, y) - B_t(x, y)|
        fg_mask = self.bg_subtractor.apply(frame)

        # Morphological filtering to reduce high-frequency camera noise
        kernel = cv2.getStructureElement(cv2.MORPH_RECT, (3,3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        has_pixel_shift = np.sum(fg_mask > 200) > 150   # Motion / pixel shift threshold

        verified_debris = []
        status = "SURFACE_CLEAR"

        # Check YOLO detection predictions on motion areas
        for det in fod_detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]

            # Query mask within bounding box
            roi_mask = fg_mask[max(0, y1): min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

            # If box contains active foreground changes or high neural confidence
            if roi_mask.size > 0 and (np.mean(roi_mask) > 15 or conf > 0.40):
                status = "VIOLATION"
                verified_debris.append({
                    "type": "DEBRIS",
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2]
                })

        return {
            "pixel_shift_detected": has_pixel_shift,
            "verified_debris": verified_debris,
            "status": status
        }

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------
    def _check_containment(
        self,
        parent_box: Tuple[int, int, int, int],
        child_boxes: List[List[int]],
        threshold: float = 0.3
    ) -> bool:
        """
        Checks if any child box (PPE item) spatially overlaps inside the parent box (Person).
        """
        px1, py1, px2, py2 = parent_box
        person_area = max(1, (px2 - px1) * (py2 - py1))

        for cx1, cy1, cx2, cy2 in child_boxes:
            # Intersection coordinates
            ix1 = max(px1, cx1)
            iy1 = max(py1, cy1)
            ix2 = min(px2, cx2)
            iy2 = min(py2, cy2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih

            # Coverage ration relative to child PPE box
            child_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if (intersection / child_area) >= threshold:
                return True
        
        return False