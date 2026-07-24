import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import torch
import os

from src.models.network import MultiTaskYOLO
from src.models.config import MultiTaskModelConfig
from src.analytics.decision_engine import DecisionEngine
from src.storage.manager import StorageManager

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Airside Operational & Safety Incident Monitoring Center",
    page_icon="✈️",
    layout="wide"
)

# Load Model Checkpoint
@st.cache_resource
def load_multitask_model(checkpoint_path: str):
    config = MultiTaskModelConfig()
    model = MultiTaskYOLO(config)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded trained model weights from: {checkpoint_path}")
    model.eval()
    return model

MODEL_PATH = "model/checkpoints/best_multitask_model.pt"
model = load_multitask_model(MODEL_PATH)

if "decision_engine" not in st.session_state:
    st.session_state.decision_engine = DecisionEngine()
if "storage_manager" not in st.session_state:
    st.session_state.storage_manager = StorageManager()

# Class mapping configurations
CLASS_NAMES = {
    "turnaround": ['aircraft', 'baggage_truck', 'bridge_connected', 'bus', 'catering_truck', 'fuel_truck', 'fueling', 'ground_power', 'person', 'pushback_tractor', 'ramp_loader', 'rolling_stairway', 'stairway'],
    "ppe": ["ear_protector", "safety_vest"],
    "fod": ["debris_object"]
}

# -----------------------------------------------------------------------------
# UI LAYOUT SETUP
# -----------------------------------------------------------------------------
st.title("✈️ Airside Operational & Safety Incident Monitoring Center")
st.caption("Real-Time Multi-Steram Deep Leanring & Spatial Temporal Analytics Dashboard")

st.sidebar.header("🕹️ Stream Simulation Controls")
run_pipeline = st.sidebar.checkbox("Run Live Multi-Stream Video Feeds", value=False)
conf_thresh = st.sidebar.slider("Detection Confidence Threshold", 0.10, 1.00, 0.35)

# Columns for 3 video channels
col_ta, col_ppe, col_fod = st.columns(3)

with col_ta:
    st.subheader("📹 Panel 1: Turnaround Stream")
    ta_video = st.empty()
    ta_status = st.empty()

with col_ppe:
    st.subheader("📹 Panel 2: PPE Compliance Guard")
    ppe_video = st.empty()
    ppe_metric = st.empty()

with col_fod:
    st.subheader("📹 Panel 3: FOD Anomaly Detector")
    fod_video = st.empty()
    fod_status = st.empty()

st.markdown("---")
st.subheader("🗄️ Live Incident & Event Log Table")
table_placeholder = st.empty()

# Sample Video File Paths
VIDEO_PATHS = {
    "turnaround": "data/samples/turnaround_sample.mp4",
    "ppe": "data/samples/ppe_sample.mp4",
    "fod": "data/samples/fod_sample.mp4"
}

# Helper to run inference on single frame
def run_stream_inference(frame: np.ndarray, task_id: int, conf_threshold: float):
    # Preprocess to 640x640 tensor
    h, w = frame.shape[:2]
    resized_img = cv2.resize(frame, (640, 640))
    tensor_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB.transpose(2, 0, 1))
    input_tensor = torch.from_numpy(tensor_img).float().unsqueeze(0) / 255.0
    task_tensor = torch.tensor([task_id], dtype=torch.long)

    with torch.no_grad():
        preds = model(input_tensor, task_tensor)

    detections = []
    # Parse predictions output (format: [batch, num_dets, 6] -> x1, y1, x2, y2, conf, cls)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if preds is not None and len(preds) > 0 and len(preds.shape) == 3:
        for det in preds[0]:
            conf = float(det[4])
            if conf >= conf_threshold:
                # Rescale BBoxes back to original frame dimensions
                x1 = int(float(det[0]) * w / 640.0)
                y1 = int(float(det[1]) * h / 640.0)
                x2 = int(float(det[2]) * w / 640.0)
                y2 = int(float(det[3]) * h / 640.0)
                cls_id = int(float(det[5]))

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id
                })
    return detections

# -----------------------------------------------------------------------------
# MAIN VIDEO LOOP
# -----------------------------------------------------------------------------
if run_pipeline:
    caps = {
        "turnaround": cv2.VideoCapture(VIDEO_PATHS["turnaround"]),
        "ppe": cv2.VideoCapture(VIDEO_PATHS["ppe"]),
        "fod": cv2.VideoCapture(VIDEO_PATHS["fod"])
    }

    while run_pipeline:
        frames = {}
        for key in caps:
            ret, frame = caps[key].read()
            if not ret:
                # Loop sample video if it ends
                caps[key].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = caps[key].read()
            frames[key] = frame
        
        # ---------------------------------------------------------------------
        # 1. PROCESS TURNAROUND STREAM (Task ID 0)
        # ---------------------------------------------------------------------
        f_ta = frames["turnaround"].copy()
        dets_ta = run_stream_inference(f_ta, task_id=0, conf_threshold=conf_thresh)
        res_ta = st.session_state.decision_engine.evaluate_turnaround_stream(dets_ta, CLASS_NAMES["turnaround"])

        # Draw Turnaround Detections
        for det in dets_ta:
            x1, y1, x2, y2 = det["box"]
            cls_name = CLASS_NAMES["turnaround"][det["class_id"]] if det["class_id"] < len(CLASS_NAMES["turnaround"]) else "vehicle"
            conf = det["confidence"]
            color = (255, 165, 0)   # Orange
            cv2.rectangle(f_ta, (x1, y1), (x2, y2), color, 2)
            cv2.putText(f_ta, f"{cls_name} {conf:.2f}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if res_ta["status"] == "VIOLATION":
            cv2.putText(f_ta, "⚠️ PROXIMITY BREACH", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            st.session_state.storage_manager.process_and_store_event(
                stream_source="Turnaround",
                event_type="Proximity Violation",
                status_value="GSE Hull Breach",
                action_trigger="Proximity Warning",
                frame=f_ta
            )
        
        # ---------------------------------------------------------------------
        # 2. PROCESS PPE COMPLIANCE STREAM (Task ID 1)
        # ---------------------------------------------------------------------
        f_ppe = frames["ppe"].copy()
        dets_ppe = run_stream_inference(f_ppe, task_id=1, conf_threshold=conf_thresh)
        person_dets = [d for d in dets_ta if CLASS_NAMES["turnaround"][d["class_id"]].lower() == "person"]
        res_ppe = st.session_state.decision_engine.evaluate_ppe_stream(person_dets, dets_ppe, CLASS_NAMES["ppe"])

        for person in res_ppe["person_details"]:
            px1, py1, px2, py2 = person["box"]
            is_comp = person["compliant"]
            color = (0, 255, 0) if is_comp else (0, 0, 255) # Green or Red
            cv2.rectangle(f_ppe, (px1, py1), (px2, py2), color, 2)

            label = "COMPLIANT" if is_comp else f"{person['missing'][0] if person['missing'] else 'NON-COMPLIANT'}"
            cv2.putText(f_ppe, label, (px1, max(15, py1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if res_ppe["status"] == "VIOLATIONS":
            st.session_state.storage_manager.process_and_store_event(
                stream_source="PPE Guard",
                event_type="Missing Equipment",
                status_value="Non-Compliant Worker",
                action_trigger="Cached Frame / Red Alert",
                frame=f_ppe
            )

        # ---------------------------------------------------------------------
        # 3. PROCESS FOD STREAM (Task ID 2)
        # ---------------------------------------------------------------------
        f_fod = frames["fod"].copy()
        dets_fod = run_stream_inference(f_fod, task_id=2, conf_threshold=conf_thresh)
        res_fod = st.session_state.decision_engine.evaluate_fod_stream(f_fod, dets_fod,)

        for dets in res_fod["verified_debris"]:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cv2.rectangle(f_fod, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(f_fod, f"HAZARD: DEBRIS {conf:.2f}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if res_fod["status"] == "VIOLATIONS":
            cv2.putText(f_fod, "🚨 DEBRIS DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            st.session_state.storage_manager.process_and_store_event(
                stream_source="FOD Detector",
                event_type="Tarmac Hazard",
                status_value="FOD Object Found",
                action_trigger="Critical Lock Alert",
                frame=f_fod
            )
        
        # Render Video Displays
        ta_video.image(f_ta, channels="BGR", use_column_width=True)
        ppe_video.image(f_ppe, channels="BGR", use_column_width=True)
        fod_video.image(f_fod, channels="BGR", use_column_width=True)

        # Render Widgets & Indicators
        ta_status.info(f"Turnaround State: **{res_ta['status']}**")
        ppe_metric.metrics("Safety Compliance Rate", f"{res_ppe['compliance_rate']}%")

        if res_fod["status"] == "VIOLATIONA":
            fod_status.error("🚨 CRITICAL HAZARD: DEBRIS ON TARMAC")
        else:
            fod_status.success("✅ SURFACE CLEAR")
        
        # Update Incident Table
        recent_logs = st.session_state.storage_manager.cold_logger.fetch_recent_logs(limit=10)
        if len(recent_logs) > 0:
            df_logs = pd.DataFrame(recent_logs)
            df_logs = df_logs[["timestamp", "stream_source", "event_type", "status_value", "action_trigger"]]
            df_logs.columns = ["Timestamp", "Stream Source", "Event Type", "Status / Value", "Action / Trigger"]
            table_placeholder.dataframe(df_logs, use_container_width=True)

        time.sleep(0.03)

    for cap in caps.values():
        cap.release()
