import onnxruntime as ort
import numpy as np
import cv2

class tensorRTEdgeEngine:
    """
    High-performance Edge Inference Runner using ONNX Runtime
    with TensorRT / CUDA Provider.
    """
    def __init__(self, onnx_model_path: str):
        print(f"⚡ Loading ONNX model for Edge Acceleration...")

        # Enable TensorRT and CUDA execution providers for edge speedup
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.task_input_name = self.session.get_inputs()[1].name

        print(f"✅ Edge Engine Ready! Active Providers: {self.session.get_providers()}")

    def infer(self, frame: np.ndarray, task_id: int) -> np.ndarray:
        # Preprocess frame to NCHW [1, 3, 640, 640]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(tensor, axis=0)
        
        task_tensor = np.array([task_id], dtype=np.int64)

        # Run ONNX / TensorRT forward pass
        outputs = self.session.run(
            None,
            {self.input_name: input_tensor, self.task_input_name: task_tensor}
        )
        return outputs[0]
        