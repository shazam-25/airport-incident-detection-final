import torch
import os
from src.models.network import MultiTaskYOLO
from src.models.config import MultiTaskModelConfig

def export_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str = "checkpoints/best_multitask_model.onnx",
    img_size: int = 640
):
    print(f"🚀 Initializing model export to ONNX format...")

    # 1. Load trained model architecture
    config = MultiTaskModelConfig()
    model = MultiTaskYOLO()

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded weights from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")
    
    model.eval()
    # 2. Prepare dummy inputs matching model inputs (Frame [1, 3, 640, 640] & Task ID [1])
    dummy_frame = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    dummy_task_id = torch.tensor([0], dtype=torch.long)

    # 3. Export to ONNX
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_frame, dummy_task_id),
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_frame", "task_id"],
        output_names=["predictions"],
        dynamic_axes={
            "input_frame": {0: "batch_size"},
            "predictions": {0: "batch_size"}
        }
    )

    print(f"🎉 Successfully exported ONNX model to: {output_onnx_path}")

if __name__ == "__main__":
    export_to_onnx(checkpoint_path="model/checkpoints/best_multitask_model.pt")