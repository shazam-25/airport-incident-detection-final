### Install packages
```bash
pip install -r requirements.txt
```

### Complile ONNX to TensorRT INT8 Engine
```bash
# Compile to FP16 TensorRT Engine
trtexec --onnx=checkpoints/best_multitask_model.onnx \
        --saveEngine=checkpoints/best_multitask_model_fp16.engine \
        --fp16

# Compile to INT8 Quantized TensorRT Engine for maximum FPS
trtexec --onnx=checkpoints/best_multitask_model.onnx \
        --saveEngine=checkpoints/best_multitask_model_int8.engine \
        --int8
```

### Run Streamlite Dashboard
```bash
streamlit run app.py
```