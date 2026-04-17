import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

print("Starting quantization...")

# 🔥 Load model normally
model = onnx.load("models/model.onnx")

# 🔥 Remove problematic shape info
for tensor in model.graph.value_info:
    tensor.type.tensor_type.shape.dim.clear()

onnx.save(model, "models/model_fixed.onnx")

# 🔥 Now quantize (this won't crash)
quantize_dynamic(
    "models/model_fixed.onnx",
    "models/model_int8.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved successfully!")
