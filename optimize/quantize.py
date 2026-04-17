import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

print("Starting quantization...")

model = onnx.load("models/model.onnx")

onnx.save(model, "models/model_clean.onnx")

quantize_dynamic(
    "models/model_clean.onnx",
    "models/model_int8.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved as models/model_int8.onnx")
