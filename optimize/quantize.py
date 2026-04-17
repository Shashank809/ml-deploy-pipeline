from onnxruntime.quantization import quantize_dynamic, QuantType

print("Starting quantization...")

quantize_dynamic(
    "models/model.onnx",
    "models/model_int8.onnx",
    weight_type=QuantType.QInt8,
    optimize_model=False 
)

print("Quantized model saved as models/model_int8.onnx")
