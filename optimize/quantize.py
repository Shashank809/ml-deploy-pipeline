from onnxruntime.quantization import quantize_dynamic, QuantType

print("Starting quantization...")

quantize_dynamic(
    model_input="models/model.onnx",
    model_output="models/model_int8.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved as models/model_int8.onnx")
