import onnxruntime as ort
import numpy as np

print("Validating models...")

fp32 = ort.InferenceSession("models/model.onnx")
int8 = ort.InferenceSession("models/model_int8.onnx")

input_name = fp32.get_inputs()[0].name

data = np.random.randn(1, 3, 224, 224).astype(np.float32)

out1 = fp32.run(None, {input_name: data})
out2 = int8.run(None, {input_name: data})

diff = np.mean(np.abs(out1[0] - out2[0]))

print("Difference:", diff)

if diff < 0.1:
    print("✅ Validation passed")
else:
    print("⚠️ Accuracy drop detected")