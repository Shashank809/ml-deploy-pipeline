import torch
import sys
import os

model_path = sys.argv[1]

print("Converting to ONNX:", model_path)

# FIXED LINE
model = torch.load(model_path, weights_only=False)

model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

os.makedirs("models", exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    input_names=["input"],
    output_names=["output"]
)

print("ONNX model saved at models/model.onnx")
