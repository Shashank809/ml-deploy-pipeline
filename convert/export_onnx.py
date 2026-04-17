import torch
import torchvision.models as models
import sys
import os

model_path = sys.argv[1]

print("Converting to ONNX:", model_path)

model = models.resnet18()
model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

os.makedirs("models", exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=12   # 
)

print("ONNX model saved")
