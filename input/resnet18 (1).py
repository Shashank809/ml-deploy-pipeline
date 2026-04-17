import torch
import torchvision.models as models

print("Loading ResNet18...")

model = models.resnet18(pretrained=True)
model.eval()

torch.save(model, "model.pt")

print("Saved as model.pt")