import torch
import torchvision.models as models

print("Loading ResNet18...")

model = models.resnet18(pretrained=True)
model.eval()

# ✅ SAVE ONLY WEIGHTS
torch.save(model.state_dict(), "model.pt")

print("Saved weights as model.pt")
