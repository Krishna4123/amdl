import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load MobileNetV2
mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
mobilenet.classifier = nn.Identity()  # remove last layer
mobilenet.eval()

if torch.cuda.is_available():
    mobilenet.cuda()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(image_bytes):
    img = Image.open(image_bytes).convert("RGB")
    img = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        img = img.cuda()

    with torch.no_grad():
        features = mobilenet(img)

    return features.cpu().numpy().flatten()
