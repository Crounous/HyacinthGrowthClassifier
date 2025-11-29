import io
from typing import List

import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

DEFAULT_LABELS = ["No Growth", "Low Growth", "Moderate Growth", "Large Growth"]


def build_model(num_classes: int = 4):
    model = models.resnet50(weights=None)  # Weights are loaded from checkpoint
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class PlantModel:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)

        checkpoint_labels: List[str] | None = None
        if isinstance(checkpoint, dict):
            checkpoint_labels = checkpoint.get("labels")

        self.labels: List[str] = checkpoint_labels or DEFAULT_LABELS
        self.label_to_idx = {
            label: idx for idx, label in enumerate(self.labels)
        }

        self.model = build_model(num_classes=len(self.labels))

        try:
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
        except Exception as exc:
            print(f"Error loading model: {exc}")
            raise exc

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            
        label_idx = predicted.item()
        return self.labels[label_idx]
