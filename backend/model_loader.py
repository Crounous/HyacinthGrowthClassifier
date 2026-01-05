import io
import json
import os
from typing import List

from PIL import Image
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

DEFAULT_LABELS = ["No Growth", "Low Growth", "Moderate Growth", "Large Growth"]


def _load_labels_for_model(model_path: str) -> List[str]:
    labels_path = os.path.splitext(model_path)[0] + ".labels.json"
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
            return labels
    except Exception:
        pass
    return DEFAULT_LABELS


def _preprocess_pil(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((256, 256), resample=Image.BILINEAR)
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, axis=0)  # NCHW
    return arr


def _build_torch_model(num_classes: int):
    import torch
    from torchvision import models
    from torch import nn

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class PlantModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.labels: List[str] = _load_labels_for_model(model_path)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        self._mode = "onnx" if model_path.lower().endswith(".onnx") else "pth"

        if self._mode == "onnx":
            if ort is None:
                raise RuntimeError(
                    "onnxruntime is not installed, but an .onnx model was provided. "
                    "Install onnxruntime or use a .pth checkpoint."
                )
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            return

        import torch
        from torchvision import transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)

        checkpoint_labels: List[str] | None = None
        if isinstance(checkpoint, dict):
            checkpoint_labels = checkpoint.get("labels")
        if checkpoint_labels:
            self.labels = checkpoint_labels
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        self.model = _build_torch_model(num_classes=len(self.labels))

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

        if getattr(self, "_mode", "pth") == "onnx":
            inp = _preprocess_pil(image)
            outputs = self.session.run([self.output_name], {self.input_name: inp})[0]
            label_idx = int(np.argmax(outputs, axis=1)[0])
            return self.labels[label_idx]

        import torch

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)

        label_idx = predicted.item()
        return self.labels[label_idx]
