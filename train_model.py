from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

DATA_DIR = Path("Plant dataset")
TRAIN_CSV = Path("train.csv")
VAL_CSV = Path("val.csv")
MODEL_OUT = Path("best_model.pth")

LABELS = ["No Growth", "Low Growth", "Moderate Growth", "Large Growth"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}


class PlantDataset(Dataset):
    def __init__(self, csv_path: Path, data_dir: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_str = row["label"]
        label = LABEL_TO_IDX[label_str]
        return img, label


def get_dataloaders(batch_size: int = 32, num_workers: int = 4):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = PlantDataset(TRAIN_CSV, DATA_DIR, transform=train_transform)
    val_ds = PlantDataset(VAL_CSV, DATA_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def build_model(num_classes: int = 4):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train(num_epochs: int = 20, lr: float = 1e-4, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = build_model(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total if val_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": LABEL_TO_IDX,
                "labels": LABELS,
            }, MODEL_OUT)
            print(f"  -> New best model saved (val acc={best_val_acc:.4f})")

    print("Training finished. Best val acc:", best_val_acc)


if __name__ == "__main__":
    train()
