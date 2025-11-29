from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from train_model import build_model, LABELS, LABEL_TO_IDX, MODEL_OUT, DATA_DIR

TEST_CSV = Path("test.csv")
TRIAL1_CSV = Path("test_trial1.csv")
TRIAL2_CSV = Path("test_trial2.csv")
TRIAL3_CSV = Path("test_trial3.csv")


class PlantTestDataset(Dataset):
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


def get_test_loader(csv_path: Path, batch_size: int = 32, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds = PlantTestDataset(csv_path, DATA_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader


def load_trained_model(device: torch.device):
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model = build_model(num_classes=len(LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_split(name: str, csv_path: Path, metrics_txt: Path,
                   cm_png: Path, per_class_png: Path,
                   batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Evaluating {name} ({csv_path}) ===")
    print("Using device:", device)

    test_loader = get_test_loader(csv_path, batch_size=batch_size)
    model = load_trained_model(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    # Compute metrics
    report = classification_report(all_labels, all_preds, target_names=LABELS, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    # Save metrics to text file
    with metrics_txt.open("w", encoding="utf-8") as f:
        f.write(f"Classification report ({name})\n")
        f.write(report)
        f.write("\n\nConfusion matrix (rows=true, cols=pred)\n")
        f.write(str(cm))

    print("Metrics written to", metrics_txt)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=LABELS, yticklabels=LABELS,
           ylabel="True label",
           xlabel="Predicted label",
           title=f"Confusion Matrix ({name})")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(cm_png, dpi=150)
    plt.close(fig)

    # Parse per-class metrics from classification_report
    # classification_report with output_dict=True gives a dict we can use
    report_dict = classification_report(all_labels, all_preds,
                                        target_names=LABELS,
                                        digits=4, output_dict=True)

    precisions = [report_dict[label]["precision"] for label in LABELS]
    recalls = [report_dict[label]["recall"] for label in LABELS]
    f1s = [report_dict[label]["f1-score"] for label in LABELS]

    x = np.arange(len(LABELS))
    width = 0.25

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x - width, precisions, width, label="Precision")
    ax2.bar(x, recalls, width, label="Recall")
    ax2.bar(x + width, f1s, width, label="F1-score")

    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, rotation=45, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title(f"Per-class Metrics ({name})")
    ax2.legend()

    fig2.tight_layout()
    fig2.savefig(per_class_png, dpi=150)
    plt.close(fig2)

    print("Saved:")
    print(" ", cm_png)
    print(" ", per_class_png)
    return None


if __name__ == "__main__":
    def run_split(name, csv_path, metrics_name):
        if not csv_path.exists():
            return
        base = Path(metrics_name).stem.replace("metrics_", "")
        evaluate_split(
            name=name,
            csv_path=csv_path,
            metrics_txt=Path(metrics_name),
            cm_png=Path(f"confusion_matrix_{base}.png"),
            per_class_png=Path(f"per_class_metrics_{base}.png"),
        )

    # Full test set
    run_split("Test", TEST_CSV, "metrics_test.txt")

    # Three trials of 50 images each
    run_split("Trial 1", TRIAL1_CSV, "metrics_trial1.txt")
    run_split("Trial 2", TRIAL2_CSV, "metrics_trial2.txt")
    run_split("Trial 3", TRIAL3_CSV, "metrics_trial3.txt")
