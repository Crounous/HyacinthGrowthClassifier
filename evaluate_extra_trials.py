from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.ticker import PercentFormatter
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_model import build_model, LABELS, LABEL_TO_IDX, MODEL_OUT

TRIAL_GROUPS = [
    (
        "Combined Extra Trial",
        [
            ("Trial Extra 1", Path("trial_extra1.csv"), "trial_extra1"),
            ("Trial Extra 2", Path("trial_extra2.csv"), "trial_extra2"),
            ("Trial Extra 3", Path("trial_extra3.csv"), "trial_extra3"),
        ],
    )
]

VARIANTS = [
    ("Original", Path("trials2")),
    ("Blurred", Path("trials2_blur")),
    ("LowRes", Path("trials2_lowres")),
]

BATCH_SIZE = 16
NUM_WORKERS = 4
PLOTS_DIR = Path("evaluation_plots")
BASELINE_VARIANT = "Original"


class TrialDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, int]], transform=None):
        if not samples:
            raise ValueError("Empty trial dataset: no samples available")
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, trial_idx = self.samples[idx]
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, trial_idx


def build_trial_samples(csv_specs: list[tuple[str, Path, str]], variant_root: Path):
    samples: list[tuple[Path, int, int]] = []
    trial_names: list[str] = []
    for trial_idx, (trial_name, csv_path, folder_name) in enumerate(csv_specs):
        trial_names.append(trial_name)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        images_root = variant_root / folder_name
        if not images_root.exists():
            raise FileNotFoundError(f"Image folder not found: {images_root}")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = images_root / row["image_path"]
            label_name = row["label"]
            if label_name not in LABEL_TO_IDX:
                raise KeyError(f"Unknown label '{label_name}' in {csv_path}")
            label_idx = LABEL_TO_IDX[label_name]
            samples.append((img_path, label_idx, trial_idx))
    return samples, trial_names


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_model(device: torch.device):
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model = build_model(num_classes=len(LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_trial(trial_name: str, variant_name: str,
                   csv_specs: list[tuple[str, Path, str]], variant_root: Path,
                   model, device: torch.device):
    samples, trial_names = build_trial_samples(csv_specs, variant_root)
    dataset = TrialDataset(samples, transform=get_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    total = 0
    correct = 0
    per_trial = {
        idx: {
            "trial": trial_name_item,
            "variant": variant_name,
            "total": 0,
            "correct": 0,
            "seconds": 0.0,
        }
        for idx, trial_name_item in enumerate(trial_names)
    }

    start = time.perf_counter()
    with torch.no_grad():
        for images, labels, trial_indices in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            for trial_idx in trial_indices.unique().tolist():
                mask = trial_indices == trial_idx
                mask = mask.to(device)
                trial_total = mask.sum().item()
                if trial_total == 0:
                    continue
                trial_correct = preds[mask].eq(labels[mask]).sum().item()
                per_trial[int(trial_idx)]["total"] += trial_total
                per_trial[int(trial_idx)]["correct"] += trial_correct
    elapsed = time.perf_counter() - start

    incorrect = total - correct
    accuracy = correct / total if total else 0.0

    for stats in per_trial.values():
        stats_total = stats["total"]
        stats_correct = stats["correct"]
        stats["accuracy"] = stats_correct / stats_total if stats_total else 0.0
        stats["incorrect"] = stats_total - stats_correct
        stats["seconds"] = elapsed * (stats_total / total) if total else 0.0

    return {
        "trial": trial_name,
        "variant": variant_name,
        "csvs": [spec[1] for spec in csv_specs],
        "image_folders": [variant_root / spec[2] for spec in csv_specs],
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "seconds": elapsed,
        "per_trial": list(per_trial.values()),
    }


def plot_bar_chart(df: pd.DataFrame, value_col: str, title: str,
                   ylabel: str, filename: str, as_percent: bool = False):
    if df.empty:
        return None
    variant_order = [name for name, _ in VARIANTS]
    df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)
    grouped = df.groupby("variant", as_index=False)[value_col].mean()
    if grouped.empty:
        return None

    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped["variant"], grouped[value_col], color=["#4c78a8", "#72b7b2", "#f58518"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if as_percent:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    for bar, value in zip(bars, grouped[value_col]):
        display_val = value * 100 if as_percent else value
        suffix = "%" if as_percent else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{display_val:.2f}{suffix}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return out_path


def plot_reliability(df: pd.DataFrame, filename: str, title: str):
    if df.empty:
        return None
    accuracy_by_variant = df.groupby("variant", as_index=False)["accuracy"].mean()
    baseline_row = accuracy_by_variant.loc[accuracy_by_variant["variant"] == BASELINE_VARIANT]
    if baseline_row.empty or baseline_row.iloc[0]["accuracy"] == 0:
        print(f"Cannot compute reliability chart '{title}' (missing or zero baseline accuracy).")
        return None

    baseline = baseline_row.iloc[0]["accuracy"]
    accuracy_by_variant["reliability"] = (accuracy_by_variant["accuracy"] / baseline).clip(upper=1.0)
    accuracy_by_variant["variant"] = pd.Categorical(
        accuracy_by_variant["variant"],
        categories=[name for name, _ in VARIANTS],
        ordered=True,
    )
    accuracy_by_variant.sort_values("variant", inplace=True)

    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(accuracy_by_variant["variant"], accuracy_by_variant["reliability"],
                  color=["#4c78a8", "#72b7b2", "#f58518"])
    ax.set_title(title)
    ax.set_ylabel("Reliability (normalized to Original)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    for bar, value in zip(bars, accuracy_by_variant["reliability"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value * 100:.2f}%",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return out_path


def slugify(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or "trial"


def generate_graphs(results: list[dict]):
    if not results:
        print("No results available to generate graphs.")
        return

    df = pd.DataFrame(results)
    plot_bar_chart(df, "seconds", "Classification Time per Variant",
                   "Seconds", "classification_time.png")
    plot_bar_chart(df, "accuracy", "Diagnostic Accuracy per Variant",
                   "Accuracy", "diagnostic_accuracy.png", as_percent=True)
    plot_reliability(df, "reliability_image_quality.png",
                     "Reliability Across Image Quality Variants")

    per_trial_rows = []
    for entry in results:
        for trial_stats in entry.get("per_trial", []):
            per_trial_rows.append({
                "trial": trial_stats["trial"],
                "variant": trial_stats["variant"],
                "seconds": trial_stats["seconds"],
                "accuracy": trial_stats["accuracy"],
            })

    if not per_trial_rows:
        return

    per_trial_df = pd.DataFrame(per_trial_rows)
    for trial_name, subset in per_trial_df.groupby("trial"):
        slug = slugify(trial_name)
        plot_bar_chart(subset, "seconds",
                       f"{trial_name} Classification Time",
                       "Seconds", f"classification_time_{slug}.png")
        plot_bar_chart(subset, "accuracy",
                       f"{trial_name} Diagnostic Accuracy",
                       "Accuracy", f"diagnostic_accuracy_{slug}.png",
                       as_percent=True)
        plot_reliability(subset,
                          f"reliability_{slug}.png",
                          f"{trial_name} Reliability Across Image Quality Variants")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(device)

    results = []
    for variant_name, variant_root in VARIANTS:
        if not variant_root.exists():
            print(f"\nSkipping variant '{variant_name}': folder {variant_root} not found")
            continue
        for trial_name, csv_specs in TRIAL_GROUPS:
            try:
                result = evaluate_trial(
                    trial_name,
                    variant_name,
                    csv_specs,
                    variant_root,
                    model,
                    device,
                )
            except FileNotFoundError as exc:
                print(f"\nSkipping {trial_name} ({variant_name}): {exc}")
                continue

            results.append(result)
            display_name = f"{trial_name} [{variant_name}]"
            print(
                f"\n{display_name}:\n"
                f"  Total images: {result['total']}\n"
                f"  Correct: {result['correct']}\n"
                f"  Incorrect: {result['incorrect']}\n"
                f"  Accuracy: {result['accuracy']*100:.2f}%\n"
                f"  Classification time: {result['seconds']:.4f}s"
            )

    if results:
        print("\nSummary (all evaluated variants):")
        for variant_name in {r['variant'] for r in results}:
            subset = [r for r in results if r['variant'] == variant_name]
            avg_acc = sum(r['accuracy'] for r in subset) / len(subset)
            avg_time = sum(r['seconds'] for r in subset) / len(subset)
            print(
                f"  {variant_name}: avg acc {avg_acc*100:.2f}% | "
                f"avg time {avg_time:.4f}s"
            )
        generate_graphs(results)


if __name__ == "__main__":
    main()
