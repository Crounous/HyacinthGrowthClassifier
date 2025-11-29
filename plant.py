from pathlib import Path
import os

import cv2
import numpy as np
import csv
import random


DATA_DIR = Path("Plant dataset")
OUTPUT_CSV = Path("labels.csv")


def count_plant_and_river_pixels(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0, 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    height, width = hsv.shape[:2]
    total_pixels = height * width

    # Rough plant (green) HSV range
    plant_lower = np.array([35, 40, 40])
    plant_upper = np.array([85, 255, 255])
    plant_mask = cv2.inRange(hsv, plant_lower, plant_upper)
    plant_pixels = int(np.count_nonzero(plant_mask))

    # Rough river (blue/cyan) HSV range
    river_lower = np.array([90, 40, 40])
    river_upper = np.array([140, 255, 255])
    river_mask = cv2.inRange(hsv, river_lower, river_upper)
    river_pixels = int(np.count_nonzero(river_mask))

    return plant_pixels, river_pixels, total_pixels


def plant_fraction(plant_pixels: int, total_pixels: int) -> float:
    if total_pixels == 0:
        return 0.0
    return plant_pixels / total_pixels


def growth_label_from_ratio(r: float) -> str:
    if r < 0.05:
        return "No Growth"
    elif r < 0.15:
        return "Low Growth"
    elif r < 0.40:
        return "Moderate Growth"
    else:
        return "Large Growth"


def main():
    rows = []
    count = 0
    summary = {"No Growth": 0, "Low Growth": 0, "Moderate Growth": 0, "Large Growth": 0}
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                img_path = Path(root) / fname
                count += 1
                print(f"Processing {count}: {img_path}")
                plant_pixels, river_pixels, total_pixels = count_plant_and_river_pixels(img_path)
                ratio = plant_fraction(plant_pixels, total_pixels)
                label = growth_label_from_ratio(ratio)
                if label in summary:
                    summary[label] += 1
                rel_path = img_path.relative_to(DATA_DIR)
                rows.append((str(rel_path), plant_pixels, river_pixels, ratio, label))

    with OUTPUT_CSV.open("w", encoding="utf-8") as f:
        f.write("image_path,plant_pixels,river_pixels,plant_fraction,label\n")
        for path, pp, rp, r, label in rows:
            f.write(f"{path},{pp},{rp},{r:.4f},{label}\n")

    print("\nSummary:")
    total = sum(summary.values())
    for label, cnt in summary.items():
        print(f"{label}: {cnt} images")
    print(f"Total: {total} images")


def create_splits(input_csv: Path, train_csv: Path, val_csv: Path, test_csv: Path,
                  train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                  seed: int = 42) -> None:
    rows = []
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    random.Random(seed).shuffle(rows)

    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # ensure all samples are used
    n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    fieldnames = ["image_path", "plant_pixels", "river_pixels", "plant_fraction", "label"]

    def write_split(path: Path, split_rows):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    write_split(train_csv, train_rows)
    write_split(val_csv, val_rows)
    write_split(test_csv, test_rows)

    print("\nData splits created:")
    print(f"Train: {len(train_rows)} images -> {train_csv}")
    print(f"Val:   {len(val_rows)} images -> {val_csv}")
    print(f"Test:  {len(test_rows)} images -> {test_csv}")


if __name__ == "__main__":
    main()
    create_splits(
        INPUT_CSV := OUTPUT_CSV,
        train_csv=Path("train.csv"),
        val_csv=Path("val.csv"),
        test_csv=Path("test.csv"),
    )

from pathlib import Path
import os
import cv2
import numpy as np

DATA_DIR = Path("Plant dataset")
OUTPUT_CSV = Path("labels.csv")

def count_plant_and_river_pixels(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rough "plant" (green) HSV range – tweak as needed
    plant_lower = np.array([35, 40, 40])
    plant_upper = np.array([85, 255, 255])
    plant_mask = cv2.inRange(hsv, plant_lower, plant_upper)
    plant_pixels = int(np.count_nonzero(plant_mask))

    # Rough "river" (blue/cyan) HSV range – tweak as needed
    river_lower = np.array([90, 40, 40])
    river_upper = np.array([140, 255, 255])
    river_mask = cv2.inRange(hsv, river_lower, river_upper)
    river_pixels = int(np.count_nonzero(river_mask))

    return plant_pixels, river_pixels

def growth_ratio(plant_pixels: int, river_pixels: int) -> float:
    if river_pixels == 0:
        return 0.0  # or np.nan if you prefer to skip these
    return plant_pixels / river_pixels

def growth_label_from_ratio(r: float) -> str:
    if r < 0.05:
        return "No Growth"
    elif r < 0.15:
        return "Low Growth"
    elif r < 0.40:
        return "Moderate Growth"
    else:
        return "Large Growth"

def main():
    rows = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                img_path = Path(root) / fname
                plant_pixels, river_pixels = count_plant_and_river_pixels(img_path)
                ratio = growth_ratio(plant_pixels, river_pixels)
                label = growth_label_from_ratio(ratio)
                rel_path = img_path.relative_to(DATA_DIR)
                rows.append((str(rel_path), plant_pixels, river_pixels, ratio, label))

    with OUTPUT_CSV.open("w", encoding="utf-8") as f:
        f.write("image_path,plant_pixels,river_pixels,ratio,label\n")
        for path, pp, rp, r, label in rows:
            f.write(f"{path},{pp},{rp},{r:.4f},{label}\n")

if __name__ == "__main__":
    main()