from pathlib import Path
import csv
import random
from collections import defaultdict


INPUT_CSV = Path("labels.csv")
TRAIN_CSV = Path("train.csv")
VAL_CSV = Path("val.csv")
TEST_CSV = Path("test.csv")


def stratified_split(rows, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)

    train_rows, val_rows, test_rows = [], [], []

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_rows.extend(items[:n_train])
        val_rows.extend(items[n_train:n_train + n_val])
        test_rows.extend(items[n_train + n_val:])

    return train_rows, val_rows, test_rows


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    with INPUT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = reader.fieldnames or [
        "image_path",
        "plant_pixels",
        "river_pixels",
        "plant_fraction",
        "label",
    ]

    train_rows, val_rows, test_rows = stratified_split(rows)

    write_csv(TRAIN_CSV, fieldnames, train_rows)
    write_csv(VAL_CSV, fieldnames, val_rows)
    write_csv(TEST_CSV, fieldnames, test_rows)

    def count_by_label(subrows):
        counts = defaultdict(int)
        for r in subrows:
            counts[r["label"]] += 1
        return counts

    print("Stratified splits created from labels.csv")
    for name, subset in [("Train", train_rows), ("Val", val_rows), ("Test", test_rows)]:
        counts = count_by_label(subset)
        total = len(subset)
        print(f"\n{name} ({total} images):")
        for label, cnt in sorted(counts.items()):
            pct = (cnt / total * 100) if total else 0
            print(f"  {label}: {cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
