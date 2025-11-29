from pathlib import Path
import csv
import random
from collections import defaultdict

# Input test split and output trial CSVs
TEST_CSV = Path("test.csv")
TRIAL1_CSV = Path("test_trial1.csv")
TRIAL2_CSV = Path("test_trial2.csv")
TRIAL3_CSV = Path("test_trial3.csv")

TRIAL_SIZE = 50  # images per trial
SEED = 123


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames


def stratified_trials(rows, labels_key="label", trial_size=50, n_trials=3, seed=123):
    """Create n_trials stratified subsets of size trial_size each.

    We sample approximately (trial_size / total) fraction from each class per trial.
    Remaining images stay unused.
    """
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[labels_key]].append(row)

    rng = random.Random(seed)
    for items in by_label.values():
        rng.shuffle(items)

    total = len(rows)
    if total == 0:
        raise ValueError("No rows found in test.csv")

    frac = trial_size / total

    # Precompute per-class counts per trial
    per_class_counts = {}  # label -> [n1, n2, n3]
    for label, items in by_label.items():
        n_label = len(items)
        # ideal per-trial count
        ideal = n_label * frac
        base = int(ideal)
        rem = n_label - base * n_trials
        counts = [base] * n_trials
        # distribute remainder
        for i in range(min(rem, n_trials)):
            counts[i] += 1
        per_class_counts[label] = counts

    # Now sample according to counts
    trials = [[] for _ in range(n_trials)]
    for label, items in by_label.items():
        start = 0
        for t in range(n_trials):
            k = per_class_counts[label][t]
            trials[t].extend(items[start:start + k])
            start += k

    # If some trials are off target by a few images due to rounding, it's okay.
    return trials


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_labels(rows, label_key="label"):
    counts = defaultdict(int)
    for r in rows:
        counts[r[label_key]] += 1
    return counts


def main():
    rows, fieldnames = load_rows(TEST_CSV)

    trials = stratified_trials(rows, trial_size=TRIAL_SIZE, n_trials=3, seed=SEED)

    out_paths = [TRIAL1_CSV, TRIAL2_CSV, TRIAL3_CSV]
    for i, (trial_rows, out_path) in enumerate(zip(trials, out_paths), start=1):
        write_csv(out_path, fieldnames, trial_rows)
        counts = count_labels(trial_rows)
        print(f"Trial {i}: {len(trial_rows)} images -> {out_path}")
        for label, c in sorted(counts.items()):
            print(f"  {label}: {c}")


if __name__ == "__main__":
    main()
