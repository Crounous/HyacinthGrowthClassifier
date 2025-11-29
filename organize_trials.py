from pathlib import Path
import csv
import shutil

DATA_DIR = Path("Plant dataset")
OUTPUT_ROOT = DATA_DIR / "trials"
TRIAL_CSVS = {
    "trial1": Path("test_trial1.csv"),
    "trial2": Path("test_trial2.csv"),
    "trial3": Path("test_trial3.csv"),
    "trial_extra1": Path("trial_extra1.csv"),
    "trial_extra2": Path("trial_extra2.csv"),
    "trial_extra3": Path("trial_extra3.csv"),
}


def read_image_paths(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["image_path"] for row in reader]


def copy_images_for_trial(trial_name: str, csv_path: Path):
    rel_paths = read_image_paths(csv_path)
    dest_root = OUTPUT_ROOT / trial_name
    copied = 0
    missing = []

    for rel_path in rel_paths:
        src = DATA_DIR / rel_path
        if not src.exists():
            missing.append(rel_path)
            continue

        dest = dest_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1

    return copied, missing


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for trial_name, csv_path in TRIAL_CSVS.items():
        if not csv_path.exists():
            print(f"Skipping {trial_name}: missing {csv_path}")
            continue
        copied, missing = copy_images_for_trial(trial_name, csv_path)
        print(f"{trial_name}: copied {copied} images to {OUTPUT_ROOT / trial_name}")
        if missing:
            print(f"  Missing {len(missing)} files (listed below):")
            for rel_path in missing:
                print(f"    {rel_path}")


if __name__ == "__main__":
    main()
