from pathlib import Path
from PIL import Image, ImageFilter

ROOT_CONFIGS = [
    (
        Path("Plant dataset") / "trials",
        Path("Plant dataset") / "trials_blur",
        Path("Plant dataset") / "trials_lowres",
    ),
    (
        Path("trials2"),
        Path("trials2_blur"),
        Path("trials2_lowres"),
    ),
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
GAUSSIAN_PERCENT = 0.10  # 10% blur strength relative to shorter image side
GAUSSIAN_RADIUS_CAP = 50  # keep massive images from using extreme radii
LOWRES_SCALE = 0.5        # resize to 50% to ensure <300 PPI equivalent
LOWRES_DPI = (200, 200)   # explicit DPI metadata (<300)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def make_blurred(img: Image.Image) -> Image.Image:
    width, height = img.size
    radius = int(min(width, height) * GAUSSIAN_PERCENT)
    radius = max(1, min(radius, GAUSSIAN_RADIUS_CAP))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_lowres(img: Image.Image) -> Image.Image:
    width, height = img.size
    new_size = (
        max(1, int(width * LOWRES_SCALE)),
        max(1, int(height * LOWRES_SCALE)),
    )
    return img.resize(new_size, Image.LANCZOS)


def process_image(src: Path, blur_dest: Path, lowres_dest: Path) -> None:
    blur_dest.parent.mkdir(parents=True, exist_ok=True)
    lowres_dest.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = im.convert("RGB")
        blurred = make_blurred(im)
        blurred.save(blur_dest)

        lowres = make_lowres(im)
        lowres.save(lowres_dest, dpi=LOWRES_DPI)


def process_root(src_root: Path, blur_root: Path, lowres_root: Path) -> None:
    if not src_root.exists():
        print(f"Skipping {src_root} (not found)")
        return

    blur_root.mkdir(parents=True, exist_ok=True)
    lowres_root.mkdir(parents=True, exist_ok=True)

    trial_dirs = [p for p in src_root.glob("trial*") if p.is_dir()]
    if not trial_dirs:
        print(f"No trial folders found under {src_root}")
        return

    for trial_dir in sorted(trial_dirs):
        count, missing = process_trial(trial_dir, src_root, blur_root, lowres_root)
        print(
            f"{src_root.name}/{trial_dir.name}: "
            f"generated {count} blurred + low-res pairs"
        )
        if missing:
            print(f"  Skipped {missing} missing files")


def process_trial(trial_dir: Path, src_root: Path, blur_root: Path,
                  lowres_root: Path) -> tuple[int, int]:
    count = 0
    missing = 0
    for path in trial_dir.rglob("*"):
        if path.is_file() and is_image(path):
            relative = path.relative_to(src_root)
            blur_dest = blur_root / relative
            lowres_dest = lowres_root / relative
            try:
                process_image(path, blur_dest, lowres_dest)
                count += 1
            except FileNotFoundError:
                missing += 1
    return count, missing


def main():
    for src_root, blur_root, lowres_root in ROOT_CONFIGS:
        process_root(src_root, blur_root, lowres_root)


if __name__ == "__main__":
    main()
