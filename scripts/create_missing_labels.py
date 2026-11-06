"""Create empty label .txt files for images that have no corresponding label."""

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(
        description="Create empty label files for images without a matching .txt "
        "(YOLO treats these as background/negative images)."
    )
    parser.add_argument("--images", required=True, help="folder containing images")
    parser.add_argument("--labels", required=True, help="folder containing label .txt files")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    if not images_dir.is_dir():
        print(f"Images folder not found: {images_dir}")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / (image_path.stem + ".txt")
        if label_path.exists():
            continue

        label_path.touch()
        created += 1
        print(f"Created: {label_path.name}")

    print(f"\nDone. {created} label file(s) created.")


if __name__ == "__main__":
    main()
