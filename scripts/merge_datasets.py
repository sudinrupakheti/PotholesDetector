#!/usr/bin/env python3
"""Merge 'my_data_0', 'my_data_1' and 'my_data_2' into one YOLO dataset.

- Keeps the original train/val/test splits of my_data_0 intact.
- Randomly splits my_data_1 + my_data_2 into train/val/test at the same ratio.
- Renames everything to a continuous frame_nnnn scheme.
- Keeps empty label files (background/negative images).
- Writes a data.yaml with 2 classes.

Default paths are relative to this repository's root (the directory containing
this scripts/ folder); override with CLI flags.
"""

import argparse
import random
import shutil
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

DEFAULT_SRC0 = REPO_ROOT / "data" / "raw_datas" / "my_data_0"
DEFAULT_SRC1 = REPO_ROOT / "data" / "raw_datas" / "my_data_1"
DEFAULT_SRC2 = REPO_ROOT / "data" / "raw_datas" / "my_data_2"
DEFAULT_OUT = REPO_ROOT / "data" / "merged_data"

SEED = 42

# Ratio from the original my_data_0 splits: 1446/310/311 of 2067 -> ~70/15/15
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def collect(split_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted [(image, label)] pairs for a YOLO images/ directory."""
    imgs_dir = split_dir
    # Works for both layouts:
    #   my_data_0/images/train -> my_data_0/labels/train
    #   my_data_1/images       -> my_data_1/labels
    if split_dir.name == "images":
        labels_dir = split_dir.parent / "labels"
    else:
        labels_dir = split_dir.parent.parent / "labels" / split_dir.name
    pairs = []
    for img in sorted(imgs_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = labels_dir / (img.stem + ".txt")
        pairs.append((img, label))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src0", type=Path, default=DEFAULT_SRC0,
                        help="original dataset with train/val/test splits")
    parser.add_argument("--src1", type=Path, default=DEFAULT_SRC1,
                        help="first manually labeled dataset")
    parser.add_argument("--src2", type=Path, default=DEFAULT_SRC2,
                        help="second manually labeled dataset")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output merged dataset directory")
    args = parser.parse_args()

    rng = random.Random(SEED)

    # Rebuild destination
    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)

    counter = 0

    def place(img: Path, label: Path, split: str) -> None:
        nonlocal counter
        new_stem = f"frame_{counter:04d}"
        counter += 1
        shutil.copy2(img, args.out / "images" / split / f"{new_stem}{img.suffix.lower()}")
        dst_label = args.out / "labels" / split / f"{new_stem}.txt"
        if label.exists():
            shutil.copy2(label, dst_label)
        else:
            dst_label.touch()  # missing label -> empty (background image)

    # 1) Original my_data_0 splits kept intact
    for split in ("train", "val", "test"):
        for img, label in collect(args.src0 / "images" / split):
            place(img, label, split)

    # 2) New data: my_data_1 + my_data_2, random split 70/15/15
    new_pairs = collect(args.src1 / "images") + collect(args.src2 / "images")
    rng.shuffle(new_pairs)
    n = len(new_pairs)
    n_train = round(n * TRAIN_FRAC)
    n_val = round(n * VAL_FRAC)
    split_plan = (
        ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    )
    for (img, label), split in zip(new_pairs, split_plan):
        place(img, label, split)

    # 3) data.yaml with 2 classes
    (args.out / "data.yaml").write_text(
        f"path: {args.out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        'names: ["pothole", "not_pothole"]\n'
    )

    # 4) Report
    print(f"Merged dataset written to: {args.out}")
    for split in ("train", "val", "test"):
        n_img = len(list((args.out / "images" / split).iterdir()))
        n_lbl = len(list((args.out / "labels" / split).iterdir()))
        empty = sum(1 for f in (args.out / "labels" / split).iterdir() if f.stat().st_size == 0)
        print(f"  {split:5s}: {n_img:5d} images | {n_lbl:5d} labels | {empty:4d} empty (background)")
    print(f"Total images: {counter}")

    # Sanity checks: names unique, counts match, class ids valid
    stems = [f.stem for f in (args.out / "images").rglob("*.*")]
    assert len(stems) == len(set(stems)), "duplicate frame names!"
    bad = []
    for lbl in (args.out / "labels").rglob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.strip() and line.split()[0] not in {"0", "1"}:
                bad.append((lbl.name, line))
    assert not bad, f"unexpected class ids: {bad[:5]}"
    print("Sanity checks passed: unique names, counts match, class ids in {0,1}.")


if __name__ == "__main__":
    main()
