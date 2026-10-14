"""Build the YOLO dataset folder from the raw FASDD_UAV download.

FASDD_UAV ships as images/ + annotations/YOLO_UAV/{labels,train.txt,val.txt,test.txt}.
Class ids in the raw labels: 0 = fire, 1 = smoke (we keep that order everywhere).

Split policy
- test : the official FASDD test split, untouched. Only used for final numbers.
- val  : a fixed random subset of the official val split (default 1500 images),
         so each epoch's validation stays fast.
- train: official train + the rest of official val (more aerial data for free).

Images are symlinked, labels are copied, so the output costs almost no disk.

    python scripts/prepare_fasdd.py --raw ~/datasets/fasdd_uav --out ~/datasets/daedalus_fire
"""
import argparse
import os
import random
import shutil
from pathlib import Path


def read_split(path: Path) -> list[str]:
    # entries look like "./images/fire_UAV000123.jpg"; keep only the file name
    return [Path(line.strip()).name for line in path.read_text().splitlines() if line.strip()]


def link_split(names, split, img_dir: Path, lbl_dir: Path, out: Path) -> int:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        src = img_dir / name
        if not src.exists():
            print(f"  missing image, skipped: {name}")
            continue
        dst = out / "images" / split / name
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        lbl = lbl_dir / (Path(name).stem + ".txt")
        # images without a label file are negatives -> empty label file
        target = out / "labels" / split / lbl.name
        if lbl.exists():
            shutil.copyfile(lbl, target)
        else:
            target.write_text("")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("~/datasets/fasdd_uav").expanduser())
    ap.add_argument("--out", type=Path, default=Path("~/datasets/daedalus_fire").expanduser())
    ap.add_argument("--val-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann = args.raw / "ann" / "annotations" / "YOLO_UAV"
    img_dir = args.raw / "images"
    train = read_split(ann / "train.txt")
    val = read_split(ann / "val.txt")
    test = read_split(ann / "test.txt")

    rng = random.Random(args.seed)
    rng.shuffle(val)
    val_keep, val_to_train = val[: args.val_size], val[args.val_size :]

    for split, names in [("train", train + val_to_train), ("val", val_keep), ("test", test)]:
        n = link_split(names, split, img_dir, ann / "labels", args.out)
        print(f"{split:5s}: {n} images")


if __name__ == "__main__":
    main()
