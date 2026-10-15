"""Held-out "filmed off a screen" test split, for the ground demo condition.

Applies the same screen-photo degradation as the synthetic training data
(tilt, bezel, moire, glare, blur, JPEG) to *test* images with a different seed.
The model never trains on these: they only measure how well it copes with a
camera looking at a monitor, which the clean test split cannot show.

    python scripts/make_test_screen.py --n 1500
    python evaluate.py weights/daedalus_fire_n.pt     # picks up test_screen automatically
"""
import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import read_labels, resize_long, screen_photo, write_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("~/datasets/daedalus_fire").expanduser())
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    imgs = sorted((args.data / "images" / "test").glob("*.jpg"))
    rng.shuffle(imgs)
    (args.data / "images" / "test_screen").mkdir(parents=True, exist_ok=True)
    (args.data / "labels" / "test_screen").mkdir(parents=True, exist_ok=True)
    for p in imgs[: args.n]:
        im = resize_long(cv2.imread(str(p)), 1024)
        im, boxes = screen_photo(im, read_labels(args.data / "labels" / "test" / (p.stem + ".txt")), rng)
        cv2.imwrite(str(args.data / "images" / "test_screen" / p.name), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_labels(args.data / "labels" / "test_screen" / (p.stem + ".txt"), boxes)
    print(f"wrote {min(args.n, len(imgs))} screen-photo test images")


if __name__ == "__main__":
    main()
