"""Add a ground-level D-Fire subset to the dataset.

Why: the ground demo shows the camera *photos* of fires (on a screen or printed),
and many of those will be ground or oblique shots, not top-down drone views.
A slice of D-Fire keeps the model from only knowing what FASDD_UAV looks like.

D-Fire class ids are 0 = smoke, 1 = fire -> swapped here to our 0 = fire, 1 = smoke.

Source: Hugging Face mirror badsaarow/d-fire (parquet: image bytes + YOLO label text).
Original: https://github.com/gaiasd/DFireDataset (check its license before redistributing).

    python scripts/prepare_dfire.py --raw ~/datasets/dfire/raw --out ~/datasets/daedalus_fire --n-train 4000
"""
import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq

SWAP = {"0": "1", "1": "0"}


def rows(files):
    for f in files:
        table = pq.read_table(f, columns=["image", "label", "filename"])
        for r in table.to_pylist():
            yield r


def write(r, split, out: Path):
    name = "dfire_" + Path(r["filename"]).stem
    img = r["image"]["bytes"]
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "images" / split / f"{name}.jpg").write_bytes(img)
    lines = []
    for line in (r["label"] or "").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in SWAP:
            lines.append(" ".join([SWAP[parts[0]]] + parts[1:]))
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("~/datasets/dfire/raw").expanduser())
    ap.add_argument("--out", type=Path, default=Path("~/datasets/daedalus_fire").expanduser())
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # reservoir-free: shards are small enough to sample with a keep-probability per row
    for split, pattern, n in [("train", "train-*.parquet", args.n_train), ("test_ground", "test-*.parquet", args.n_test)]:
        files = sorted((args.raw / "data").glob(pattern))
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        p = min(1.0, n / max(total, 1))
        kept = 0
        for r in rows(files):
            if kept < n and rng.random() < p * 1.05:
                write(r, split, args.out)
                kept += 1
        print(f"{split}: kept {kept} of {total} D-Fire images")


if __name__ == "__main__":
    main()
