"""Score trained weights on the held-out real test splits.

    python evaluate.py runs/real/weights/best.pt runs/synth/weights/best.pt

Splits:
- test        : official FASDD_UAV test split (aerial, never trained on)
- test_ground : D-Fire test subset (ground-level), if prepared
- test_screen : test images made to look filmed off a monitor (scripts/make_test_screen.py)
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--data-root", type=Path, default=Path("~/datasets/daedalus_fire").expanduser())
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    results = {}
    for split in ("test", "test_ground", "test_screen"):
        if not (args.data_root / "images" / split).exists():
            continue
        yaml = Path("runs") / f"eval_{split}.yaml"
        yaml.write_text(f"path: {args.data_root}\ntrain: images/val\nval: images/{split}\nnames: {{0: fire, 1: smoke}}\n")
        for w in args.weights:
            m = YOLO(str(w)).val(data=str(yaml), imgsz=args.imgsz, batch=16, split="val", plots=False, verbose=False,
                                 project=str(Path("runs").resolve()), name=f"eval_{w.parent.parent.name}_{split}", exist_ok=True)
            r = {
                "mAP50": round(float(m.box.map50), 4),
                "mAP50-95": round(float(m.box.map), 4),
                "fire_AP50": round(float(m.box.ap50[0]), 4),
                "smoke_AP50": round(float(m.box.ap50[1]), 4),
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
            }
            results.setdefault(str(w), {})[split] = r
            print(f"{split:12s} {str(w):45s} {r}")
    Path("runs/eval_results.json").write_text(json.dumps(results, indent=2))
    print("saved runs/eval_results.json")


if __name__ == "__main__":
    main()
