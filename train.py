"""Fine-tune a small YOLO on fire/smoke.

    python train.py --name real                      # real data only (FASDD_UAV + D-Fire)
    python train.py --name synth --synthetic         # + copy-paste and screen-photo synthetic
    python train.py --name smoke --fraction 0.05 --epochs 1   # quick smoke test

Weights land in runs/<name>/weights/best.pt. Val is real images only.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

CLASSES = ["fire", "smoke"]


def write_data_yaml(root: Path, synthetic: bool, out: Path) -> Path:
    train = ["images/train"]
    if synthetic:
        train += [s for s in ("images/train_synth", "images/train_screen") if (root / s).exists()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"path: {root}\n"
        f"train: [{', '.join(train)}]\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {dict(enumerate(CLASSES))}\n"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("~/datasets/daedalus_fire").expanduser())
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained COCO checkpoint to start from")
    ap.add_argument("--name", default="real")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()

    data = write_data_yaml(args.data_root, args.synthetic, Path("runs") / f"{args.name}_data.yaml")
    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        cache=False,           # RAM cache of 30k+ images would blow the memory budget
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=True,
        patience=12,
        cos_lr=True,
        close_mosaic=5,
        # colour augmentation matters: screens and prints shift hue/saturation
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.45,
        degrees=10.0, flipud=0.5,  # top-down views have no "up"
        plots=True,
    )


if __name__ == "__main__":
    main()
