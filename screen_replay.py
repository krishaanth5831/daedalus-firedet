"""Ground test: show held-out wildfire images on a screen, film them with the camera, score it.

This is the main way to test without flying. Point the Innomaker at a monitor
(or the laptop screen), run this, and it cycles through *test-split* images
(never seen in training), shows each one fullscreen, lets the camera look at it
for --dwell seconds, and records whether the detector fired. At the end it
prints image-level detection rate and false-alarm rate, and writes a CSV.

    # camera filming the screen (the real test); --display-x moves the slideshow to a 2nd monitor
    python screen_replay.py --source 2 --n 100 --display-x 1920
    # no camera: feed the same images straight in (upper bound, shows the "screen gap")
    python screen_replay.py --direct --n 300
    # your own folder of images (e.g. aerial wildfire frames you found), no labels needed
    python screen_replay.py --source 2 --images ~/Pictures/wildfire_tests --unlabelled

Two windows: "slideshow" (what the camera should film) and "camera" (what the model sees).
Keys in either window: q quit early (results so far are still scored), space pause.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def gt_classes(label_path: Path):
    if not label_path.exists():
        return set()
    return {int(l.split()[0]) for l in label_path.read_text().splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/daedalus_fire_n.pt")
    ap.add_argument("--data-root", type=Path, default=Path("~/datasets/daedalus_fire").expanduser())
    ap.add_argument("--split", default="test", help="test (aerial) or test_ground (D-Fire)")
    ap.add_argument("--images", type=Path, help="use this folder instead of a dataset split")
    ap.add_argument("--unlabelled", action="store_true", help="--images has no labels: just log detections")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--source", default="0")
    ap.add_argument("--direct", action="store_true", help="skip the camera, feed images straight to the model")
    ap.add_argument("--dwell", type=float, default=2.0, help="seconds the camera looks at each image")
    ap.add_argument("--settle", type=float, default=0.4, help="seconds ignored after each image switch (exposure/refresh)")
    ap.add_argument("--hits", type=int, default=3, help="frames with a detection needed to count as 'detected'")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--display-x", type=int, default=0, help="x offset of the monitor that shows the slideshow")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.images:
        imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        lbl_dir = None if args.unlabelled else args.images
    else:
        imgs = sorted((args.data_root / "images" / args.split).glob("*.jpg"))
        lbl_dir = args.data_root / "labels" / args.split
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[: args.n]
    if not imgs:
        raise SystemExit("no images found")

    model = YOLO(args.weights)
    cap = None
    if not args.direct:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise SystemExit(f"could not open camera {args.source!r}")
        cv2.namedWindow("slideshow", cv2.WINDOW_NORMAL)
        cv2.moveWindow("slideshow", args.display_x, 0)
        cv2.setWindowProperty("slideshow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", 960, 540)

    rows, quit_ = [], False
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        gt = gt_classes(lbl_dir / (p.stem + ".txt")) if lbl_dir else None
        hits = {0: 0, 1: 0}
        frames = 0
        if args.direct:
            sources = [img]
        else:
            cv2.imshow("slideshow", img)
            cv2.waitKey(1)
            t_end = time.time() + args.settle + args.dwell
            t_settle = time.time() + args.settle
            sources = None
        while True:
            if args.direct:
                if not sources:
                    break
                frame = sources.pop()
            else:
                if time.time() > t_end:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if time.time() < t_settle:
                    cv2.waitKey(1)
                    continue
            res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            frames += 1
            seen = {int(c) for c in res.boxes.cls.tolist()}
            for c in seen:
                hits[c] += 1
            vis = res.plot()
            cv2.putText(vis, f"{i + 1}/{len(imgs)}  gt={sorted(gt) if gt is not None else '?'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("camera", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                quit_ = True
                break
            if k == ord(" "):
                cv2.waitKey(0)
        need = 1 if args.direct else args.hits
        det = {c for c, n in hits.items() if n >= need}
        rows.append({"image": p.name, "gt": sorted(gt) if gt is not None else "", "detected": sorted(det),
                     "frames": frames, "fire_frames": hits[0], "smoke_frames": hits[1]})
        if quit_:
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

    out = Path("runs") / f"screen_replay_{'direct' if args.direct else 'camera'}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} images, results in {out}")
    if lbl_dir is None:
        print(f"images with any detection: {sum(bool(r['detected']) for r in rows)}/{len(rows)}")
        return
    pos = [r for r in rows if r["gt"]]
    neg = [r for r in rows if not r["gt"]]
    if pos:
        any_hit = np.mean([bool(r["detected"]) for r in pos])
        print(f"fire/smoke images flagged (any class):      {any_hit:6.1%}  ({len(pos)} images)")
        for c, name in ((0, "fire"), (1, "smoke")):
            have = [r for r in pos if c in r["gt"]]
            if have:
                print(f"  {name:5s} images where {name} was found:     {np.mean([c in r['detected'] for r in have]):6.1%}  ({len(have)})")
    if neg:
        print(f"no-fire images that raised a false alarm:  {np.mean([bool(r['detected']) for r in neg]):6.1%}  ({len(neg)} images)")


if __name__ == "__main__":
    main()
