"""Live fire/smoke detection from a camera (Innomaker USB cam, laptop webcam, or a video file).

    python demo.py --list-cameras                 # which /dev/videoN is the Innomaker?
    python demo.py --source 2                     # open camera index 2 in a window
    python demo.py --source /dev/v4l/by-id/usb-...-video-index0   # stable path, survives replugging
    python demo.py --source clip.mp4              # run on a video file

Window keys:  q / Esc quit   s save snapshot   r start/stop recording   f fullscreen
The "conf" slider sets the confidence threshold live.
The red banner only fires after --confirm consecutive frames with fire/smoke,
so a single flickering false positive does not trigger an alarm.
"""
import argparse
import glob
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

COLOURS = {0: (0, 69, 255), 1: (200, 200, 200)}  # fire = orange-red, smoke = grey (BGR)
WIN = "Daedalus fire detection"


def list_cameras():
    by_id = sorted(glob.glob("/dev/v4l/by-id/*"))
    for p in by_id:
        print(f"{p}\n    -> {os.path.realpath(p)}")
    if not by_id:
        print("no /dev/v4l/by-id entries; trying indices 0-9")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            print(f"index {i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()


def open_source(src, width, height, fps):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
    elif src.startswith("/dev/"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    else:
        return cv2.VideoCapture(src)
    # MJPG lets most UVC cams (Innomaker included) deliver 1080p at full frame rate over USB 2
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/daedalus_fire_n.pt")
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--confirm", type=int, default=5, help="consecutive frames needed for the alarm banner")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    model = YOLO(args.weights)
    names = model.names
    cap = open_source(args.source, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {args.source!r}; try --list-cameras")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.createTrackbar("conf x100", WIN, int(args.conf * 100), 95, lambda v: None)

    out_dir = Path("recordings")
    writer, streak, fs = None, 0, False
    times = deque(maxlen=30)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        conf = max(0.05, cv2.getTrackbarPos("conf x100", WIN) / 100)
        t0 = time.perf_counter()
        res = model.predict(frame, conf=conf, imgsz=args.imgsz, verbose=False, device=args.device)[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        times.append(time.perf_counter())

        n_fire = n_smoke = 0
        for box, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            c = int(c)
            n_fire += c == 0
            n_smoke += c == 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOURS[c], 3)
            label = f"{names[c]} {s:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), COLOURS[c], -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        streak = streak + 1 if (n_fire or n_smoke) else 0
        H, W = frame.shape[:2]
        if streak >= args.confirm:
            what = " + ".join(k for k, n in (("FIRE", n_fire), ("SMOKE", n_smoke)) if n)
            cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 200), -1)
            cv2.putText(frame, f"{what} DETECTED", (20, 43), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
        hud = f"{fps:4.1f} FPS  infer {infer_ms:4.0f} ms  conf {conf:.2f}  fire {n_fire}  smoke {n_smoke}"
        cv2.putText(frame, hud, (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, hud, (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer:
            writer.write(frame)
            cv2.circle(frame, (W - 30, 90), 10, (0, 0, 255), -1)

        cv2.imshow(WIN, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if k == ord("s"):
            out_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(out_dir / f"demo_{stamp}.jpg"), frame)
        elif k == ord("r"):
            if writer:
                writer.release()
                writer = None
            else:
                out_dir.mkdir(exist_ok=True)
                writer = cv2.VideoWriter(str(out_dir / f"demo_{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 10), (W, H))
        elif k == ord("f"):
            fs = not fs
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fs else cv2.WINDOW_NORMAL)

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
