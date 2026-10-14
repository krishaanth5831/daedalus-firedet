"""Export the trained weights for other hardware.

    python export.py                       # ONNX (runs on CPU, Jetson, Pi via onnxruntime)
    python export.py --format engine       # TensorRT, run this ON the target Jetson
    python export.py --format ncnn         # Raspberry Pi / ARM CPU

The onboard computer is not decided yet, so ONNX is the default: it is the most portable.
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/daedalus_fire_n.pt")
ap.add_argument("--format", default="onnx")
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()
print(YOLO(args.weights).export(format=args.format, imgsz=args.imgsz, half=args.format == "engine"))
