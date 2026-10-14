# How to run it

Everything you need to run the wildfire detector, test it and retrain it.
Commands assume Linux (Ubuntu/Fedora); macOS and Windows work the same way with `python` instead of `.venv/bin/python`.

## 1. Install (once)

You need Python 3.10–3.12. A GPU is optional: the model is small enough for a laptop CPU (measured: 44 FPS PyTorch, 18 FPS ONNX on a Ryzen 7 8845HS at 1280×720).

```bash
git clone https://github.com/krishaanth5831/daedalus-firedet.git
cd daedalus-firedet

# with uv (fast)
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt

# or with plain pip
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

source .venv/bin/activate
```

Check it worked: `python -c "import ultralytics, cv2; print('ok')"`.

The trained model is already in the repo: `weights/daedalus_fire_n.pt` (about 5 MB, classes `fire` and `smoke`).

## 2. Run the live demo

### Find the camera

```bash
python demo.py --list-cameras
```

This prints every camera, for example:

```
/dev/v4l/by-id/usb-Innomaker_..._video-index0 -> /dev/video2
openable cameras:
  --source 0   (/dev/video0, default 640x480)     <- laptop webcam
  --source 2   (/dev/video2, default 1920x1080)   <- Innomaker
```

The Innomaker is a normal USB (UVC) camera: plug it in and it shows up, no driver needed.
Use the `/dev/v4l/by-id/...` path if you want a name that stays the same after re-plugging.

### Start it

```bash
python demo.py --source 2                     # Innomaker (use the number from --list-cameras)
python demo.py --source 0                     # laptop webcam
python demo.py --source wildfire_clip.mp4     # a video file instead of a camera
```

A window opens showing the camera image with boxes:
**orange = fire, grey = smoke**, plus FPS and inference time at the bottom.
A red **"FIRE DETECTED"** banner appears once fire or smoke is seen in 5 frames in a row,
so a single flicker does not trigger it.

| Key | Does |
|---|---|
| `q` / `Esc` | quit |
| `r` | start/stop recording to `recordings/demo_<time>.mp4` |
| `s` | save a snapshot to `recordings/` |
| `f` | fullscreen |
| `conf` slider | confidence threshold; lower = more detections and more false alarms |

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--conf 0.35` | 0.35 | starting confidence threshold |
| `--confirm 5` | 5 | frames in a row before the banner shows |
| `--width 1920 --height 1080` | 1280×720 | camera resolution to request |
| `--imgsz 640` | 640 | model input size; 960 finds smaller fires but is slower |
| `--record` | off | start recording immediately |
| `--device cpu` | auto | force CPU, or `cuda:0` |
| `--weights path.pt` | `weights/daedalus_fire_n.pt` | use other weights |

### What to point it at

The model was trained on **drone and high-vantage photos of wildfires**. Good demo material:
- drone wildfire footage played fullscreen on a monitor or TV (YouTube, or the FLAME dataset videos)
- printed aerial wildfire photos (hold them so they fill about a quarter of the image)
- for false alarms: sunsets, fog, orange roofs. Ideally nothing fires.

Close-up candles or lighters are far from the training data and may or may not be detected.

## 3. Test it properly on the ground

See **[TESTING.md](TESTING.md)** for the full plan. The short version:

```bash
# best case: images straight into the model, no camera
python screen_replay.py --direct --n 300

# real test: camera films a slideshow of held-out test images on a screen, scored automatically
python screen_replay.py --source 2 --n 100
#   --display-x 1920   puts the slideshow on a second monitor
#   --images DIR --unlabelled   use your own folder of pictures
```

It prints the detection rate (fire/smoke images that were caught) and the false-alarm rate
(no-fire images that triggered), and saves a CSV in `runs/`.
`screen_replay.py` needs the dataset (step 5). With `--images DIR --unlabelled` it works without it.

## 4. Run the model in your own code

```python
from ultralytics import YOLO

model = YOLO("weights/daedalus_fire_n.pt")
for r in model.predict("some_image.jpg", conf=0.35):
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        print(model.names[int(cls)], float(conf), box.tolist())   # fire 0.71 [x1, y1, x2, y2]
```

`predict` also takes a folder, a video path, a camera index or a numpy frame (BGR).

## 5. Rebuild the dataset and retrain

About 17 GB of downloads. Everything lands in `~/datasets/`.

```bash
uv pip install --python .venv/bin/python "huggingface_hub[cli]"

# FASDD_UAV (aerial, the main dataset)
hf download hanvithSai/gavin-fasdd-uav-raw --repo-type dataset --local-dir ~/datasets/fasdd_uav/raw
mkdir -p ~/datasets/fasdd_uav/ann
tar xf ~/datasets/fasdd_uav/raw/annotations.tar -C ~/datasets/fasdd_uav/ann
tar xf ~/datasets/fasdd_uav/raw/images.tar -C ~/datasets/fasdd_uav --exclude='*/._*' --exclude='._*'

# D-Fire subset (ground-level)
hf download badsaarow/d-fire --repo-type dataset --local-dir ~/datasets/dfire/raw \
  data/train-00004-of-00009.parquet data/train-00005-of-00009.parquet data/train-00006-of-00009.parquet \
  data/train-00007-of-00009.parquet data/train-00008-of-00009.parquet data/test-00002-of-00003.parquet

python scripts/prepare_fasdd.py               # -> ~/datasets/daedalus_fire (train/val/test)
python scripts/prepare_dfire.py               # + 4000 train, 1000 test_ground
python scripts/make_synthetic.py --preview 16 # check preview_synthetic.jpg looks sane
python scripts/make_synthetic.py              # 6000 copy-paste + 3000 "screen photo" images
```

Train and compare:

```bash
python train.py --name real                       # real images only
python train.py --name synth --synthetic          # real + synthetic
python evaluate.py runs/real/weights/best.pt runs/synth/weights/best.pt
cp runs/<winner>/weights/best.pt weights/daedalus_fire_n.pt
```

On an RTX 1000 Ada laptop (6 GB) one epoch takes about 4 min (real) or about 6 min (with synthetic),
and training uses about 5 GB of RAM. Lower `--batch` if you get CUDA out-of-memory errors.
Use `--fraction 0.05 --epochs 1` for a 1-minute smoke test.

## 6. Export for the drone computer

```bash
python export.py --format onnx      # portable: CPU, onnxruntime, most boards
python export.py --format engine    # NVIDIA Jetson (TensorRT). Run this ON the Jetson
python export.py --format ncnn      # Raspberry Pi / ARM CPU
python demo.py --weights weights/daedalus_fire_n.onnx --source 0   # exported models run in the demo too
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `The function is not implemented. Rebuild the library with ... GTK/Qt` / no window opens | You have `opencv-python-headless` (often pulled in by other packages). `pip uninstall opencv-python-headless opencv-python && pip install opencv-python` |
| `could not open source` | Wrong index: run `--list-cameras`. Is another app (Zoom, browser) using the camera? |
| Black or frozen image | Try the other `/dev/videoN` of the same camera (each USB cam lists two; one is metadata only) |
| Low FPS | `--imgsz 480`, a lower `--width/--height`, or run on a GPU |
| Window blinks or warnings about `wayland` / fonts | Harmless Qt messages, ignore |
| Too many false alarms | Raise the `conf` slider to 0.5 or more and `--confirm 10` |
