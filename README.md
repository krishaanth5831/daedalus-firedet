# daedalus-firedet

Wildfire detection (fire + smoke boxes) for the Team Daedalus UAV: a small YOLO model fine-tuned on
**aerial** wildfire imagery, plus a live camera demo and a ground-testing kit for when we cannot fly.

> Branches: `dev` = work in progress, `main` = finished and working. `main` is protected (PR only).

## Quick start: live demo

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

python demo.py --list-cameras          # find the Innomaker (USB/UVC) camera
python demo.py --source 2              # opens a window: boxes, FPS, "FIRE DETECTED" banner
```

Point the camera at a screen playing drone wildfire footage or at a printed aerial fire photo.
In the window: `conf` slider = threshold, `r` record, `s` snapshot, `f` fullscreen, `q` quit.
The weights in `weights/daedalus_fire_n.pt` are used by default (about 5 MB, runs on CPU too).

How to test it properly on the ground (screen replay with automatic scoring, printouts as
simulated altitude, false-alarm hunting): see **[TESTING.md](TESTING.md)**.

## Model

RESULTS_PLACEHOLDER

## Data

| Source | What | Used for | License |
|---|---|---|---|
| [FASDD_UAV](https://doi.org/10.57760/sciencedb.j00104.00103) (via HF mirror `hanvithSai/gavin-fasdd-uav-raw`) | 25,097 UAV / high-vantage images, fire + smoke boxes, ~37 % no-fire | train / val / **test (official split, untouched)** | CC BY 4.0 |
| [D-Fire](https://github.com/gaiasd/DFireDataset) (via HF mirror `badsaarow/d-fire`) | ground-level fire/smoke; 4,000 train + 1,000 test sampled | train + `test_ground` | see upstream |
| Synthetic copy-paste (`scripts/make_synthetic.py`) | real FASDD fire/smoke cut-outs pasted onto no-fire aerial scenes | train only | derived from FASDD |
| Synthetic "screen photo" (`scripts/make_synthetic.py`) | real train images made to look filmed off a monitor (tilt, bezel, moiré, glare, blur, JPEG) | train only | derived |

Classes: `0 = fire`, `1 = smoke` (D-Fire's reversed order is remapped).
Synthetic images never go into val or test; every number below is on real held-out images.

FASDD citation (required by the license): Ming Wang, Peng Yue, Liangcun Jiang, Dayu Yu, Tianyu Tuo, Jian Li.
"An open flame and smoke detection dataset for deep learning in remote sensing based fire detection."
*Geo-Spatial Information Science* 28(2), 511–526, 2024. doi:10.1080/10095020.2024.2347922

## Rebuild everything

```bash
# 1. data (~16 GB download), lands in ~/datasets
hf download hanvithSai/gavin-fasdd-uav-raw --repo-type dataset --local-dir ~/datasets/fasdd_uav/raw
mkdir -p ~/datasets/fasdd_uav/ann && tar xf ~/datasets/fasdd_uav/raw/annotations.tar -C ~/datasets/fasdd_uav/ann
tar xf ~/datasets/fasdd_uav/raw/images.tar -C ~/datasets/fasdd_uav --exclude='*/._*' --exclude='._*'
hf download badsaarow/d-fire --repo-type dataset --local-dir ~/datasets/dfire/raw \
   data/train-0000{4,5,6,7,8}-of-00009.parquet data/test-00002-of-00003.parquet

python scripts/prepare_fasdd.py
python scripts/prepare_dfire.py
python scripts/make_synthetic.py --preview 16     # look at preview_synthetic.jpg first
python scripts/make_synthetic.py

# 2. train (A/B: with and without synthetic) and score on the real test splits
python train.py --name real
python train.py --name synth --synthetic
python evaluate.py runs/real/weights/best.pt runs/synth/weights/best.pt

# 3. export for the onboard computer once it is chosen
python export.py --format onnx
```

## Open hardware questions

The camera and onboard computer for the drone are **not decided yet**. This model is RGB-only.
- If the final payload is RGB: retrain at the real resolution and lens, and re-run the TESTING.md altitude test.
- If it is thermal (radiometric or not): this model does not transfer; thermal needs its own data
  (e.g. FLAME 2 has paired RGB/thermal drone fire footage). The demo and test harness still apply.
- Onboard compute: export to ONNX (portable), TensorRT (Jetson) or NCNN (Pi) and measure FPS there.

## Next improvements

1. **Hard negatives**: collect what fools it (sunsets, orange roofs, fog, glare on water) and add them with empty labels.
2. **Generative synthetic data**: image-to-image diffusion to add smoke/fire to our own drone footage
   of local terrain (Dutch/European landscapes look nothing like FASDD's California hills).
3. Higher input resolution (`--imgsz 960`) for small, distant fires, if the onboard compute allows.
4. Tracking across frames (ByteTrack) so one detection is not one alarm.
