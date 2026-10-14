# Testing on the ground

The model is trained for drone views, but we cannot fly it yet. These tests get as close as
possible from the ground, ordered from "most like the real thing and measurable" to "show-off demo".
Write every result down (the scripts save CSVs in `runs/`), so model versions can be compared.

## 0. First: the upper bound (no camera)

```bash
python screen_replay.py --direct --n 300                      # aerial test split
python screen_replay.py --direct --n 300 --split test_ground  # ground-level D-Fire test subset
```

This feeds held-out test images straight into the model: the best it can do.
Every camera test below scores lower; the difference is the "camera gap" to shrink.

## 1. Screen replay (main test, scored automatically)

Point the Innomaker at a monitor. The script runs a slideshow of **test-split** images
(never used for training, so the score is honest), shows each one fullscreen,
watches through the camera for 2 s and checks whether fire or smoke was found.

```bash
python demo.py --list-cameras                                      # find the Innomaker
python screen_replay.py --source <idx> --n 100 --display-x 1920    # slideshow on the 2nd monitor
```

It prints:
- **detection rate**: share of fire/smoke images where the model fired, overall and per class
- **false-alarm rate**: share of no-fire images where it fired anyway (FASDD's test split has many no-fire aerial images)

Change one thing at a time and re-run the same set (`--n 100 --seed 0`):

| Variable | Why |
|---|---|
| Distance (screen fills 100 % / 50 % / 25 % of the frame) | smaller on screen = "higher altitude" |
| Screen brightness, room lights on/off | exposure, glare |
| Camera tilt (straight on vs 30°) | oblique views when the drone banks |
| `--conf` 0.25 / 0.35 / 0.5 | picks the alarm threshold: detection rate vs false alarms |

## 2. Printouts at a known distance = simulated altitude

Print a few aerial fire images and hold them at measured distances. What the sensor sees depends
only on size / distance, so a print can stand in for altitude:

```
equivalent altitude = real fire width × (camera-to-print distance / fire width on the print)
```

Example: a 10 m fire front printed 5 cm wide, seen from 60 cm, looks to the camera like that
fire from **120 m**. Move the print back until detection stops: that is roughly the maximum
altitude for that fire size, **for this camera, lens and resolution** (redo it if any change).

## 3. Video replay

Play real drone wildfire footage fullscreen (e.g. FLAME dataset videos, YouTube) and film it:

```bash
python demo.py --source <idx>        # r = record, s = snapshot, f = fullscreen
python demo.py --source footage.mp4  # same, straight from the file (no camera gap)
```

Watch for: frames until the red banner fires (`--confirm 5`), flickering boxes,
and what happens when smoke hides the fire.

## 4. False-alarm hunting (just as important)

A wildfire camera that alarms at every sunset is useless. Show it things that look like fire:

- sunsets, orange skies, red and orange roofs and cars, autumn trees
- clouds, fog, mist, dust, kettle steam, exhaust (they look like smoke)
- red clothing, traffic and tail lights
- sun glare on water or glass (a classic aerial false positive)

```bash
python screen_replay.py --source <idx> --images ~/Pictures/fire_lookalikes --unlabelled
```

Keep every image that triggers in a "hard negatives" folder. Adding those (with empty label files)
to training is the most useful next improvement.

## 5. Real, small, controlled fire (supervised)

Candle, lighter, barbecue, incense or smoke pellets, filmed from a few metres. Only outdoors or
in a lab, with an extinguisher or fire blanket and a responsible person present.
This tests real flames (real colour, flicker, sensor saturation) instead of pixels of flames.
Close-up candles are far from the aerial training data, so treat it as a sanity check, not a score.

## 6. Rooftop "mini altitude" test

From a roof or high balcony (with permission), point the camera down at an open area with a
small smoke source, and an orange tarp as a false-positive check. Looking down from 10–20 m is the
closest thing to a drone view without flying.

## 7. Speed on the real hardware

The onboard computer is not decided. Once it is, export and read FPS / inference ms off the `demo.py` HUD:

```bash
python export.py --format onnx      # portable (CPU, onnxruntime)
python export.py --format engine    # Jetson / TensorRT, run on the Jetson itself
python export.py --format ncnn      # Raspberry Pi / ARM CPU
```

## A 3-minute demo

1. `demo.py` on the Innomaker pointed at a monitor playing drone wildfire footage: boxes and the red banner.
2. Hold up a printed aerial fire and walk it back: detection drops out at "high altitude".
3. Show a sunset: no alarm (or, if there is one, explain the hard-negatives plan).
4. One slide with the screen-replay numbers: detection rate and false-alarm rate on held-out images.
