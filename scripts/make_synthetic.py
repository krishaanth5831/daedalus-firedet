"""Synthetic training data: copy-paste fire/smoke + "photo of a screen" degradation.

Two generators, both write YOLO images/labels into <out>/{images,labels}/<split>:

1. paste  (split train_synth)
   Cut real fire/smoke patches out of FASDD *train* boxes, feather them with a
   soft mask, jitter scale/colour, and paste them onto aerial
   "neitherFireNorSmoke" backgrounds. Boxes come for free from the paste position.
   This multiplies the number of (background, fire) combinations the model sees,
   which is the cheap version of synthetic data. No GPU or generative model needed.

2. screen (split train_screen)
   Take real train images (aerial + D-Fire) and make them look like a camera
   filmed them off a monitor or print: perspective tilt, dark bezel, moire,
   colour cast, glare, blur, JPEG artifacts. Boxes are warped with the image.
   This targets the ground demo, where the camera looks at pictures of fires.

Synthetic images never go into val or test. The evaluation is always on real images.

    python scripts/make_synthetic.py --data ~/datasets/daedalus_fire --n-paste 6000 --n-screen 3000
    python scripts/make_synthetic.py --preview 12     # writes a contact sheet, nothing else
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

FIRE, SMOKE = 0, 1


# ---------------------------------------------------------------- label helpers
def read_labels(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), *map(float, p[1:])))
    return out


def write_labels(path: Path, boxes):
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))


def to_xyxy(b, w, h):
    c, x, y, bw, bh = b
    return c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h


def to_yolo(c, x1, y1, x2, y2, w, h):
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


# ---------------------------------------------------------------- 1. copy-paste
def collect_patches(data: Path, max_patches: int, rng: random.Random):
    """Crop fire/smoke boxes out of real train images (with a small context margin)."""
    imgs = sorted((data / "images" / "train").glob("*.jpg"))
    imgs = [p for p in imgs if not p.name.startswith(("neither", "dfire_"))]
    rng.shuffle(imgs)
    patches = {FIRE: [], SMOKE: []}
    for p in imgs:
        if min(len(patches[FIRE]), len(patches[SMOKE])) >= max_patches:
            break
        boxes = read_labels(data / "labels" / "train" / (p.stem + ".txt"))
        if not boxes:
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        H, W = im.shape[:2]
        for b in boxes:
            c, x1, y1, x2, y2 = to_xyxy(b, W, H)
            if len(patches[c]) >= max_patches or (x2 - x1) < 12 or (y2 - y1) < 12:
                continue
            m = 0.05  # small margin so the mask's feathered edge does not clip the flame
            dx, dy = (x2 - x1) * m, (y2 - y1) * m
            crop = im[int(max(0, y1 - dy)) : int(min(H, y2 + dy)), int(max(0, x1 - dx)) : int(min(W, x2 + dx))]
            if crop.size:
                patches[c].append(crop.copy())
    return patches


def soft_mask(h, w, cls, patch):
    """Alpha for pasting. Fire: only flame-coloured pixels. Smoke: soft ellipse weighted by greyness.

    Pasting the whole box drags the source background along (a dark oval around the
    flame), which the model could learn as a shortcut, so the mask follows the content.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    ellipse = np.clip((1.3 - d) / 0.6, 0, 1)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[..., i].astype(np.float32) for i in range(3))
    if cls == FIRE:
        flame = ((hue <= 35) | (hue >= 170)) & (sat > 70) & (val > 110) | (val > 235)
        a = flame.astype(np.float32)
        if a.mean() < 0.03:  # no clear flame pixels (e.g. smouldering): fall back to brightness
            a = np.clip(val / 255 - 0.2, 0, 1)
        a = cv2.dilate(a, np.ones((3, 3), np.uint8)) * ellipse
    else:
        a = ellipse * np.clip(1.2 - sat / 255 * 1.5, 0, 1) * 0.85
    return cv2.GaussianBlur(a, (0, 0), max(0.8, min(h, w) / 25))[..., None]


def looks_like_sky(region):
    """Blue or bright washed-out region: fires do not sit there."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0].mean(), hsv[:, 1].mean(), hsv[:, 2].mean()
    return (90 <= h <= 130 and s > 35 and v > 110) or (v > 190 and s < 35)


def jitter_colour(patch, rng):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] *= rng.uniform(0.8, 1.2)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste_one(bg, patches, rng):
    H, W = bg.shape[:2]
    out = bg.astype(np.float32)
    boxes = []
    for _ in range(rng.randint(1, 5)):
        cls = FIRE if rng.random() < 0.6 else SMOKE
        if not patches[cls]:
            continue
        p = rng.choice(patches[cls])
        if rng.random() < 0.5:
            p = cv2.flip(p, 1)
        # aerial fires are mostly small: bias towards 2-15 % of image width
        target_w = W * min(0.4, rng.lognormvariate(np.log(0.07), 0.6))
        s = min(2.0, target_w / p.shape[1])  # upscaling tiny crops further just makes blur
        pw, ph = max(8, int(p.shape[1] * s)), max(8, int(p.shape[0] * s))
        if pw >= W or ph >= H:
            continue
        p = jitter_colour(cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), rng)
        a = soft_mask(ph, pw, cls, p)
        # pick a spot on the ground: the patch's lower part must not be sky
        for _ in range(15):
            x, y = rng.randint(0, W - pw), rng.randint(0, H - ph)
            base = bg[y + ph * 2 // 3 : y + ph, x : x + pw]
            if base.size and not looks_like_sky(base):
                break
        else:
            continue
        roi = out[y : y + ph, x : x + pw]
        out[y : y + ph, x : x + pw] = roi * (1 - a) + p.astype(np.float32) * a
        # box = region where the mask is clearly visible, not the whole padded patch
        ys, xs = np.where(a[..., 0] > 0.35)
        if len(xs):
            b = to_yolo(cls, x + xs.min(), y + ys.min(), x + xs.max(), y + ys.max(), W, H)
            if b:
                boxes.append(b)
    return out.astype(np.uint8), boxes


# ---------------------------------------------------------------- 2. screen photo
def screen_photo(im, boxes, rng):
    """Make an image look like a phone/USB camera filmed it off a monitor or print."""
    H, W = im.shape[:2]
    # moire / scanlines on the "displayed" image before it is warped
    if rng.random() < 0.6:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        f, ang = rng.uniform(0.6, 1.6), rng.uniform(0, np.pi)
        pat = np.sin(f * (xx * np.cos(ang) + yy * np.sin(ang)))
        im = np.clip(im.astype(np.float32) * (1 + rng.uniform(0.02, 0.08) * pat[..., None]), 0, 255).astype(np.uint8)

    # place the screen inside a bigger dark frame (bezel / desk) with perspective tilt
    pad = rng.uniform(0.03, 0.18)
    OW, OH = int(W * (1 + 2 * pad)), int(H * (1 + 2 * pad))
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    j = lambda: rng.uniform(-0.08, 0.08)
    dst = np.float32([
        [W * pad + W * j(), H * pad + H * j()],
        [W * (1 + pad) + W * j(), H * pad + H * j()],
        [W * (1 + pad) + W * j(), H * (1 + pad) + H * j()],
        [W * pad + W * j(), H * (1 + pad) + H * j()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    bezel = tuple(int(v) for v in rng.choice([(10, 10, 10), (25, 25, 28), (60, 55, 50), (200, 200, 195)]))
    out = cv2.warpPerspective(im, M, (OW, OH), borderValue=bezel)

    new_boxes = []
    for b in boxes:
        c, x1, y1, x2, y2 = to_xyxy(b, W, H)
        pts = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), M)[0]
        nb = to_yolo(c, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), OW, OH)
        if nb:
            new_boxes.append(nb)

    out = out.astype(np.float32)
    # white balance cast + exposure + gamma
    out *= np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    out = 255 * (np.clip(out, 0, 255) / 255) ** rng.uniform(0.75, 1.35)
    # glare blob
    if rng.random() < 0.5:
        cx, cy, r = rng.uniform(0, OW), rng.uniform(0, OH), rng.uniform(0.1, 0.4) * OW
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(40, 120)
        out += g[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    # defocus / motion blur
    if rng.random() < 0.6:
        k = rng.choice([3, 5, 7])
        if rng.random() < 0.5:
            out = cv2.GaussianBlur(out, (k, k), 0)
        else:
            ker = np.zeros((k, k), np.float32)
            ker[k // 2] = 1 / k
            out = cv2.filter2D(out, -1, ker)
    # sensor noise
    out = np.clip(out.astype(np.int16) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(2, 8), out.shape), 0, 255).astype(np.uint8)
    # JPEG round trip
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 85)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), new_boxes


def resize_long(im, long_side):
    s = long_side / max(im.shape[:2])
    return cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else im


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("~/datasets/daedalus_fire").expanduser())
    ap.add_argument("--n-paste", type=int, default=6000)
    ap.add_argument("--n-screen", type=int, default=3000)
    ap.add_argument("--max-patches", type=int, default=3000, help="per class")
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, help="only write N examples to preview_synthetic.jpg")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    d = args.data

    train_imgs = sorted((d / "images" / "train").glob("*.jpg"))
    backgrounds = [p for p in train_imgs if p.name.startswith("neitherFireNorSmoke")]
    positives = [p for p in train_imgs if not p.name.startswith("neitherFireNorSmoke")]
    print(f"{len(backgrounds)} aerial backgrounds, {len(positives)} real positives")

    n_paste, n_screen = (args.preview // 2, args.preview - args.preview // 2) if args.preview else (args.n_paste, args.n_screen)
    patches = collect_patches(d, min(args.max_patches, 200) if args.preview else args.max_patches, rng)
    print(f"patches: fire={len(patches[FIRE])} smoke={len(patches[SMOKE])}")

    previews = []

    def emit(split, name, im, boxes):
        if args.preview:
            vis = im.copy()
            for c, x, y, w, h in boxes:
                H, W = vis.shape[:2]
                cv2.rectangle(vis, (int((x - w / 2) * W), int((y - h / 2) * H)), (int((x + w / 2) * W), int((y + h / 2) * H)), (0, 0, 255) if c == FIRE else (200, 200, 200), 2)
            previews.append(cv2.resize(vis, (480, 360)))
            return
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "images" / split / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_labels(d / "labels" / split / f"{name}.txt", boxes)

    for i in range(n_paste):
        bg = cv2.imread(str(rng.choice(backgrounds)))
        im, boxes = paste_one(resize_long(bg, args.long_side), patches, rng)
        emit("train_synth", f"synth_paste_{i:06d}", im, boxes)
        if i and i % 1000 == 0:
            print(f"  paste {i}/{n_paste}")

    for i in range(n_screen):
        p = rng.choice(positives)
        im = resize_long(cv2.imread(str(p)), args.long_side)
        boxes = read_labels(d / "labels" / "train" / (p.stem + ".txt"))
        im, boxes = screen_photo(im, boxes, rng)
        emit("train_screen", f"synth_screen_{i:06d}", im, boxes)
        if i and i % 1000 == 0:
            print(f"  screen {i}/{n_screen}")

    if args.preview:
        rows = [np.hstack(previews[i : i + 4] + [np.zeros_like(previews[0])] * (4 - len(previews[i : i + 4]))) for i in range(0, len(previews), 4)]
        cv2.imwrite("preview_synthetic.jpg", np.vstack(rows))
        print("wrote preview_synthetic.jpg")
    else:
        print(f"done: {n_paste} paste + {n_screen} screen images in {d}")


if __name__ == "__main__":
    main()
