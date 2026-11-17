"""Convert touch-screen captures from the SD card to 28x28 grey PNGs in own_data/.

The lab_1_1/lcd_sd_card_example firmware writes lcd_sd_card_example_<n>.txt:
280*240 ASCII '0'/'1' characters (row-major, width 280, no newlines).
Image files (png/bmp/jpg) are also accepted.

Labels: a file name starting with a digit (7_1.txt) uses that digit; otherwise the
files are labelled 0,1,2,... in numeric order of <n>, or via --labels 0 1 2 ...

Usage: python3 tools/collect_touch_digits.py <sd_dir> [--labels 0 1 2 ...]
"""
import argparse
import os
import re

import numpy as np
from PIL import Image

W, H = 280, 240


def to_mnist(a):
    """Crop, pad, resize and center a grayscale digit on a 28x28 black canvas."""
    ys, xs = np.nonzero(a > 0.1)
    if len(ys) == 0:
        raise ValueError("empty drawing")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(a.shape) * 1.4)  # MNIST-like margin around the digit
    sq = np.zeros((side, side), dtype="float32")
    y0, x0 = (side - a.shape[0]) // 2, (side - a.shape[1]) // 2
    sq[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    img = Image.fromarray((sq * 255).astype("uint8")).resize((28, 28), Image.BOX)
    pixels = np.asarray(img, dtype="float32")
    mass = pixels.sum()
    if mass:
        rows, cols = np.indices(pixels.shape)
        cy = float((rows * pixels).sum() / mass)
        cx = float((cols * pixels).sum() / mass)
        img = img.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, cx - 13.5, 0, 1, cy - 13.5),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
        pixels = np.asarray(img, dtype="float32")
        peak = pixels.max()
        if peak:
            img = Image.fromarray((pixels * (255.0 / peak)).clip(0, 255).astype("uint8"))
    return img


def load(path):
    if path.lower().endswith(".txt"):
        raw = np.frombuffer(open(path, "rb").read(), dtype="uint8")
        raw = raw[(raw == ord("0")) | (raw == ord("1"))]
        if raw.size != W * H:
            raise ValueError(f"expected {W * H} chars, got {raw.size}")
        return (raw == ord("1")).reshape(H, W).astype("float32")
    a = np.asarray(Image.open(path).convert("L"), dtype="float32") / 255.0
    return 1.0 - a if a.mean() > 0.5 else a


def num(name):
    m = re.findall(r"\d+", name)
    return int(m[-1]) if m else 0


ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--labels", nargs="*", type=int)
a = ap.parse_args()
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "own_data")
os.makedirs(dst, exist_ok=True)
files = sorted((f for f in os.listdir(a.src) if f.lower().endswith((".txt", ".png", ".bmp", ".jpg", ".jpeg"))),
               key=num)
for i, f in enumerate(files):
    stem = os.path.splitext(f)[0]
    label = int(f[0]) if f[0].isdigit() else (a.labels[i] if a.labels else i % 10)
    out = f"{label}_{stem}.png"
    to_mnist(load(os.path.join(a.src, f))).save(os.path.join(dst, out))
    print(f, "->", out)
