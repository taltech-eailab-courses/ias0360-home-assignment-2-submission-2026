"""Fixed board-image preprocessing. TEST is exported separately, never used for tuning."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def preprocess(path):
    with Image.open(path) as image:
        mask = (np.asarray(image.convert("L")) < 128).astype(np.uint8)
    # Remove isolated touch specks; retain separate legitimate pen strokes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 8
    cleaned = keep[labels] * 255
    ys, xs = np.where(cleaned)
    if len(xs) == 0:
        raise ValueError(f"No usable drawing: {path}")
    cropped = Image.fromarray(cleaned).crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = 20 / max(cropped.size)
    resized = cropped.resize(tuple(max(1, round(side * scale)) for side in cropped.size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    # A two-pixel board stroke otherwise nearly disappears when reduced ~10x.
    canvas = canvas.filter(ImageFilter.MaxFilter(3))
    arr = np.asarray(canvas, dtype=np.float32)
    yy, xx = np.indices(arr.shape)
    mass = float(arr.sum())
    dx = round(13.5 - float((arr * xx).sum()) / mass)
    dy = round(13.5 - float((arr * yy).sum()) / mass)
    centered = canvas.transform((28, 28), Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=0)
    result = np.asarray(centered, dtype=np.float32)
    result /= max(float(result.max()), 1.0)
    return result[..., np.newaxis]


def prepare(data, output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"shape": [28, 28, 1], "dtype": "float32", "range": [0, 1],
                "foreground": "white strokes on black", "method": "remove connected components smaller than 8 pixels; crop; fit within 20x20 with aspect ratio; pad to 28x28; max filter 3; center of mass; normalize peak to 1",
                "test_policy": "Same fixed preprocessing for all splits; TEST remains a separate file and is excluded from training/model selection."}
    for split in ["TRAIN", "VAL", "TEST"]:
        paths = sorted((data / split).glob("*.PGM"))
        x = np.stack([preprocess(path) for path in paths])
        y = np.array([int(path.stem[1]) for path in paths], dtype=np.int64)
        assert x.shape == (len(paths), 28, 28, 1)
        assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
        assert np.all(x.sum(axis=(1, 2, 3)) > 0)
        np.savez_compressed(output / f"own_{split.lower()}.npz", x=x, y=y, filenames=np.array([path.name for path in paths]))
        metadata[split] = {"count": len(paths), "images_per_digit": np.bincount(y, minlength=10).tolist()}
        # Inspect only training/validation, keeping test images out of development.
        if split == "TEST":
            continue
        rounds = 3
        sheet = Image.new("RGB", (10 * 112, rounds * 130 + 28), "#cccccc")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), f"{split}: model input, enlarged 4x without smoothing", fill="black")
        for row in range(rounds):
            for digit in range(10):
                name = f"D{digit}_{row + 1:03d}.PGM"
                index = next(i for i, path in enumerate(paths) if path.name == name)
                tile = Image.fromarray((x[index, :, :, 0] * 255).astype(np.uint8)).resize((112, 112), Image.Resampling.NEAREST)
                sheet.paste(tile, (digit * 112, row * 130 + 46))
                draw.text((digit * 112 + 2, row * 130 + 30), name, fill="black")
        sheet.save(output / f"input_preview_{split.lower()}.png")
    (output / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.output)
