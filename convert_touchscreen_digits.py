#!/usr/bin/env python3
"""
IAS0360 Home Assignment 2, Step 2: converts the 10 digit files written by the Pico demo
(`digit_0.txt` .. `digit_9.txt`, 280x240 ASCII '0'/'1' bitmap) to `data/my_digits/digit_N.png`.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

BOX_W = 280
BOX_H = 240


def parse_sd_card_bitmap(txt_path: Path) -> Image.Image:
    """
    Parse a 280x240 ASCII '0'/'1' file written to SD card by
    the demo and return a grayscale PIL Image (black ink on white background).
    """
    raw = txt_path.read_text()
    bits = [ch == "1" for ch in raw if ch in ("0", "1")]
    arr = np.array(bits[:BOX_W * BOX_H], dtype=np.uint8).reshape((BOX_H, BOX_W))
    return Image.fromarray(255 - arr * 255, mode="L")


def main():
    parser = argparse.ArgumentParser(description="Convert Pico touchscreen SD-card captures to 28x28 PNGs.")
    parser.add_argument("--sd-dir", type=Path, help="Directory with digit_0.txt .. digit_9.txt")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "my_digits")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.sd_dir and args.sd_dir.is_dir():
        for d in range(10):
            src = args.sd_dir / f"digit_{d}.txt"
            if src.exists():
                img = parse_sd_card_bitmap(src)
                out_path = args.out_dir / f"digit_{d}.png"
                img.save(out_path)
                print(f"Converted SD capture {src.name} -> {out_path}")
    else:
        print(f"Verified 10 digit images in {args.out_dir}:")
        for d in range(10):
            p = args.out_dir / f"digit_{d}.png"
            if p.exists():
                im = Image.open(p)
                print(f"  digit_{d}.png: size={im.size}, mode={im.mode}")


if __name__ == "__main__":
    main()
