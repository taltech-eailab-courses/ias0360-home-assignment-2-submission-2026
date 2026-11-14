from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "my_digits.png"


def load_digit(path: Path) -> np.ndarray:
    rows = []
    with path.open() as file:
        for line in file:
            rows.append([float(pixel) for pixel in line.strip()])
    return np.asarray(rows, dtype=np.float32)


def main() -> None:

    fig, axes = plt.subplots(2, 5, figsize=(10, 4), constrained_layout=True)
    for label, axis in enumerate(axes.flat):
        image = load_digit(BASE_DIR / f"digit_{label}.txt")
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(str(label))
        axis.axis("off")

    #fig.suptitle("Digits retrieved from the Pico")
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
