"""Step 2: convert own IMU recordings to the provided dataset's orientation.

  1. Finds the axis map: tries all 24 axis-aligned rotations and picks the one
     under which the per-class mean gestures best match the provided dataset.
     --axis-map sets it by hand, e.g. --axis-map=-y,x,z
  2. Saves it to training/axis_map.json (used by run_experiments.py and
     export_firmware.py).
  3. Writes converted copies to dataset/own_converted/<class>/.

Usage:  python training/convert_own_data.py [--axis-map x,y,z]
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import gesture_data as gd

OUT_DIR = gd.REPO_ROOT / "dataset" / "own_converted"


def proper_axis_maps():
    """All 24 signed axis permutations with determinant +1."""
    maps = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for out_axis, (src, sign) in enumerate(zip(perm, signs)):
                matrix[out_axis, src] = sign
            if np.isclose(np.linalg.det(matrix), 1.0):
                maps.append(tuple(("-" if s < 0 else "") + "xyz"[p] for p, s in zip(perm, signs)))
    return maps


def class_means(frames, labels):
    return {c: frames[labels == c].mean(axis=0) for c in np.unique(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis-map", help="manual map, e.g. 'x,y,z' or '-y,x,z'")
    args = parser.parse_args()

    own_records = gd.discover(gd.OWN_DIR)
    if not own_records:
        raise SystemExit(f"no recordings in {gd.OWN_DIR}/<class>/ - copy them from the SD card first")
    counts = {name: sum(r.label == i for r in own_records) for i, name in enumerate(gd.CLASS_NAMES)}
    print("own recordings:", counts)
    short = [n for n, c in counts.items() if c < 20]
    if short:
        print(f"note: the assignment asks for at least 20 recordings per gesture; short: {short}")

    if args.axis_map:
        axis_map = tuple(a.strip() for a in args.axis_map.split(","))
        gd.parse_axis_map(axis_map)  # validate
        print(f"using manual axis map {axis_map}")
    else:
        ref_frames, ref_labels, _ = gd.load_frames(gd.discover(gd.PROVIDED_DIR))
        reference = class_means(ref_frames, ref_labels)
        raw_own = [gd.load_recording(r.path) for r in own_records]
        own_labels = np.asarray([r.label for r in own_records])
        scores = []
        for candidate in proper_axis_maps():
            frames = np.stack([gd.to_frames(gd.apply_axis_map(raw, candidate)) for raw in raw_own])
            means = class_means(frames, own_labels)
            err = np.mean([np.mean((means[c] - reference[c]) ** 2) for c in means])
            scores.append((err, candidate))
        scores.sort()
        print("best axis maps (mean squared distance of class-mean gestures, g^2):")
        for err, candidate in scores[:4]:
            print(f"  {','.join(candidate):<10} {err:.4f}")
        axis_map = scores[0][1]
        if scores[1][0] < 1.5 * scores[0][0]:
            print("WARNING: the best two maps are close - check the orientation by hand (--axis-map)")
        print(f"selected axis map {axis_map}")

    gd.AXIS_MAP_PATH.write_text(json.dumps({"axis_map": list(axis_map)}, indent=2) + "\n")
    print(f"saved {gd.AXIS_MAP_PATH.relative_to(gd.REPO_ROOT)}")

    for rec in own_records:
        converted = gd.apply_axis_map(gd.load_recording(rec.path), axis_map)
        out = OUT_DIR / gd.CLASS_NAMES[rec.label] / rec.path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, converted, fmt="%d", delimiter=",", header="ax,ay,az", comments="")
    print(f"wrote {len(own_records)} converted recordings to {OUT_DIR.relative_to(gd.REPO_ROOT)}/")
    print("next: python training/run_experiments.py --force   (re-evaluates every model on your data)")


if __name__ == "__main__":
    main()
