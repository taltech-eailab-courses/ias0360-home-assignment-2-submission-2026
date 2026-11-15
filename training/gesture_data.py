"""IMU gesture data loading, preprocessing and augmentation.

Preprocessing matches the firmware's DataPreprocessor
(firmware/gesture_inference/src/data_preprocessor.cpp):

    raw int16 counts --(axis map)--> g = counts / 16384
        --> cut to the first WINDOW_LENGTH samples (or pad with the per-axis
            mode of the last PAD_MODE_WINDOW samples)
        --> block-average down to TARGET_LENGTH frames
        --> (x - mean) / std
        --> uint8 quantization (TFLite input tensor parameters)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Written to firmware/gesture_inference/src/preprocessing_settings.hpp by export_firmware.py.
COUNTS_PER_G = 16384.0
WINDOW_LENGTH = 4000      # raw samples per axis the model input is built from
TARGET_LENGTH = 128       # frames per axis fed to the model
PAD_MODE_WINDOW = 100     # tail samples used to compute the padding value
NUM_AXES = 3
CLASS_NAMES = ("negative", "ring", "slope", "wing")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVIDED_DIR = REPO_ROOT / "dataset" / "provided"
OWN_DIR = REPO_ROOT / "dataset" / "own"
AXIS_MAP_PATH = REPO_ROOT / "training" / "axis_map.json"

IDENTITY_AXIS_MAP = ("x", "y", "z")


def load_recording(path: Path) -> np.ndarray:
    """Load one ``ax,ay,az`` CSV recording as an (N, 3) int array of raw counts."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != NUM_AXES or len(data) == 0:
        raise ValueError(f"{path}: expected an ax,ay,az CSV with at least one row")
    return data


@dataclass(frozen=True)
class Recording:
    path: Path
    label: int


def discover(root: Path, class_names=CLASS_NAMES) -> list[Recording]:
    """Find ``root/<class>/*.txt`` recordings (label = index in class_names)."""
    records = []
    for label, name in enumerate(class_names):
        for path in sorted((root / name).glob("*.txt")):
            records.append(Recording(path, label))
    return records


# Orientation conversion (Step 2)
def parse_axis_map(axis_map) -> tuple[np.ndarray, np.ndarray]:
    """``("-y", "x", "z")`` -> (source index per output axis, sign per output axis).

    Output axis i is the recorded axis axis_map[i], negated if prefixed with '-'.
    """
    if len(axis_map) != NUM_AXES:
        raise ValueError("axis map needs exactly three entries")
    index, sign = [], []
    for entry in axis_map:
        entry = entry.strip().lower()
        negative = entry.startswith("-")
        name = entry.lstrip("+-")
        if name not in ("x", "y", "z"):
            raise ValueError(f"bad axis map entry {entry!r}")
        index.append("xyz".index(name))
        sign.append(-1 if negative else 1)
    if sorted(index) != [0, 1, 2]:
        raise ValueError(f"axis map {axis_map} must use each axis exactly once")
    return np.asarray(index), np.asarray(sign)


def apply_axis_map(raw: np.ndarray, axis_map=IDENTITY_AXIS_MAP) -> np.ndarray:
    index, sign = parse_axis_map(axis_map)
    out = raw[:, index] * sign
    # -(-32768) does not fit in int16.
    return np.clip(out, -32768, 32767)


def load_axis_map(path: Path = AXIS_MAP_PATH) -> tuple[str, str, str]:
    if not path.exists():
        return IDENTITY_AXIS_MAP
    return tuple(json.loads(path.read_text())["axis_map"])


# Preprocessing (same as the firmware)
def _tail_mode(column: np.ndarray) -> float:
    """Most frequent value of the last PAD_MODE_WINDOW samples; ties -> most recent."""
    tail = column[-PAD_MODE_WINDOW:]
    best_value, best_count = tail[-1], 0
    for value in tail[::-1]:
        count = int(np.sum(tail == value))
        if count > best_count:
            best_value, best_count = value, count
    return float(best_value)


def to_frames(raw: np.ndarray) -> np.ndarray:
    """Raw counts (N, 3) -> (TARGET_LENGTH, 3) float32 in g."""
    g = raw[:WINDOW_LENGTH].astype(np.float32) / np.float32(COUNTS_PER_G)
    n = len(g)
    if n < WINDOW_LENGTH:
        pad = np.empty((WINDOW_LENGTH - n, NUM_AXES), dtype=np.float32)
        for axis in range(NUM_AXES):
            pad[:, axis] = _tail_mode(g[:, axis])
        g = np.concatenate([g, pad])
    out = np.empty((TARGET_LENGTH, NUM_AXES), dtype=np.float32)
    for k in range(TARGET_LENGTH):
        begin = k * WINDOW_LENGTH // TARGET_LENGTH
        end = max(begin + 1, (k + 1) * WINDOW_LENGTH // TARGET_LENGTH)
        out[k] = g[begin:end].mean(axis=0)
    return out


def load_frames(records: list[Recording], axis_map=IDENTITY_AXIS_MAP):
    """Records -> (X frames (n, 128, 3) in g, y labels, raw sample counts)."""
    frames, labels, lengths = [], [], []
    for rec in records:
        raw = apply_axis_map(load_recording(rec.path), axis_map)
        frames.append(to_frames(raw))
        labels.append(rec.label)
        lengths.append(len(raw))
    if not frames:
        return (np.zeros((0, TARGET_LENGTH, NUM_AXES), np.float32),
                np.zeros((0,), np.int64), np.zeros((0,), np.int64))
    return np.stack(frames), np.asarray(labels, np.int64), np.asarray(lengths)


def fit_normalizer(frames: np.ndarray) -> dict:
    return {
        "mean": frames.mean(axis=(0, 1)).astype(np.float32),
        "std": np.maximum(frames.std(axis=(0, 1)), 1e-3).astype(np.float32),
    }


def normalize(frames: np.ndarray, stats: dict) -> np.ndarray:
    """(n, 128, 3) g -> (n, 128, 3, 1) normalized model input."""
    x = (frames - stats["mean"]) / stats["std"]
    return x.astype(np.float32)[..., np.newaxis]


# Augmentation: random speed, start time, rotation, intensity and noise.
def _random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def augment(frames: np.ndarray, labels: np.ndarray, copies: int, seed: int = 0,
            max_rotation_deg: float = 15.0):
    """Return the originals plus ``copies`` randomly perturbed versions of each."""
    rng = np.random.default_rng(seed)
    out_x, out_y = [frames], [labels]
    t = np.arange(TARGET_LENGTH, dtype=np.float32)
    for _ in range(copies):
        batch = np.empty_like(frames)
        for i, seq in enumerate(frames):
            stretch = rng.uniform(0.85, 1.15)          # gesture speed
            shift = rng.uniform(-10, 10)               # start time (frames)
            src = np.clip((t - shift) / stretch, 0, TARGET_LENGTH - 1)
            warped = np.stack([np.interp(src, t, seq[:, a]) for a in range(NUM_AXES)], axis=1)
            warped = warped @ _random_rotation(rng, max_rotation_deg).T
            warped *= rng.uniform(0.9, 1.1)            # intensity
            warped += rng.normal(0, 0.01, size=warped.shape)
            batch[i] = warped
        out_x.append(batch)
        out_y.append(labels)
    return np.concatenate(out_x).astype(np.float32), np.concatenate(out_y)


def stratified_folds(labels: np.ndarray, k: int, seed: int = 42):
    """Yield (train_idx, val_idx) for k stratified folds."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label))
        fold_of[idx] = np.arange(len(idx)) % k
    for fold in range(k):
        yield np.flatnonzero(fold_of != fold), np.flatnonzero(fold_of == fold)
