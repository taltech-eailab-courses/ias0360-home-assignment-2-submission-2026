"""Steps 1, 4, 5 and 6: train, optimize and compare models.

Each experiment runs a stratified 5-fold cross-validation over the provided and
own recordings; fold k trains on 4/5 of both and is scored on the held-out fifth:
  * validation accuracy = held-out provided recordings (mean over folds),
  * own data accuracy   = held-out own recordings, uint8 TFLite model,
  * average inference time = uint8 TFLite time per held-out own recording (PC),
  * memory = parameters, float32 weight size, uint8 .tflite size.
Rows with own=False train on the provided data only.

The deployed model trains on all provided recordings + own folds 1-4; own fold 0
is held out and written to results/own_test_split.json.

Results go to results/experiments.jsonl (finished experiments are skipped, use
--force to redo) and results/optimization_table.md / .csv.

Usage (inside the course Docker container):
    python training/run_experiments.py              # everything
    python training/run_experiments.py --only initial,aug_x10
    python training/run_experiments.py --quick      # 2 folds, few epochs: smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import gesture_data as gd
import models

RESULTS_DIR = gd.REPO_ROOT / "training" / "results"
MODELS_DIR = RESULTS_DIR / "models"
JSONL_PATH = RESULTS_DIR / "experiments.jsonl"
SEED = 10

# fmt: off
# Each entry overrides only the keys it lists on top of BASE.
BASE = dict(arch="lecture_cnn", model={}, optimizer="adam", lr=1e-3, epochs=30, batch=16, aug=0, own=True)
EXPERIMENTS = [
    dict(name="initial_provided_only", desc="Initial model trained on the provided dataset only (no own recordings in training)", own=False),
    dict(name="best_provided_only",   desc="Step 6 MobileNet-style CNN trained on the provided dataset only (no own recordings in training)",
         own=False, arch="mobilenet_ds_cnn", model=dict(width=16, dropout=0.2), aug=10, epochs=80),
    dict(name="initial",             desc="Initial model (lab 2.2 CNN: 8/16 filters, dense 16, ReLU, dropout 0.1; Adam lr=0.001, 30 epochs, batch 16)"),
    dict(name="ep10_lr0.05",          desc="Number of epochs = 10; learning rate = 0.05",       epochs=10, lr=0.05),
    dict(name="ep10",                 desc="Number of epochs = 10",                             epochs=10),
    dict(name="ep60",                 desc="Number of epochs = 60",                             epochs=60),
    dict(name="ep100",                desc="Number of epochs = 100",                            epochs=100),
    dict(name="lr0.01",               desc="Learning rate = 0.01",                              lr=0.01),
    dict(name="lr0.0003_ep60",        desc="Learning rate = 0.0003, 60 epochs",                 lr=3e-4, epochs=60),
    dict(name="sgd_lr0.01_ep60",      desc="Optimizer SGD (momentum 0.9), lr = 0.01, 60 epochs", optimizer="sgd", lr=0.01, epochs=60),
    dict(name="rmsprop",              desc="Optimizer RMSprop, lr = 0.001",                     optimizer="rmsprop"),
    dict(name="batch8",               desc="Batch size = 8",                                    batch=8),
    dict(name="batch32_ep60",         desc="Batch size = 32, 60 epochs",                        batch=32, epochs=60),
    dict(name="act_tanh",             desc="Activation = tanh",                                 model=dict(activation="tanh")),
    dict(name="dropout0",             desc="No Dropout",                                        model=dict(dropout=0.0)),
    dict(name="dropout0.3",           desc="Dropout = 0.3",                                     model=dict(dropout=0.3)),
    dict(name="filters16_32_dense32", desc="More neurons: 16/32 filters, dense 32",             model=dict(num_filters1=16, num_filters2=32, dense_units=32)),
    dict(name="extra_conv_layer",     desc="One more Conv2D+MaxPool layer",                     model=dict(extra_conv_block=True)),
    dict(name="aug_x10",              desc="Data augmentation x10 (time warp/shift, rotation +-15 deg, scale, noise)", aug=10),
    dict(name="combined_best",        desc="Combined: 16/32 filters, dense 32, dropout 0.3, augmentation x10, 80 epochs",
         model=dict(num_filters1=16, num_filters2=32, dense_units=32, dropout=0.3), aug=10, epochs=80),
    # Step 6: other architectures
    dict(name="mobilenet_ds_cnn",     desc="Step 6: MobileNet-style depthwise-separable CNN (augmentation x10, 80 epochs)",
         arch="mobilenet_ds_cnn", model=dict(width=16, dropout=0.2), aug=10, epochs=80),
    dict(name="squeezenet_cnn",       desc="Step 6: SqueezeNet-style Fire-module CNN (augmentation x10, 80 epochs)",
         arch="squeezenet_cnn", model=dict(width=16, dropout=0.2), aug=10, epochs=80),
]
# fmt: on


def make_optimizer(name: str, lr: float):
    if name == "adam":
        return tf.keras.optimizers.Adam(lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(lr)
    raise ValueError(name)


def train(cfg: dict, x: np.ndarray, y: np.ndarray, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    model = models.build(cfg["arch"], len(gd.CLASS_NAMES), **cfg["model"])
    model.compile(optimizer=make_optimizer(cfg["optimizer"], cfg["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=cfg["epochs"], batch_size=cfg["batch"], shuffle=True, verbose=0)
    return model


def prepare_train(frames, labels, cfg, stats, seed):
    if cfg["aug"]:
        frames, labels = gd.augment(frames, labels, cfg["aug"], seed=seed)
    return gd.normalize(frames, stats), labels


def to_uint8_tflite(model: tf.keras.Model, cfg: dict, representative: np.ndarray) -> bytes:
    def rep():
        for i in range(min(200, len(representative))):
            yield [representative[i:i + 1]]

    # Static batch size 1 so Flatten becomes a single RESHAPE op.
    static = models.build(cfg["arch"], len(gd.CLASS_NAMES), batch_size=1, **cfg["model"])
    static.set_weights(model.get_weights())
    converter = tf.lite.TFLiteConverter.from_keras_model(static)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()


def evaluate_tflite(tflite_bytes: bytes, x: np.ndarray, y: np.ndarray):
    """Accuracy and mean per-gesture latency (ms) of a uint8 TFLite model on this PC."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero_point = inp["quantization"]
    correct, elapsed = 0, 0.0
    for xi, yi in zip(x, y):
        q = np.clip(np.round(xi / scale) + zero_point, 0, 255).astype(np.uint8)
        interpreter.set_tensor(inp["index"], q[np.newaxis])
        start = time.perf_counter()
        interpreter.invoke()
        elapsed += time.perf_counter() - start
        correct += int(np.argmax(interpreter.get_tensor(out["index"])[0]) == yi)
    n = max(len(x), 1)
    return correct / n, 1000.0 * elapsed / n


def train_set(frames, labels, own_frames, own_labels, p_idx, o_idx, use_own):
    if use_own and len(o_idx):
        return (np.concatenate([frames[p_idx], own_frames[o_idx]]),
                np.concatenate([labels[p_idx], own_labels[o_idx]]))
    return frames[p_idx], labels[p_idx]


def run_experiment(cfg, provided, own, folds, quick):
    frames, labels = provided
    own_frames, own_labels = own
    if quick:
        cfg = dict(cfg, epochs=max(2, cfg["epochs"] // 10), aug=min(cfg["aug"], 1))
    has_own = len(own_frames) > 0
    p_folds = list(gd.stratified_folds(labels, folds, seed=42))
    o_folds = list(gd.stratified_folds(own_labels, folds, seed=43)) if has_own else None
    no_idx = np.zeros(0, dtype=np.int64)

    # k-fold CV over provided + own
    fold_acc, own_correct, own_total, own_ms = [], 0, 0, 0.0
    for k in range(folds):
        p_tr, p_va = p_folds[k]
        o_tr, o_va = o_folds[k] if has_own else (no_idx, no_idx)
        x_f, y_f = train_set(frames, labels, own_frames, own_labels, p_tr, o_tr, cfg["own"])
        stats = gd.fit_normalizer(x_f)
        x_tr, y_tr = prepare_train(x_f, y_f, cfg, stats, seed=SEED + k)
        model = train(cfg, x_tr, y_tr, seed=SEED + k)
        _, acc = model.evaluate(gd.normalize(frames[p_va], stats), labels[p_va], verbose=0)
        fold_acc.append(acc)
        if has_own:
            tfl = to_uint8_tflite(model, cfg, gd.normalize(x_f, stats))
            acc_o, ms = evaluate_tflite(tfl, gd.normalize(own_frames[o_va], stats), own_labels[o_va])
            own_correct += round(acc_o * len(o_va))
            own_total += len(o_va)
            own_ms += ms * len(o_va)

    # Deployed model: all provided + own folds 1..k-1 (own fold 0 held out)
    held_out = o_folds[0][1] if has_own else no_idx
    keep = np.setdiff1d(np.arange(len(own_frames)), held_out)
    x_f, y_f = train_set(frames, labels, own_frames, own_labels, np.arange(len(frames)), keep, cfg["own"])
    stats = gd.fit_normalizer(x_f)
    x_all, y_all = prepare_train(x_f, y_f, cfg, stats, seed=SEED)
    start = time.perf_counter()
    model = train(cfg, x_all, y_all, seed=SEED)
    train_time = time.perf_counter() - start
    tflite = to_uint8_tflite(model, cfg, gd.normalize(x_f, stats))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / f"{cfg['name']}.tflite").write_bytes(tflite)
    model.save(MODELS_DIR / f"{cfg['name']}.keras")

    if has_own:
        own_acc_uint8 = own_correct / own_total
        latency_ms = own_ms / own_total
        latency_source = f"held-out own data ({own_total} gestures)"
        heldout_acc, _ = evaluate_tflite(tflite, gd.normalize(own_frames[held_out], stats), own_labels[held_out])
    else:
        own_acc_uint8 = heldout_acc = None
        _, latency_ms = evaluate_tflite(tflite, gd.normalize(frames, stats), labels)
        latency_source = f"provided data ({len(frames)} gestures) - no own data yet"

    params = int(model.count_params())
    return {
        "name": cfg["name"], "description": cfg["desc"], "arch": cfg["arch"], "own_in_training": cfg["own"],
        "config": {k: cfg[k] for k in ("model", "optimizer", "lr", "epochs", "batch", "aug", "own")},
        "val_acc_mean": float(np.mean(fold_acc)), "val_acc_std": float(np.std(fold_acc)),
        "val_acc_folds": [float(a) for a in fold_acc],
        "own_acc_uint8": own_acc_uint8, "own_count": own_total,
        "deployed_heldout_acc_uint8": heldout_acc, "deployed_heldout_count": int(len(held_out)),
        "latency_ms_host_uint8": latency_ms, "latency_source": latency_source,
        "params": params, "keras_float32_kb": params * 4 / 1024, "tflite_uint8_kb": len(tflite) / 1024,
        "train_time_s": train_time, "quick": quick,
        "normalizer": {k: [float(v) for v in stats[k]] for k in ("mean", "std")},
    }


def fmt_pct(value):
    return "n/a" if value is None else f"{100 * value:.1f} %"


def write_tables(rows):
    order = {e["name"]: i for i, e in enumerate(EXPERIMENTS)}
    rows = sorted(rows, key=lambda r: order.get(r["name"], 999))
    header = ("| Optimization method | Inference average speed* | Memory size** (params / float32 / uint8 TFLite) "
              "| Validation accuracy (5-fold CV, provided) | Own data accuracy*** (uint8 TFLite) |\n"
              "|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        own = fmt_pct(r["own_acc_uint8"])
        if r["own_acc_uint8"] is not None:
            own += f" ({round(r['own_acc_uint8'] * r['own_count'])}/{r['own_count']})"
        lines.append(
            f"| {r['description']} | {r['latency_ms_host_uint8']:.3f} ms "
            f"| {r['params']:,} / {r['keras_float32_kb']:.1f} KB / {r['tflite_uint8_kb']:.1f} KB "
            f"| {100 * r['val_acc_mean']:.1f} % ± {100 * r['val_acc_std']:.1f} | {own} |")
    sources = sorted({r["latency_source"] for r in rows})
    note = ("\n\\* Total uint8 TFLite prediction time divided by the number of gestures, measured on the "
            f"training PC ({'; '.join(sources)}). On-device (Pico 2 W / RP2350) times for the deployed "
            "models are printed by the `own_data_test` and `live_inference` firmware.\n"
            "\\*\\* Total parameters as reported by `model.summary()` (incl. BatchNorm statistics, which "
            "are folded into the convolutions on conversion), float32 weight size, and the size of the "
            "deployed uint8 `.tflite` flatbuffer (weights + graph).\n"
            "\\*\\*\\* Every own recording is classified exactly once, by the fold model that did not train on "
            "it. Rows marked 'provided dataset only' never see own recordings in training; all other rows "
            "train on provided + own recordings of the other folds.\n")
    if any(r.get("quick") for r in rows):
        note += "\n> WARNING: some rows come from a --quick smoke-test run and are not real results.\n"
    (RESULTS_DIR / "optimization_table.md").write_text(header + "\n".join(lines) + "\n" + note)

    import csv
    with open(RESULTS_DIR / "optimization_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "description", "own_in_training", "latency_ms_host_uint8", "params",
                         "keras_float32_kb", "tflite_uint8_kb", "val_acc_mean", "val_acc_std",
                         "own_acc_uint8", "own_count"])
        for r in rows:
            writer.writerow([r["name"], r["description"], r.get("own_in_training"),
                             round(r["latency_ms_host_uint8"], 4), r["params"],
                             round(r["keras_float32_kb"], 2), round(r["tflite_uint8_kb"], 2),
                             round(r["val_acc_mean"], 4), round(r["val_acc_std"], 4),
                             r["own_acc_uint8"], r["own_count"]])


def load_rows():
    if not JSONL_PATH.exists():
        return {}
    rows = {}
    for line in JSONL_PATH.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["name"]] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated experiment names")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="re-run experiments already in experiments.jsonl")
    parser.add_argument("--quick", action="store_true", help="smoke test: 2 folds, epochs/10")
    args = parser.parse_args()

    selected = [dict(BASE, **e) for e in EXPERIMENTS]
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {e["name"] for e in selected}
        if unknown:
            raise SystemExit(f"unknown experiments: {sorted(unknown)}")
        selected = [e for e in selected if e["name"] in wanted]

    provided_frames, provided_labels, provided_len = gd.load_frames(gd.discover(gd.PROVIDED_DIR))
    axis_map = gd.load_axis_map()
    own_frames, own_labels, _ = gd.load_frames(gd.discover(gd.OWN_DIR), axis_map)
    print(f"provided: {len(provided_frames)} recordings "
          f"({provided_len.min()}-{provided_len.max()} samples each), "
          f"own: {len(own_frames)} recordings (axis map {axis_map})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    folds = 2 if args.quick else args.folds

    # Own recordings held out from the deployed models (own fold 0).
    own_records = gd.discover(gd.OWN_DIR)
    if own_records:
        held_out = list(gd.stratified_folds(own_labels, folds, seed=43))[0][1]
        split = sorted(f"{gd.CLASS_NAMES[own_records[i].label]}/{own_records[i].path.name}" for i in held_out)
        (RESULTS_DIR / "own_test_split.json").write_text(json.dumps({"held_out": split}, indent=1) + "\n")
        print(f"deployed models hold out {len(split)} own recordings -> results/own_test_split.json")
    for cfg in selected:
        if cfg["name"] in rows and not args.force and rows[cfg["name"]].get("quick") == args.quick:
            print(f"skip {cfg['name']} (already done)")
            continue
        t0 = time.time()
        row = run_experiment(cfg, (provided_frames, provided_labels), (own_frames, own_labels),
                             folds, args.quick)
        rows[cfg["name"]] = row
        JSONL_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        write_tables(list(rows.values()))
        print(f"{cfg['name']:<22} val {100 * row['val_acc_mean']:5.1f}% ±{100 * row['val_acc_std']:4.1f} "
              f"own {fmt_pct(row['own_acc_uint8']):>8}  {row['params']:>6} params "
              f"{row['tflite_uint8_kb']:6.1f} KB  {row['latency_ms_host_uint8']:.3f} ms  ({time.time() - t0:.0f}s)")
    write_tables(list(rows.values()))
    print(f"table: {RESULTS_DIR / 'optimization_table.md'}")


if __name__ == "__main__":
    main()
