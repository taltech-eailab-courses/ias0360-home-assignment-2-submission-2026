from __future__ import annotations

import csv
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras

CLASS_NAMES = ("negative", "ring", "slope", "wing", "ramp", "triangle")
COUNTS_PER_G = 16384.0
SAMPLE_RATE_HZ = 2000.0
INPUT_WINDOW = 6000
TARGET_LENGTH = 128
NUM_AXES = 3
TAIL_MODE_WINDOW = 100

ARCHITECTURE_RUNS = (
    {"name": "baseline_cnn", "architecture": "cnn", "filters": (8, 16), "dense_units": 24, "dropout": 0.0, "learning_rate": 1e-3, "batch_size": 16},
    {"name": "lstm_baseline", "architecture": "lstm", "filters": (8, 16), "dense_units": 24, "lstm_units": 12, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 16},
    {"name": "lstm_24_units", "architecture": "lstm", "filters": (8, 16), "dense_units": 24, "lstm_units": 24, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 16},
    {"name": "lstm_32_regularized", "architecture": "lstm", "filters": (8, 16), "dense_units": 24, "lstm_units": 32, "dropout": 0.2, "learning_rate": 5e-4, "batch_size": 32},
    {"name": "lstm_48_regularized", "architecture": "lstm", "filters": (8, 16), "dense_units": 24, "lstm_units": 48, "dropout": 0.25, "learning_rate": 5e-4, "batch_size": 16},
    {"name": "lstm_24_fast", "architecture": "lstm", "filters": (8, 16), "dense_units": 24, "lstm_units": 24, "dropout": 0.0, "learning_rate": 2e-3, "batch_size": 32},
    {"name": "lstm_24_rmsprop", "architecture": "lstm", "filters": (8, 16), "dense_units": 24, "lstm_units": 24, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 16, "optimizer": "rmsprop"},
    {"name": "separable_cnn", "architecture": "separable", "filters": (8, 16), "dense_units": 24, "dropout": 0.1, "learning_rate": 5e-4, "batch_size": 16},
    {"name": "tuned_cnn", "architecture": "cnn", "filters": (8, 16), "dense_units": 24, "dropout": 0.1, "learning_rate": 5e-4, "batch_size": 32},
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_recording(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.ndim != 2 or values.shape[1] != NUM_AXES or len(values) < 2:
        raise ValueError(f"{path}: expected at least two ax,ay,az rows")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 32768):
        raise ValueError(f"{path}: invalid signed sensor counts")
    return values.astype(np.float32)


def split_recordings(dataset_dir: Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}
    counts = {}
    for label, class_name in enumerate(CLASS_NAMES):
        files = sorted((Path(dataset_dir) / class_name).glob("*.txt"))
        if len(files) < 6:
            raise ValueError(f"{dataset_dir}/{class_name}: need at least 6 recordings; found {len(files)}")
        shuffled = [files[i] for i in rng.permutation(len(files))]
        n_train = int(0.70 * len(shuffled))
        n_validation = int(0.15 * len(shuffled))
        groups = {
            "train": shuffled[:n_train],
            "validation": shuffled[n_train:n_train + n_validation],
            "test": shuffled[n_train + n_validation:],
        }
        counts[class_name] = {name: len(items) for name, items in groups.items()}
        for split_name, paths in groups.items():
            splits[split_name].extend((path, label) for path in paths)
    for records in splits.values():
        rng.shuffle(records)
    return splits, counts


def _tail_mode(values: np.ndarray) -> np.ndarray:
    tail = values[-min(len(values), TAIL_MODE_WINDOW):]
    result = np.empty(NUM_AXES, dtype=np.float32)
    for axis in range(NUM_AXES):
        counts = {}
        for value in tail[:, axis]:
            counts[value.item()] = counts.get(value.item(), 0) + 1
        maximum = max(counts.values())
        result[axis] = next(value for value in tail[::-1, axis] if counts[value.item()] == maximum)
    return result


def preprocess_recording(raw: np.ndarray, mean: np.ndarray, std: np.ndarray, target_length: int = TARGET_LENGTH):
    raw = raw[:INPUT_WINDOW]
    normalized = (raw / COUNTS_PER_G - mean) / std
    if len(normalized) < INPUT_WINDOW:
        mode = _tail_mode(normalized)
        padding = np.repeat(mode[None, :], INPUT_WINDOW - len(normalized), axis=0)
        normalized = np.concatenate((normalized, padding), axis=0)
    pooled = np.empty((target_length, NUM_AXES), dtype=np.float32)
    for index in range(target_length):
        start = index * INPUT_WINDOW // target_length
        end = max(start + 1, (index + 1) * INPUT_WINDOW // target_length)
        pooled[index] = normalized[start:end].mean(axis=0)
    return pooled[..., None]


def prepare_dataset(dataset_dir: Path, seed: int = 42, target_length: int = TARGET_LENGTH):
    splits, file_counts = split_recordings(dataset_dir, seed)
    train_raw = [load_recording(path)[:INPUT_WINDOW] for path, _ in splits["train"]]
    training_g = np.concatenate(train_raw, axis=0) / COUNTS_PER_G
    mean = training_g.mean(axis=0).astype(np.float32)
    std = np.maximum(training_g.std(axis=0), 1e-3).astype(np.float32)
    prepared = {"mean": mean, "std": std, "file_counts": file_counts}
    for split_name, records in splits.items():
        features, labels, filenames = [], [], []
        for path, label in records:
            features.append(preprocess_recording(load_recording(path), mean, std, target_length))
            labels.append(label)
            filenames.append(str(path))
        prepared[f"X_{split_name}"] = np.stack(features).astype(np.float32)
        prepared[f"y_{split_name}"] = np.asarray(labels, dtype=np.int64)
        prepared[f"files_{split_name}"] = np.asarray(filenames)
    return prepared


def build_model(config: dict):
    target_length = config.get("target_length", TARGET_LENGTH)
    inputs = tf_keras.Input(shape=(target_length, NUM_AXES, 1), name="imu_window")
    architecture = config["architecture"]
    filters = config["filters"]
    if architecture == "lstm":
        x = tf_keras.layers.Reshape((target_length, NUM_AXES))(inputs)
        x = tf_keras.layers.LSTM(
            config["lstm_units"],
            activation="tanh",
            recurrent_activation="sigmoid",
        )(x)
    elif architecture == "separable":
        x = tf_keras.layers.DepthwiseConv2D((7, 3), padding="same", activation="relu")(inputs)
        x = tf_keras.layers.Conv2D(filters[0], (1, 1), activation="relu")(x)
        x = tf_keras.layers.MaxPooling2D((2, 1))(x)
        x = tf_keras.layers.SeparableConv2D(filters[1], (5, 1), padding="same", activation="relu")(x)
        x = tf_keras.layers.MaxPooling2D((2, 1))(x)
        x = tf_keras.layers.Flatten()(x)
        x = tf_keras.layers.Dense(config["dense_units"], activation="relu")(x)
    elif architecture == "cnn":
        x = tf_keras.layers.Conv2D(filters[0], (7, 3), padding="same", activation="relu")(inputs)
        x = tf_keras.layers.MaxPooling2D((2, 1))(x)
        x = tf_keras.layers.Conv2D(filters[1], (5, 1), padding="same", activation="relu")(x)
        x = tf_keras.layers.MaxPooling2D((2, 1))(x)
        x = tf_keras.layers.Flatten()(x)
        x = tf_keras.layers.Dense(config["dense_units"], activation="relu")(x)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    if config["dropout"]:
        x = tf_keras.layers.Dropout(config["dropout"])(x)
    outputs = tf_keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf_keras.Model(inputs, outputs, name=config["name"])
    optimizer_name = config.get("optimizer", "adam").lower()
    optimizer_types = {
        "adam": tf_keras.optimizers.Adam,
        "rmsprop": tf_keras.optimizers.RMSprop,
    }
    if optimizer_name not in optimizer_types:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    model.compile(optimizer=optimizer_types[optimizer_name](config["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def benchmark_keras(model, X: np.ndarray, limit: int = 48) -> float:
    infer = tf.function(lambda batch: model(batch, training=False))
    samples = tf.convert_to_tensor(X[:limit], dtype=tf.float32)
    infer(samples[:1])
    timings = []
    for index in range(len(samples)):
        start = time.perf_counter_ns()
        infer(samples[index:index + 1])
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.mean(timings))


def train_architectures(data: dict, artifacts_dir: Path, epochs: int = 60):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trained, rows = {}, []
    for run_index, config in enumerate(ARCHITECTURE_RUNS):
        config = {**config, "target_length": int(data["X_train"].shape[1])}
        tf_keras.backend.clear_session()
        seed_everything(42 + run_index)
        model = build_model(config)
        callbacks = [
            tf_keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
            tf_keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ]
        history = model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_validation"], data["y_validation"]),
            epochs=epochs, batch_size=config["batch_size"], callbacks=callbacks, verbose=2,
        )
        _, train_accuracy = model.evaluate(data["X_train"], data["y_train"], verbose=0)
        val_loss, val_accuracy = model.evaluate(data["X_validation"], data["y_validation"], verbose=0)
        _, test_accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        model_path = artifacts_dir / f"{config['name']}.keras"
        model.save(model_path)
        host_inference_ms = benchmark_keras(model, data["X_test"])
        trained[config["name"]] = {"model": model, "config": config, "history": history.history, "validation_accuracy": float(val_accuracy)}
        optimizer_name = config.get("optimizer", "adam").lower()
        rows.append({
            "model": config["name"], "method": "float architecture / hyperparameter trial",
            "parameters": int(model.count_params()), "train_accuracy": float(train_accuracy),
            "validation_accuracy": float(val_accuracy),
            "test_accuracy": float(test_accuracy), "own_data_accuracy": float(test_accuracy),
            "model_bytes": model_path.stat().st_size, "host_inference_ms": round(host_inference_ms, 4),
            "pico_inference_ms": "measure on Pico", "epochs_run": len(history.history["loss"]),
            "optimizer": "RMSprop" if optimizer_name == "rmsprop" else "Adam",
            "learning_rate": config["learning_rate"], "batch_size": config["batch_size"],
        })
    best_name = max(trained, key=lambda name: trained[name]["validation_accuracy"])
    return trained, best_name, rows


def train_pruned_and_qat(best_model, data: dict, artifacts_dir: Path, epochs: int = 30):
    import tensorflow_model_optimization as tfmot
    artifacts_dir = Path(artifacts_dir)
    batch_size = 16
    steps = math.ceil(len(data["y_train"]) / batch_size) * epochs
    schedule = tfmot.sparsity.keras.PolynomialDecay(0.0, 0.5, 0, max(steps, 1))
    pruned = tfmot.sparsity.keras.prune_low_magnitude(best_model, pruning_schedule=schedule)
    pruned.compile(optimizer=tf_keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    pruned_history = pruned.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_validation"], data["y_validation"]),
        epochs=epochs, batch_size=batch_size,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=2,
    )
    stripped = tfmot.sparsity.keras.strip_pruning(pruned)
    stripped.save(artifacts_dir / "pruned_float.keras")
    qat = tfmot.quantization.keras.quantize_model(best_model)
    qat.compile(optimizer=tf_keras.optimizers.Adam(2e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    qat_history = qat.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_validation"], data["y_validation"]),
        epochs=epochs, batch_size=batch_size,
        callbacks=[tf_keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)],
        verbose=2,
    )
    qat.save(artifacts_dir / "qat.keras")
    weights = stripped.get_weights()
    nonzero = sum(np.count_nonzero(weight) for weight in weights)
    total = sum(weight.size for weight in weights)
    training_info = {
        "pruned_epochs": len(pruned_history.history["loss"]),
        "qat_epochs": len(qat_history.history["loss"]),
    }
    return {"pruned": stripped, "qat": qat}, 1.0 - nonzero / total, training_info


def convert_to_int8(model, representative_data: np.ndarray) -> bytes:
    if any(isinstance(layer, tf_keras.layers.LSTM) for layer in model.layers):
        source_model = model
        fixed_input = tf_keras.Input(
            batch_shape=(1, *model.input_shape[1:]),
            name=model.input_names[0],
        )
        model = tf_keras.models.clone_model(model, input_tensors=fixed_input)
        model.set_weights(source_model.get_weights())
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        for sample in representative_data[:100]:
            yield [sample[np.newaxis, ...].astype(np.float32)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()


def evaluate_tflite(model_content: bytes, X: np.ndarray, y: np.ndarray):
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("X and y must contain the same nonzero number of examples")
    interpreter = tf.lite.Interpreter(model_content=model_content, num_threads=1)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    if input_info["dtype"] != np.uint8:
        raise ValueError(f"Expected uint8 model input, got {input_info['dtype']}")
    input_scale, input_zero = input_info["quantization"]
    if input_scale <= 0:
        raise ValueError("Model input has an invalid quantization scale")
    if output_info["dtype"] != np.uint8:
        raise ValueError(f"Expected uint8 model output, got {output_info['dtype']}")
    predictions, timings = [], []
    for index, sample in enumerate(X):
        quantized = np.clip(np.rint(sample / input_scale + input_zero), 0, 255).astype(np.uint8)
        interpreter.set_tensor(input_info["index"], quantized[np.newaxis, ...])
        if index == 0:
            interpreter.invoke()
        start = time.perf_counter_ns()
        interpreter.invoke()
        timings.append((time.perf_counter_ns() - start) / 1e6)
        predictions.append(int(np.argmax(interpreter.get_tensor(output_info["index"])[0])))
    return {
        "accuracy": float(np.mean(np.asarray(predictions) == y)),
        "average_ms": float(np.mean(timings)),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "input_quantization": input_info["quantization"],
        "output_quantization": output_info["quantization"],
    }


def measure_pico_latency(log_path: Path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    timings = [
        float(value)
        for value in re.findall(r"Inference time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text)
    ]
    if not timings:
        raise ValueError(f"No 'Inference time: ... ms' measurements found in {log_path}")
    values = np.asarray(timings, dtype=np.float64)
    return {
        "count": len(timings),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "std_ms": float(values.std()),
    }


def write_results_csv(rows, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "method", "parameters", "train_accuracy", "validation_accuracy", "test_accuracy", "own_data_accuracy", "model_bytes", "host_inference_ms", "pico_inference_ms", "epochs_run", "optimizer", "learning_rate", "batch_size"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pico_artifacts(model_content: bytes, mean: np.ndarray, std: np.ndarray, pico_dir: Path, class_names=CLASS_NAMES, target_length: int = TARGET_LENGTH):
    pico_dir = Path(pico_dir)
    model_dir = pico_dir / "lib" / "model"
    settings_path = pico_dir / "src" / "preprocessing_settings.hpp"
    model_dir.mkdir(parents=True, exist_ok=True)
    (pico_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (pico_dir / "artifacts" / "gesture_model.tflite").write_bytes(model_content)
    input_details = tf.lite.Interpreter(model_content=model_content).get_input_details()[0]
    scale, zero_point = input_details["quantization"]
    if scale <= 0:
        raise ValueError("Exported model has invalid input quantization")
    model_bytes = list(model_content)
    lines = [", ".join(f"0x{value:02x}" for value in model_bytes[i:i + 12]) for i in range(0, len(model_bytes), 12)]
    source = '#include "gesture_model_data.h"\n\nalignas(8) const unsigned char gesture_model_data[] = {\n  ' + ",\n  ".join(lines) + '\n};\nconst int gesture_model_data_len = sizeof(gesture_model_data);\n'
    (model_dir / "gesture_model_data.cpp").write_text(source, encoding="ascii")
    mean_values = ", ".join(f"{value:.10g}f" for value in mean)
    std_values = ", ".join(f"{value:.10g}f" for value in std)
    settings = f'''#ifndef PREPROCESSING_SETTINGS_HPP
#define PREPROCESSING_SETTINGS_HPP
#include <cstdint>
constexpr int kNumAxes = 3;
constexpr int kInputWindowLength = {INPUT_WINDOW};
constexpr int kTargetLength = {target_length};
constexpr int kPaddingModeWindow = {TAIL_MODE_WINDOW};
constexpr float kCountsPerG = {COUNTS_PER_G:.1f}f;
constexpr float kAxisMean[kNumAxes] = {{{mean_values}}};
constexpr float kAxisStd[kNumAxes] = {{{std_values}}};
constexpr float kInputScale = {scale:.10g}f;
constexpr int32_t kInputZeroPoint = {zero_point};
#endif
'''
    settings_path.write_text(settings, encoding="ascii")
    labels = ", ".join(f'"{name.title()}"' for name in class_names)
    model_settings = f'''#ifndef TFLITE_INFERENCE_TEST_MODEL_SETTINGS_H_
#define TFLITE_INFERENCE_TEST_MODEL_SETTINGS_H_
#define GESTURE_DATA_IMU_AXIS 3
#define GESTURE_DATA_MAX_LEN {target_length}
#define MAX_GESTURE_RAW_DATA_LEN {INPUT_WINDOW}
#define CLASS_COUNT {len(class_names)}
#define CLASS_LABELS {{{labels}}}
constexpr int arena_size = 60 * 1024;
#endif
'''
    (model_dir / "model_settings.h").write_text(model_settings, encoding="ascii")
    (model_dir / "labels.txt").write_text("\n".join(class_names) + "\n", encoding="ascii")
    metadata = {
        "class_names": list(class_names), "input_shape": [1, target_length, NUM_AXES, 1],
        "input_dtype": "uint8", "input_scale": float(scale), "input_zero_point": int(zero_point),
        "axis_mean": [float(value) for value in mean], "axis_std": [float(value) for value in std],
        "raw_window_samples": INPUT_WINDOW, "sample_rate_hz": SAMPLE_RATE_HZ, "counts_per_g": COUNTS_PER_G,
    }
    (pico_dir / "artifacts" / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
