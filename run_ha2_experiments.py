#!/usr/bin/env python3
"""
IAS0360 Home Assignment 2, image domain: MNIST and 10 digits drawn on the Pico touchscreen.
Joosep Parts (256780IAXD)

Steps 1 to 6: trains the models, tries the optimisations, compares three architectures, writes
the tables and figures to output/ and the selected int8 model to firmware/inference/models/.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Helper Utilities: Preprocessing, Quantization, Timing, C++ Export
# ============================================================================

def load_and_preprocess_digit(path: Path, center_and_pad: bool = False) -> np.ndarray:
    """
    Convert a drawn digit into 28x28 grayscale in [0, 1], white on black like MNIST.
    If center_and_pad=True, crops bounding box of foreground stroke and centers
    it inside a 20x20 box padded to 28x28 (standard MNIST normalization).
    """
    img = Image.open(path).convert("L")
    if not center_and_pad:
        img_28 = img.resize((28, 28))
        arr = (255.0 - np.asarray(img_28, dtype=np.float32)) / 255.0
        return arr

    # MNIST steps: crop, fit into a 20x20 box, centre by mass in 28x28
    fg = (255.0 - np.asarray(img, dtype=np.float32)) / 255.0
    mask = fg > 0.20
    ys, xs = np.where(mask)
    if len(xs) > 5 and len(ys) > 5:
        y0, y1 = max(0, ys.min() - 2), min(fg.shape[0], ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(fg.shape[1], xs.max() + 3)
        crop = fg[y0:y1, x0:x1]
        h, w = crop.shape
        scale = 20.0 / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        crop_img = Image.fromarray((crop * 255.0).astype(np.uint8)).resize(
            (nw, nh), resample=Image.Resampling.BILINEAR
        )
        canvas = np.zeros((28, 28), dtype=np.float32)
        top = (28 - nh) // 2
        left = (28 - nw) // 2
        canvas[top:top + nh, left:left + nw] = np.asarray(crop_img, dtype=np.float32) / 255.0
        # Shift by center of mass to (14, 14)
        total = canvas.sum()
        if total > 0:
            cy, cx = np.indices(canvas.shape)
            com_y = (cy * canvas).sum() / total
            com_x = (cx * canvas).sum() / total
            dy = int(round(13.5 - com_y))
            dx = int(round(13.5 - com_x))
            canvas = np.roll(canvas, shift=(dy, dx), axis=(0, 1))
        return np.clip(canvas, 0.0, 1.0)
    else:
        img_28 = img.resize((28, 28))
        return (255.0 - np.asarray(img_28, dtype=np.float32)) / 255.0


def load_my_10_digits(digits_dir: Path):
    raw_x, centered_x, y = [], [], []
    for d in range(10):
        p = digits_dir / f"digit_{d}.png"
        raw_x.append(load_and_preprocess_digit(p, center_and_pad=False))
        centered_x.append(load_and_preprocess_digit(p, center_and_pad=True))
        y.append(d)
    raw_x = np.array(raw_x, dtype=np.float32)[..., np.newaxis]
    centered_x = np.array(centered_x, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int64)
    return raw_x, centered_x, y


def measure_keras_own_data_speed_ms(model, own_x: np.ndarray, repeats: int = 50) -> float:
    """
    Step 5 definition:
      * your own data (10 digits) total prediction time divided by N.
    Uses direct forward call model(own_x, training=False) after warm-up.
    """
    tensor_x = tf.constant(own_x, dtype=tf.float32)
    # Warm-up
    for _ in range(5):
        _ = model(tensor_x, training=False)
    n_items = len(own_x)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(tensor_x, training=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / (repeats * n_items)


def save_keras_and_get_size_kb(model, path: Path) -> float:
    model.save(str(path))
    return path.stat().st_size / 1024.0


def convert_to_int8_tflite(model, rep_data: np.ndarray, out_path: Path, io_dtype=tf.int8) -> float:
    def representative_dataset():
        for i in range(min(200, len(rep_data))):
            yield [rep_data[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # TFLite Micro on the Pico has one scale per Dense layer, not one per neuron.
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = io_dtype
    converter.inference_output_type = io_dtype
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    return len(tflite_bytes) / 1024.0


def evaluate_tflite_model(tflite_path: Path, test_x: np.ndarray, test_y: np.ndarray,
                          own_x: np.ndarray, own_y: np.ndarray, repeats_own: int = 100):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    in_dtype = in_det["dtype"]
    qmin = -128 if in_dtype == np.int8 else 0
    qmax = 127 if in_dtype == np.int8 else 255

    # 1. Accuracy on test_x
    correct = 0
    for i in range(len(test_x)):
        x = test_x[i:i + 1]
        if in_scale > 0:
            x = np.clip(np.round(x / in_scale + in_zp), qmin, qmax).astype(in_dtype)
        else:
            x = x.astype(in_dtype)
        interpreter.set_tensor(in_det["index"], x)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det["index"])[0]
        if int(np.argmax(out)) == int(test_y[i]):
            correct += 1
    val_acc = correct / len(test_x)

    # 2. Accuracy and speed on own_x (total time for N items divided by N)
    own_preds = []
    # Warm-up
    for i in range(len(own_x)):
        x = own_x[i:i + 1]
        if in_scale > 0:
            x = np.clip(np.round(x / in_scale + in_zp), qmin, qmax).astype(in_dtype)
        interpreter.set_tensor(in_det["index"], x)
        interpreter.invoke()

    t0 = time.perf_counter()
    for r in range(repeats_own):
        for i in range(len(own_x)):
            x = own_x[i:i + 1]
            if in_scale > 0:
                x = np.clip(np.round(x / in_scale + in_zp), qmin, qmax).astype(in_dtype)
            interpreter.set_tensor(in_det["index"], x)
            interpreter.invoke()
            if r == 0:
                out = interpreter.get_tensor(out_det["index"])[0]
                own_preds.append(int(np.argmax(out)))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    avg_speed_ms = elapsed_ms / (repeats_own * len(own_x))
    own_acc = float(np.mean(np.array(own_preds) == own_y))
    return val_acc, own_acc, avg_speed_ms, own_preds


def export_tflite_to_cpp(tflite_path: Path, cpp_path: Path, array_name: str, header_include: str):
    data = tflite_path.read_bytes()
    hex_lines = []
    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
    body = ",\n".join(hex_lines)
    content = (
        f'#include "{header_include}"\n\n'
        f"const int {array_name}_len = {len(data)};\n"
        f"alignas(16) const unsigned char {array_name}[] = {{\n"
        f"{body}\n"
        f"}};\n"
    )
    cpp_path.write_text(content)


# ============================================================================
# 2. Architecture Builders (Step 1, Step 4, Step 6)
# ============================================================================

def build_slide56_mlp(input_shape=(28, 28, 1), num_classes=10, activation="elu",
                      units1=80, units2=60, dropout=0.0):
    """Lecture 08 Slide 56 baseline architecture: Flatten -> Dense(80, elu) -> Dense(60, elu) -> Dense(10)."""
    layers = [
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units1, activation=activation),
    ]
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(units2, activation=activation))
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return tf.keras.Sequential(layers)


def build_custom_mnist_cnn(input_shape=(28, 28, 1), num_classes=10,
                           f1=16, f2=32, dense_units=64, activation="relu", dropout=0.0):
    """Custom 2D-CNN for MNIST digit classification."""
    layers = [
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(f1, (3, 3), padding="same", activation=activation),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(f2, (3, 3), padding="same", activation=activation),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
    ]
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.extend([
        tf.keras.layers.Dense(dense_units, activation=activation),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return tf.keras.Sequential(layers)


def build_squeezenet_micro(input_shape=(28, 28, 1), num_classes=10, pool_2d=True):
    """
    Step 6 Architecture 2: SqueezeNet-Micro (Iandola et al., 2016 adapted for 28x28 / 128x3 embedded input).
    Uses Fire modules: 1x1 squeeze convolution followed by parallel 1x1 and 3x3 expand convolutions.
    """
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D((2, 2) if pool_2d else (3, 3), padding="same")(x)

    # Fire module 1: squeeze=8, expand_1x1=16, expand_3x3=16 (output 32 ch)
    s1 = tf.keras.layers.Conv2D(8, (1, 1), padding="same", activation="relu")(x)
    e1_1 = tf.keras.layers.Conv2D(16, (1, 1), padding="same", activation="relu")(s1)
    e1_3 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(s1)
    x = tf.keras.layers.Concatenate()([e1_1, e1_3])
    x = tf.keras.layers.MaxPooling2D((2, 2) if pool_2d else (3, 1), padding="same")(x)

    # Fire module 2: squeeze=12, expand_1x1=24, expand_3x3=24 (output 48 ch)
    s2 = tf.keras.layers.Conv2D(12, (1, 1), padding="same", activation="relu")(x)
    e2_1 = tf.keras.layers.Conv2D(24, (1, 1), padding="same", activation="relu")(s2)
    e2_3 = tf.keras.layers.Conv2D(24, (3, 3), padding="same", activation="relu")(s2)
    x = tf.keras.layers.Concatenate()([e2_1, e2_3])
    x = tf.keras.layers.MaxPooling2D((2, 2) if pool_2d else (3, 1), padding="same")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="SqueezeNet_Micro")


def build_mobilenetv2_micro(input_shape=(28, 28, 1), num_classes=10, pool_2d=True):
    """
    Step 6 Architecture 3: MobileNetV2-Micro (Sandler et al., 2018 adapted for embedded input).
    Uses Inverted Residual & Linear Bottleneck blocks with DepthwiseConv2D.
    """
    bn_mom = 0.99 if pool_2d else 0.80
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization(momentum=bn_mom)(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # Inverted Residual Block 1 (expand 16 -> 32, depthwise 3x3 stride 2, project -> 24)
    m = tf.keras.layers.Conv2D(32, (1, 1), padding="same", use_bias=False)(x)
    m = tf.keras.layers.BatchNormalization(momentum=bn_mom)(m)
    m = tf.keras.layers.ReLU(max_value=6.0)(m)
    m = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(m)
    m = tf.keras.layers.BatchNormalization(momentum=bn_mom)(m)
    m = tf.keras.layers.ReLU(max_value=6.0)(m)
    x = tf.keras.layers.Conv2D(24, (1, 1), padding="same", use_bias=False)(m)
    x = tf.keras.layers.BatchNormalization(momentum=bn_mom)(x)

    # Inverted Residual Block 2 with residual skip connection (stride 1, 24 -> 48 -> 24)
    m = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False)(x)
    m = tf.keras.layers.BatchNormalization(momentum=bn_mom)(m)
    m = tf.keras.layers.ReLU(max_value=6.0)(m)
    m = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", use_bias=False)(m)
    m = tf.keras.layers.BatchNormalization(momentum=bn_mom)(m)
    m = tf.keras.layers.ReLU(max_value=6.0)(m)
    m = tf.keras.layers.Conv2D(24, (1, 1), padding="same", use_bias=False)(m)
    m = tf.keras.layers.BatchNormalization(momentum=bn_mom)(m)
    x = tf.keras.layers.Add()([x, m])

    # Inverted Residual Block 3 (24 -> 48 -> 32, stride 2)
    m = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False)(x)
    m = tf.keras.layers.BatchNormalization(momentum=bn_mom)(m)
    m = tf.keras.layers.ReLU(max_value=6.0)(m)
    m = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(m)
    m = tf.keras.layers.BatchNormalization(momentum=bn_mom)(m)
    m = tf.keras.layers.ReLU(max_value=6.0)(m)
    x = tf.keras.layers.Conv2D(32, (1, 1), padding="same", use_bias=False)(m)
    x = tf.keras.layers.BatchNormalization(momentum=bn_mom)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2) if pool_2d else (4, 1), padding="same")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="MobileNetV2_Micro")


# ============================================================================
# 3. Part A: Image Domain (MNIST + 10 Digits) Experiments
# ============================================================================

def run_mnist_experiments():
    print("\n" + "=" * 72)
    print("PART A: IMAGE DOMAIN (MNIST + 10 CUSTOM DIGITS)")
    print("=" * 72)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
    x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]

    my_raw_x, my_centered_x, my_y = load_my_10_digits(DATA_DIR / "my_digits")

    with open(DATA_DIR / "own_digits_28x28.csv", "w") as f:
        for label, img in zip(my_y, my_centered_x):
            pixels = np.round(img.flatten() * 255).astype(int)
            f.write(f"{label}," + ",".join(str(v) for v in pixels) + "\n")

    fig, axes = plt.subplots(2, 10, figsize=(7.2, 1.7))
    for i in range(10):
        for row, images in enumerate((my_raw_x, my_centered_x)):
            axes[row, i].imshow(images[i, :, :, 0], cmap="gray_r", vmin=0, vmax=1)
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
    axes[0, 0].set_ylabel("Resize", fontsize=9)
    axes[1, 0].set_ylabel("Centered", fontsize=9)
    plt.tight_layout(pad=0.2)
    plt.savefig(OUTPUT_DIR / "fig_my_digits_predictions.pdf")
    plt.savefig(OUTPUT_DIR / "fig_my_digits_predictions.png", dpi=150)
    plt.close()

    # Use the bounding-box centered 28x28 digits as our primary preprocessed 10-digit input
    # (and also track raw uncentered accuracy to explain preprocessing gains in the report!)
    step5_rows = []

    configs = [
        {
            "id": "init_mlp",
            "method": "Initial model (Slide 56 MLP: 80-60 ELU, Adam lr=0.001, ep=5, bs=32)",
            "builder": lambda: build_slide56_mlp(activation="elu", units1=80, units2=60),
            "opt": lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
            "epochs": 5,
            "batch_size": 32,
        },
        {
            "id": "mlp_sgd_lr005",
            "method": "Slide 56 MLP (ep=10, lr=0.05, SGD, bs=64)",
            "builder": lambda: build_slide56_mlp(activation="elu", units1=80, units2=60),
            "opt": lambda: tf.keras.optimizers.SGD(learning_rate=0.05),
            "epochs": 10,
            "batch_size": 64,
        },
        {
            "id": "cnn_base",
            "method": "Custom CNN Baseline (Conv 16/32, Dense 64, ReLU, Adam lr=0.001, ep=5, bs=64)",
            "builder": lambda: build_custom_mnist_cnn(f1=16, f2=32, dense_units=64, activation="relu", dropout=0.0),
            "opt": lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
            "epochs": 5,
            "batch_size": 64,
        },
        {
            "id": "cnn_dropout_ep10",
            "method": "Custom CNN + Dropout(0.25) + 10 epochs (Adam lr=0.001, bs=64)",
            "builder": lambda: build_custom_mnist_cnn(f1=16, f2=32, dense_units=64, activation="relu", dropout=0.25),
            "opt": lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
            "epochs": 10,
            "batch_size": 64,
        },
        {
            "id": "cnn_wide_acc",
            "method": "Accuracy-opt CNN (Conv 32/64, Dense 128, Dropout 0.25, Adam lr=0.001, ep=8)",
            "builder": lambda: build_custom_mnist_cnn(f1=32, f2=64, dense_units=128, activation="relu", dropout=0.25),
            "opt": lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
            "epochs": 8,
            "batch_size": 64,
        },
        {
            "id": "cnn_rmsprop",
            "method": "Custom CNN + RMSprop (lr=0.001, ep=5, bs=64)",
            "builder": lambda: build_custom_mnist_cnn(f1=16, f2=32, dense_units=64, activation="relu", dropout=0.1),
            "opt": lambda: tf.keras.optimizers.RMSprop(learning_rate=0.001),
            "epochs": 5,
            "batch_size": 64,
        },
        {
            "id": "cnn_compact",
            "method": "Memory/Speed-opt Micro-CNN (Conv 8/16, Dense 32, ReLU, Adam lr=0.001, ep=8)",
            "builder": lambda: build_custom_mnist_cnn(f1=8, f2=16, dense_units=32, activation="relu", dropout=0.1),
            "opt": lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
            "epochs": 8,
            "batch_size": 64,
        },
    ]

    trained_models = {}

    for cfg in configs:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        m = cfg["builder"]()
        m.compile(optimizer=cfg["opt"](), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t_train0 = time.perf_counter()
        m.fit(x_train, y_train, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
              validation_split=0.1, verbose=0)
        train_s = time.perf_counter() - t_train0

        _, val_acc = m.evaluate(x_test, y_test, verbose=0)
        own_probs = m.predict(my_centered_x, verbose=0)
        own_preds = np.argmax(own_probs, axis=1)
        own_acc = float(np.mean(own_preds == my_y))
        raw_acc = float(np.mean(np.argmax(m.predict(my_raw_x, verbose=0), axis=1) == my_y))

        speed_ms = measure_keras_own_data_speed_ms(m, my_centered_x, repeats=50)
        params = int(m.count_params())
        h5_path = MODELS_DIR / f"mnist_{cfg['id']}.h5"
        size_kb = save_keras_and_get_size_kb(m, h5_path)

        trained_models[cfg["id"]] = m
        row = {
            "id": cfg["id"],
            "method": cfg["method"],
            "speed_ms": round(speed_ms, 4),
            "params": params,
            "size_kb": round(size_kb, 2),
            "val_acc": round(float(val_acc), 4),
            "own_acc": round(own_acc, 4),
            "own_raw_acc": round(raw_acc, 4),
            "train_time_s": round(train_s, 2),
            "own_preds": [int(x) for x in own_preds],
        }
        step5_rows.append(row)
        print(f"[MNIST Step 5] {cfg['id']:16s} | val={val_acc:.4f} | own={own_acc:.2f} (raw {raw_acc:.2f}) | "
              f"params={params:6d} | size={size_kb:6.1f} KB | speed={speed_ms:.4f} ms/digit")

    # Add PTQ Int8 and QAT Int8 on Custom CNN and Micro-CNN
    if trained_models:
        base_cnn = trained_models["cnn_dropout_ep10"]
        ptq_path = MODELS_DIR / "mnist_cnn_ptq_int8.tflite"
        ptq_kb = convert_to_int8_tflite(base_cnn, x_train[:200], ptq_path, io_dtype=tf.int8)
        ptq_val, ptq_own, ptq_speed, ptq_preds = evaluate_tflite_model(
            ptq_path, x_test, y_test, my_centered_x, my_y
        )
        step5_rows.append({
            "id": "cnn_ptq_int8",
            "method": "Custom CNN + Post-Training Int8 Quantization (PTQ TFLite)",
            "speed_ms": round(ptq_speed, 4),
            "params": int(base_cnn.count_params()),
            "size_kb": round(ptq_kb, 2),
            "val_acc": round(ptq_val, 4),
            "own_acc": round(ptq_own, 4),
            "own_raw_acc": round(float(evaluate_tflite_model(ptq_path, x_test[:10], y_test[:10], my_raw_x, my_y)[1]), 4),
            "train_time_s": 0.0,
            "own_preds": ptq_preds,
        })
        print(f"[MNIST Step 5] {'cnn_ptq_int8':16s} | val={ptq_val:.4f} | own={ptq_own:.2f} | "
              f"params={base_cnn.count_params():6d} | size={ptq_kb:6.1f} KB | speed={ptq_speed:.4f} ms/digit")

        # QAT Int8 on Custom CNN (clone without Dropout for clean TFMOT wrapping)
        cnn_no_drop = trained_models["cnn_base"]
        qat_model = tfmot.quantization.keras.quantize_model(cnn_no_drop)
        qat_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t_qat0 = time.perf_counter()
        qat_model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=0)
        qat_train_s = time.perf_counter() - t_qat0
        qat_path = MODELS_DIR / "mnist_cnn_qat_int8.tflite"
        qat_kb = convert_to_int8_tflite(qat_model, x_train[:200], qat_path, io_dtype=tf.int8)
        qat_val, qat_own, qat_speed, qat_preds = evaluate_tflite_model(
            qat_path, x_test, y_test, my_centered_x, my_y
        )
        step5_rows.append({
            "id": "cnn_qat_int8",
            "method": "Custom CNN + Quantization-Aware Training (QAT Int8 TFLite)",
            "speed_ms": round(qat_speed, 4),
            "params": int(cnn_no_drop.count_params()),
            "size_kb": round(qat_kb, 2),
            "val_acc": round(qat_val, 4),
            "own_acc": round(qat_own, 4),
            "own_raw_acc": round(float(evaluate_tflite_model(qat_path, x_test[:10], y_test[:10], my_raw_x, my_y)[1]), 4),
            "train_time_s": round(qat_train_s, 2),
            "own_preds": qat_preds,
        })
        print(f"[MNIST Step 5] {'cnn_qat_int8':16s} | val={qat_val:.4f} | own={qat_own:.2f} | "
              f"params={cnn_no_drop.count_params():6d} | size={qat_kb:6.1f} KB | speed={qat_speed:.4f} ms/digit")

        # Compact Micro-CNN PTQ Int8 (extreme memory & speed optimization for Pico W)
        compact_cnn = trained_models["cnn_compact"]
        compact_ptq_path = MODELS_DIR / "mnist_compact_ptq_int8.tflite"
        c_kb = convert_to_int8_tflite(compact_cnn, x_train[:200], compact_ptq_path, io_dtype=tf.int8)
        c_val, c_own, c_speed, c_preds = evaluate_tflite_model(
            compact_ptq_path, x_test, y_test, my_centered_x, my_y
        )
        step5_rows.append({
            "id": "compact_ptq_int8",
            "method": "Micro-CNN (8/16, Dense 32) + Int8 PTQ TFLite (Best Memory & Speed)",
            "speed_ms": round(c_speed, 4),
            "params": int(compact_cnn.count_params()),
            "size_kb": round(c_kb, 2),
            "val_acc": round(c_val, 4),
            "own_acc": round(c_own, 4),
            "own_raw_acc": round(float(evaluate_tflite_model(compact_ptq_path, x_test[:10], y_test[:10], my_raw_x, my_y)[1]), 4),
            "train_time_s": 0.0,
            "own_preds": c_preds,
        })
        print(f"[MNIST Step 5] {'compact_ptq_int8':16s} | val={c_val:.4f} | own={c_own:.2f} | "
              f"params={compact_cnn.count_params():6d} | size={c_kb:6.1f} KB | speed={c_speed:.4f} ms/digit")

    # ------------------------------------------------------------------------
    # Step 6: Compare 3 Architectures on MNIST (Custom CNN, SqueezeNet, MobileNetV2)
    # ------------------------------------------------------------------------
    print("\n--- Step 6: 3-Architecture Comparison on MNIST ---")
    arch_rows = []
    arch_specs = [
        ("Custom 2D-CNN", lambda: build_custom_mnist_cnn(f1=16, f2=32, dense_units=64, dropout=0.25)),
        ("SqueezeNet-Micro (Fire modules)", lambda: build_squeezenet_micro((28, 28, 1), 10, pool_2d=True)),
        ("MobileNetV2-Micro (Inverted Residuals)", lambda: build_mobilenetv2_micro((28, 28, 1), 10, pool_2d=True)),
    ]

    for arch_name, arch_fn in arch_specs:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        am = arch_fn()
        am.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.perf_counter()
        am.fit(x_train, y_train, epochs=8, batch_size=64, validation_split=0.1, verbose=0)
        train_s = time.perf_counter() - t0

        _, val_acc = am.evaluate(x_test, y_test, verbose=0)
        own_preds = np.argmax(am.predict(my_centered_x, verbose=0), axis=1)
        own_acc = float(np.mean(own_preds == my_y))
        speed_ms = measure_keras_own_data_speed_ms(am, my_centered_x, repeats=50)
        params = int(am.count_params())

        slug = arch_name.split()[0].lower().replace("-", "_")
        h5_p = MODELS_DIR / f"arch_mnist_{slug}.h5"
        fp32_kb = save_keras_and_get_size_kb(am, h5_p)

        tfl_p = MODELS_DIR / f"arch_mnist_{slug}_int8.tflite"
        int8_kb = convert_to_int8_tflite(am, x_train[:200], tfl_p, io_dtype=tf.int8)
        int8_val, int8_own, int8_speed, _ = evaluate_tflite_model(
            tfl_p, x_test, y_test, my_centered_x, my_y
        )

        arch_rows.append({
            "architecture": arch_name,
            "params": params,
            "fp32_size_kb": round(fp32_kb, 2),
            "int8_size_kb": round(int8_kb, 2),
            "fp32_speed_ms": round(speed_ms, 4),
            "int8_speed_ms": round(int8_speed, 4),
            "fp32_val_acc": round(float(val_acc), 4),
            "int8_val_acc": round(int8_val, 4),
            "fp32_own_acc": round(own_acc, 4),
            "int8_own_acc": round(int8_own, 4),
            "train_time_s": round(train_s, 2),
        })
        print(f"[MNIST Step 6] {arch_name:38s} | params={params:6d} | FP32={fp32_kb:6.1f}KB ({val_acc:.4f}, own={own_acc:.2f}) | "
              f"INT8={int8_kb:5.1f}KB ({int8_val:.4f}, own={int8_own:.2f}, {int8_speed:.4f}ms)")

    # The model selected for the Pico
    export_tflite_to_cpp(MODELS_DIR / "arch_mnist_mobilenetv2_micro_int8.tflite",
                         BASE_DIR / "firmware" / "inference" / "models" / "mnist_model_data.cpp",
                         "mnist_model_data", "mnist_model_data.h")

    pd.DataFrame(step5_rows).to_csv(OUTPUT_DIR / "step5_mnist_optimizations.csv", index=False)
    pd.DataFrame(arch_rows).to_csv(OUTPUT_DIR / "step6_architectures_mnist.csv", index=False)
    return step5_rows, arch_rows


# ============================================================================
# 4. Generate Report Figures (IEEE two-column format)
# ============================================================================

def generate_report_figures(mnist_step5, mnist_arch):
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300)

    # (a) Model size vs. accuracy
    by_id = {r["id"]: r for r in mnist_step5}
    fp32_ids = ["init_mlp", "mlp_sgd_lr005", "cnn_base", "cnn_dropout_ep10", "cnn_wide_acc", "cnn_rmsprop", "cnn_compact"]
    int8_ids = ["cnn_ptq_int8", "cnn_qat_int8", "compact_ptq_int8"]
    ax1.scatter([by_id[i]["size_kb"] for i in fp32_ids], [by_id[i]["val_acc"] * 100 for i in fp32_ids],
                c="#1f77b4", marker="o", s=45, label="Float32 (.h5)")
    ax1.scatter([by_id[i]["size_kb"] for i in int8_ids], [by_id[i]["val_acc"] * 100 for i in int8_ids],
                c="#ff7f0e", marker="s", s=45, label="Int8 (.tflite)")
    selected = mnist_arch[2]
    ax1.scatter([selected["int8_size_kb"]], [selected["int8_val_acc"] * 100], c="#d62728", marker="*", s=200,
                edgecolors="black", linewidth=0.7, zorder=5, label="Selected (Int8)")
    ax1.set_xscale("log")
    ax1.set_xlim(20, 7500)
    ax1.set_ylim(96.4, 99.6)
    ax1.set_xlabel("Model size (KB, log scale)")
    ax1.set_ylabel("MNIST test accuracy (%)")
    ax1.set_title("(a) Memory size vs. accuracy", fontsize=10.5)
    ax1.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8.5)

    # (b) Three architectures
    names = ["Custom CNN", "SqueezeNet\nMicro", "MobileNetV2\nMicro"]
    fp32_sizes = [r["fp32_size_kb"] for r in mnist_arch]
    int8_sizes = [r["int8_size_kb"] for r in mnist_arch]
    x = np.arange(len(names))
    width = 0.36
    ax2.bar(x - width / 2, fp32_sizes, width, color="#1f77b4", label="Float32 (.h5)")
    ax2.bar(x + width / 2, int8_sizes, width, color="#ff7f0e", label="Int8 (.tflite)")
    for i in range(len(names)):
        ax2.text(i - width / 2, fp32_sizes[i] + 25, f"{fp32_sizes[i]:.0f}", ha="center", fontsize=8.5)
        ax2.text(i + width / 2, int8_sizes[i] + 25, f"{int8_sizes[i]:.0f}", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n}\n{r['int8_val_acc'] * 100:.2f}%" for n, r in zip(names, mnist_arch)], fontsize=8)
    ax2.set_ylim(0, 1500)
    ax2.set_ylabel("Model size (KB)")
    ax2.set_title("(b) Three architectures", fontsize=10.5)
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_ha2_optimizations.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_ha2_optimizations.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    t_start = time.perf_counter()
    mnist_step5, mnist_arch = run_mnist_experiments()
    generate_report_figures(mnist_step5, mnist_arch)

    summary = {
        "mnist_step5": mnist_step5,
        "mnist_step6_architectures": mnist_arch,
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
    }
    (OUTPUT_DIR / "ha2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] All HA2 experiments completed in {summary['total_runtime_s']:.1f}s.")
    print(f"Summary saved to: {OUTPUT_DIR / 'ha2_summary.json'}")


if __name__ == "__main__":
    main()
