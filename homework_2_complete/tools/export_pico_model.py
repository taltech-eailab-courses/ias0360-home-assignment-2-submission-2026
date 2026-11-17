#!/usr/bin/env python3
"""Train the compact Homework 2 CNN and export a Pico-compatible int8 model."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

if __package__:
    from .experiments import custom_cnn, load_mnist, load_own_data, tflite_bytes, tflite_predict
else:
    from experiments import custom_cnn, load_mnist, load_own_data, tflite_bytes, tflite_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "optimized_cnn.tflite"
CPP_PATH = PROJECT_ROOT / "models" / "optimized_cnn_model_data.cpp"


def write_cpp_array(model_bytes: bytes, cpp_path: Path) -> None:
    values_per_line = 12
    rows = []
    for start in range(0, len(model_bytes), values_per_line):
        row = ", ".join(
            f"0x{value:02x}" for value in model_bytes[start:start + values_per_line]
        )
        rows.append(f"  {row},")

    cpp_path.write_text(
        '#include "mnist_model_data.h"\n\n'
        "alignas(8) const unsigned char mnist_model_data[] = {\n"
        + "\n".join(rows)
        + "\n};\n"
        + f"const int mnist_model_data_len = {len(model_bytes)};\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--cpp-output", type=Path, default=CPP_PATH)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    tf.keras.utils.set_random_seed(0)
    x_train, y_train, x_val, y_val = load_mnist()
    model = custom_cnn((8, 16), 32)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        verbose=2,
    )

    model_bytes = tflite_bytes(model, x_train, quantize=True, fixed_batch=True)
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if input_details["dtype"] != np.int8 or output_details["dtype"] != np.int8:
        raise RuntimeError("The exported model must use int8 input and output tensors")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.cpp_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_bytes(model_bytes)
    write_cpp_array(model_bytes, args.cpp_output)

    val_predictions, _, _ = tflite_predict(model_bytes, x_val)
    print(f"Parameters: {model.count_params():,}")
    print(f"Validation accuracy: {np.mean(val_predictions == y_val) * 100:.2f}%")
    print(f"TFLite size: {len(model_bytes):,} bytes")
    print(f"Input tensor: {input_details['shape']} {input_details['dtype']}")
    print(f"Output tensor: {output_details['shape']} {output_details['dtype']}")
    print(f"Wrote {args.model_output}")
    print(f"Wrote {args.cpp_output}")

    own = load_own_data(str(PROJECT_ROOT / "own_data"))
    if own is not None:
        own_predictions, _, _ = tflite_predict(model_bytes, own[0])
        print(f"Custom-data accuracy: {np.mean(own_predictions == own[1]) * 100:.1f}%")


if __name__ == "__main__":
    main()
