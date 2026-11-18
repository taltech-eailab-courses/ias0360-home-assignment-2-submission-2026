"""Train and evaluate the initial and optimized MNIST CNNs."""
import argparse
import csv
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mnist():
    (x, y), _ = keras.datasets.mnist.load_data()
    x = (x.astype("float32") / 255.0)[..., None]
    return x[:-6000], y[:-6000], x[-6000:], y[-6000:]


def load_own_data(folder):
    from PIL import Image, ImageOps
    xs, ys, names = [], [], []
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) or not f[0].isdigit():
            continue
        img = Image.open(os.path.join(folder, f)).convert("L").resize((28, 28))
        a = np.asarray(img, dtype="float32") / 255.0
        if a.mean() > 0.5:  # dark digit on light background ------> invert
            a = 1.0 - a
        xs.append(a[..., None])
        ys.append(int(f[0]))
        names.append(f)
    if not xs:
        return None
    return np.stack(xs), np.array(ys), names


def act(name):
    return {"relu": "relu", "tanh": "tanh", "sigmoid": "sigmoid"}[name]


def custom_cnn(filters=(16, 32), dense=64, dropout=0.0, activation="relu"):
    m = keras.Sequential([layers.Input((28, 28, 1))])
    for f in filters:
        m.add(layers.Conv2D(f, 3, activation=act(activation)))
        m.add(layers.MaxPooling2D())
    m.add(layers.Flatten())
    if dropout:
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(dense, activation=act(activation)))
    m.add(layers.Dense(10, activation="softmax"))
    return m


def tflite_bytes(model, calib, quantize, fixed_batch=False):
    if fixed_batch:
        concrete_function = tf.function(model).get_concrete_function(
            tf.TensorSpec((1, 28, 28, 1), tf.float32)
        )
        conv = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_function], model
        )
    else:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([c[None]] for c in calib[:200])
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_predict(blob, x):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds, confidences = [], []
    t0 = time.perf_counter()
    for s in x:
        s = s[None]
        if inp["dtype"] == np.int8:
            sc, zp = inp["quantization"]
            s = np.clip(np.round(s / sc + zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], s)
        it.invoke()
        scores = it.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8:
            scale, zero_point = out["quantization"]
            scores = (scores.astype("float32") - zero_point) * scale
        predicted = int(np.argmax(scores))
        preds.append(predicted)
        confidences.append(float(scores[predicted]))
    return (
        np.array(preds),
        np.array(confidences),
        (time.perf_counter() - t0) / len(x) * 1000,
    )


def run(name, optimization, model, opt, lr, epochs, batch, data, own,
        quantize, results, own_predictions_by_model):
    xt, yt, xval, yval = data
    opts = {"adam": keras.optimizers.Adam, "sgd": keras.optimizers.SGD}
    model.compile(optimizer=opts[opt](lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(xt, yt, epochs=epochs, batch_size=batch,
              validation_data=(xval, yval), verbose=0)
    blob = tflite_bytes(model, xt, quantize)
    val_predictions, _, _ = tflite_predict(blob, xval)
    own_predictions, own_confidences, own_ms = tflite_predict(blob, own[0])
    if own_predictions_by_model is not None:
        for image_name, actual, predicted, confidence in zip(
            own[2], own[1], own_predictions, own_confidences
        ):
            own_predictions_by_model.append({
                "model": name,
                "image": image_name,
                "actual": int(actual),
                "predicted": int(predicted),
                "correct": predicted == actual,
                "confidence": f"{confidence * 100:.2f}%",
            })
    result = {
        "model": name, "optimization": optimization, "optimizer": opt,
        "learning_rate": lr, "epochs": epochs, "batch_size": batch,
        "parameters": model.count_params(), "tflite_model_bytes": len(blob),
        "own_data_avg_inference_ms": f"{own_ms:.3f}",
        "validation_accuracy": f"{np.mean(val_predictions == yval) * 100:.2f}%",
        "own_data_accuracy": f"{np.mean(own_predictions == own[1]) * 100:.1f}%"}
    results.append(result)
    print(result, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--own", default=os.path.join(ROOT, "own_data"))
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results", "cnn_runs"))
    ap.add_argument(
        "--training-sweep-only",
        action="store_true",
        help="run training-parameter trials using the existing architecture comparison as reference",
    )
    a = ap.parse_args()
    if a.epochs < 1:
        ap.error("--epochs must be at least 1")
    keras.utils.set_random_seed(0)
    data = load_mnist()
    own = load_own_data(a.own)
    if own is None:
        raise SystemExit(f"No labelled images found in {a.own}")
    if len(own[1]) != 10 or sorted(own[1].tolist()) != list(range(10)):
        raise SystemExit(f"Expected exactly one labelled image for each digit 0-9 in {a.own}")
    res = []
    own_predictions_by_model = []
    os.makedirs(a.output_dir, exist_ok=True)
    results_path = os.path.join(a.output_dir, "results.csv")
    plan = [
        ("Initial CNN 16/32/d64", lambda: custom_cnn()),
        ("Optimized CNN 8/16/d32", lambda: custom_cnn((8, 16), 32)),
    ]
    if not a.training_sweep_only:
        for n, builder in plan:
            keras.utils.set_random_seed(0)
            run(n, "Architecture comparison + int8", builder(), "adam",
                1e-3, a.epochs, 128, data, own, True, res,
                own_predictions_by_model)
        with open(results_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)
        predictions_out = os.path.join(a.output_dir, "own_data_predictions.csv")
        with open(predictions_out, "w", newline="") as f:
            fields = ["model", "image", "actual", "predicted", "correct", "confidence"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(own_predictions_by_model)
        print("saved", predictions_out)
    if not os.path.isfile(results_path):
        raise SystemExit(f"Reference results not found: {results_path}. Run the comparison first.")
    with open(results_path, newline="") as result_file:
        reference_rows = list(csv.DictReader(result_file))
    reference_by_name = {row["model"]: row for row in reference_rows}
    missing_references = [name for name, _ in plan if name not in reference_by_name]
    if missing_references:
        raise SystemExit(
            "Missing reference models in results.csv: "
            + ", ".join(missing_references)
            + ". Run the architecture comparison first."
        )

    configurations = [
        ("Initial assignment settings", "sgd", 0.05, 10),
        ("Adam comparison", "adam", 0.001, 10),
        ("Extended training", "adam", 0.001, 50),
    ]
    trial_results = []
    for name, builder in plan:
        reference = dict(reference_by_name[name])
        reference["optimization"] = "Reference: Adam, 0.001"
        trial_results.append(reference)
        for description, optimizer, learning_rate, epochs in configurations:
            keras.utils.set_random_seed(0)
            result = run(
                name,
                description,
                builder(),
                optimizer,
                learning_rate,
                epochs,
                128,
                data,
                own,
                True,
                [],
                None,
            )
            trial_results.append(result)

    trials_out = os.path.join(a.output_dir, "optimization_trials.csv")
    with open(trials_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_results[0]))
        w.writeheader()
        w.writerows(trial_results)
    print("saved", trials_out)

    def accuracy(row, field):
        return float(row[field].rstrip("%"))

    best_accuracy = max(
        trial_results,
        key=lambda row: (
            accuracy(row, "validation_accuracy"),
            accuracy(row, "own_data_accuracy"),
            -int(row["tflite_model_bytes"]),
        ),
    )
    smallest_model = min(
        trial_results,
        key=lambda row: (
            int(row["tflite_model_bytes"]),
            -accuracy(row, "validation_accuracy"),
            -accuracy(row, "own_data_accuracy"),
        ),
    )
    summary_rows = []
    for label, row in (
        ("Best validation accuracy", best_accuracy),
        ("Smallest TFLite model", smallest_model),
    ):
        summary_rows.append({
            "selection": label,
            **{
                key: row[key]
                for key in (
                    "model",
                    "optimization",
                    "optimizer",
                    "learning_rate",
                    "epochs",
                    "batch_size",
                    "own_data_avg_inference_ms",
                    "parameters",
                    "tflite_model_bytes",
                    "validation_accuracy",
                    "own_data_accuracy",
                )
            },
        })
    summary_out = os.path.join(a.output_dir, "optimization_summary.csv")
    summary_fields = [
        "selection",
        "model",
        "optimization",
        "optimizer",
        "learning_rate",
        "epochs",
        "batch_size",
        "own_data_avg_inference_ms",
        "parameters",
        "tflite_model_bytes",
        "validation_accuracy",
        "own_data_accuracy",
    ]
    with open(summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print("saved", summary_out)


if __name__ == "__main__":
    main()
