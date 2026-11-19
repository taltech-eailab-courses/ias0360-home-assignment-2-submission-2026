"""Evaluate a fixed TFLite model; no training or model selection."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

def prepare_input(interpreter, values):
    detail = interpreter.get_input_details()[0]
    if list(detail["shape"]) != list(values.shape):
        interpreter.resize_tensor_input(detail["index"], values.shape, strict=False)
        interpreter.allocate_tensors()
    detail = interpreter.get_input_details()[0]
    if detail["dtype"] == np.int8:
        scale, zero = detail["quantization"]
        return np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)
    return values.astype("float32")


def predict(interpreter, x):
    predictions = []
    for start in range(0, len(x), 256):
        values = prepare_input(interpreter, x[start:start + 256])
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], values)
        interpreter.invoke()
        predictions.extend(interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).argmax(axis=1).tolist())
    return np.array(predictions)


def timing(interpreter, ten):
    values = prepare_input(interpreter, ten)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    def call():
        interpreter.set_tensor(input_index, values)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)
    for _ in range(10):
        call()
    elapsed = []
    for _ in range(500):
        start = time.perf_counter_ns()
        call()
        elapsed.append((time.perf_counter_ns() - start) / 1000 / 10)
    return {"mean_us_per_digit": float(np.mean(elapsed)),
            "median_us_per_digit": float(np.median(elapsed)),
            "std_us_per_digit": float(np.std(elapsed)),
            "repeats": len(elapsed), "warmups": 10}


def run(args):
    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=4)
    interpreter.allocate_tensors()
    with np.load(args.root / 'data/own_test.npz') as data:
        x, y, names = data['x'], data['y'], data['filenames']
    predictions = predict(interpreter, x)
    subset = np.array([str(name).endswith('_001.PGM') for name in names])
    assert subset.sum() == 10 and len(np.unique(y[subset])) == 10
    row = dict(model=str(args.model), sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
               tflite_file_bytes=args.model.stat().st_size, correct=int((predictions == y).sum()),
               count=len(y), own_test_accuracy=float(np.mean(predictions == y)),
               **timing(interpreter, x[subset]))
    if args.mnist:
        _, (mx, my) = tf.keras.datasets.mnist.load_data()
        mx = mx.astype('float32')[..., None] / 255
        row['mnist_test_accuracy'] = float(np.mean(predict(interpreter, mx) == my))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
        with (args.output / 'predictions.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filename', 'true_digit', 'predicted_digit'])
            writer.writerows(zip(names, y, predictions))
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=root)
    parser.add_argument('--model', type=Path, default=root / 'models/final_int8.tflite')
    parser.add_argument('--mnist', action='store_true', help='Also evaluate the official MNIST test set')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())
