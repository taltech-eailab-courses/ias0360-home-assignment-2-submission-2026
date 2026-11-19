"""Run an exported float32/INT8 TFLite model on one original PGM drawing."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from prepare_data import preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=4)
    interpreter.allocate_tensors()
    detail = interpreter.get_input_details()[0]
    image = preprocess(args.image)[None, ...]
    if detail["dtype"] == np.int8:
        scale, zero = detail["quantization"]
        image = np.clip(np.rint(image / scale + zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(detail["index"], image)
    interpreter.invoke()
    output = interpreter.get_output_details()[0]
    scores = interpreter.get_tensor(output["index"])[0]
    print(f"Model: {args.model.name}; file size: {args.model.stat().st_size} bytes")
    print(f"Image: {args.image.name}; predicted digit: {int(scores.argmax())}")
