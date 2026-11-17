#!/usr/bin/env python3
"""Fine-tune and evaluate SqueezeNet 1.1 on MNIST."""

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .experiments import load_mnist, load_own_data
    from .pretrained_comparison import ROOT, build_model, preprocess
else:
    from experiments import load_mnist, load_own_data
    from pretrained_comparison import ROOT, build_model, preprocess

MODELS = {"squeezenet": "SqueezeNet 1.1"}
TRIALS = [
    ("reference", "Reference: Adam, 0.001", "adam", 0.001, 100),
    ("assignment", "Initial assignment settings", "sgd", 0.05, 10),
    ("adam", "Adam comparison", "adam", 0.001, 10),
    ("extended", "Extended training", "adam", 0.001, 50),
]
BATCH_SIZE = 128


def train(name, output, smoke, device, selected_optimizer, sgd_lr):
    x_train, y_train, _, _ = load_mnist()
    if smoke:
        x_train, y_train = x_train[:256], y_train[:256]
    training_inputs = torch.from_numpy(x_train.transpose(0, 3, 1, 2).copy())
    labels = torch.from_numpy(y_train.astype(np.int64))
    del x_train, y_train
    optimizers = ("adam", "sgd") if selected_optimizer == "all" else (selected_optimizer,)
    for optimizer_name in optimizers:
        torch.manual_seed(0)
        model = build_model(MODELS[name])
        model.to(device)
        model.train()
        loader = DataLoader(TensorDataset(training_inputs, labels), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=0.001)
            if optimizer_name == "adam"
            else torch.optim.SGD(model.parameters(), lr=sgd_lr)
        )
        epochs = 1 if smoke else (100 if optimizer_name == "adam" else 10)
        for epoch in range(1, epochs + 1):
            for batch, target in loader:
                optimizer.zero_grad()
                logits = model(preprocess(batch).to(device))
                loss = F.cross_entropy(logits, target.to(device))
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite loss: {name} {optimizer_name} epoch {epoch}")
                loss.backward()
                optimizer.step()
            if not torch.stack([
                torch.isfinite(value).all()
                for value in model.state_dict().values() if value.is_floating_point()
            ]).all().item():
                raise RuntimeError(
                    f"Non-finite model state: {name} {optimizer_name} epoch {epoch}; "
                    f"SGD learning rate {sgd_lr}"
                )
            print(f"{name} {optimizer_name}: epoch {epoch}/{epochs}", flush=True)
            for trial, _, opt, _, trial_epochs in TRIALS:
                if opt == optimizer_name and epoch == (1 if smoke else trial_epochs):
                    torch.save(model.state_dict(), output / f"{name}_{trial}.pth")
    del training_inputs
    gc.collect()
    (output / f"{name}_training.json").write_text(json.dumps({
        "model": MODELS[name], "pretrained_weights": "ImageNet-1K V1",
        "training_mode": "full fine-tuning",
        "device": device, "training_examples": len(labels), "batch_size": BATCH_SIZE,
        "image_size": 64, "seed": 0, "smoke": smoke, "sgd_learning_rate": sgd_lr,
    }, indent=2) + "\n", encoding="utf-8")


class NHWCModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images.permute(0, 3, 1, 2))


def model_inputs(images):
    batches = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = torch.from_numpy(images[start:start + BATCH_SIZE].transpose(0, 3, 1, 2).copy())
        batches.append(preprocess(batch).permute(0, 2, 3, 1).numpy())
    return np.concatenate(batches)


def predict(interpreter, images):
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise ValueError("Expected int8 input and output")
    input_scale, input_zero = inp["quantization"]
    output_scale, output_zero = out["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("Invalid input or output quantization scale")
    predictions, confidence = [], []
    start = time.perf_counter()
    for image in images:
        quantized = np.clip(np.round(image[None] / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized)
        interpreter.invoke()
        logits = (interpreter.get_tensor(out["index"])[0].astype(np.float32) - output_zero) * output_scale
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        index = int(np.argmax(logits))
        predictions.append(index)
        confidence.append(float(probabilities[index]))
    return np.array(predictions), np.array(confidence), (time.perf_counter() - start) * 1000 / len(images)


def measure(name, output, smoke):
    manifest = json.loads((output / f"{name}_training.json").read_text(encoding="utf-8"))
    if (manifest["training_mode"] != "full fine-tuning" or manifest["smoke"] != smoke
            or manifest["training_examples"] != (256 if smoke else 54000)):
        raise ValueError(f"Training metadata does not match the requested run: {manifest}")
    x_train, _, x_val, y_val = load_mnist()
    own = load_own_data(str(ROOT / "own_data"))
    if own is None or len(own[1]) != 10 or sorted(own[1].tolist()) != list(range(10)):
        raise ValueError("Expected one labelled custom image for each digit 0–9")
    if smoke:
        x_val, y_val = x_val[:40], y_val[:40]
    calibration = model_inputs(x_train[:200])
    val = model_inputs(x_val)
    own_inputs = model_inputs(own[0])
    calibration_path = output / f"{name}_calibration.npy"
    np.save(calibration_path, calibration)
    rows, own_rows = [], []
    for trial, description, opt, lr, epochs in (TRIALS[:1] if smoke else TRIALS):
        if opt == "sgd":
            lr = manifest["sgd_learning_rate"]
            if lr != 0.05:
                description = "SGD, adjusted learning rate"
        torch.manual_seed(0)
        model = build_model(MODELS[name])
        model.load_state_dict(torch.load(output / f"{name}_{trial}.pth", map_location="cpu", weights_only=True))
        if not all(torch.isfinite(value).all().item()
                   for value in model.state_dict().values() if value.is_floating_point()):
            raise ValueError(f"Non-finite checkpoint: {name} {trial}")
        model.eval()
        wrapped = NHWCModel(model).eval()
        onnx_path = output / f"{name}_{trial}.onnx"
        torch.onnx.export(
            wrapped, torch.zeros(1, 64, 64, 3), str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=17, dynamo=False,
        )
        converted = output / f"{name}_{trial}_converted"
        converter = Path(sys.executable).parent / "onnx2tf"
        command = [
            str(converter), "-i", str(onnx_path), "-o", str(converted),
            "--tflite_backend", "tf_converter", "-kt", "input",
            "-oiqt", "-iqd", "int8", "-oqd", "int8",
            "-cind", "input", str(calibration_path), "[[[[0,0,0]]]]", "[[[[1,1,1]]]]",
        ]
        log_path = output / f"{name}_{trial}_conversion.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"Conversion failed; see {log_path}")
        model_path = converted / f"{onnx_path.stem}_full_integer_quant.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        val_pred, _, _ = predict(interpreter, val)
        own_pred, own_conf, own_ms = predict(interpreter, own_inputs)
        check_inputs = val[:200]
        with torch.inference_mode():
            float_pred = np.concatenate([
                wrapped(torch.from_numpy(check_inputs[start:start + BATCH_SIZE])).argmax(dim=1).numpy()
                for start in range(0, len(check_inputs), BATCH_SIZE)
            ])
        agreement = np.mean(float_pred == val_pred[:len(check_inputs)])
        rows.append({
            "model": MODELS[name], "optimization": description, "optimizer": opt,
            "training_mode": "full fine-tuning",
            "learning_rate": lr, "epochs": 1 if smoke else epochs, "batch_size": BATCH_SIZE,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "tflite_model_bytes": model_path.stat().st_size,
            "own_data_avg_inference_ms": f"{own_ms:.3f}",
            "validation_accuracy": f"{np.mean(val_pred == y_val) * 100:.2f}%",
            "own_data_accuracy": f"{np.mean(own_pred == own[1]) * 100:.1f}%",
            "float_int8_agreement_first_200": f"{agreement * 100:.2f}%",
        })
        if trial == "reference":
            for filename, actual, predicted, confidence in zip(own[2], own[1], own_pred, own_conf):
                own_rows.append({
                    "model": MODELS[name], "image": filename, "actual": int(actual),
                    "predicted": int(predicted), "correct": bool(predicted == actual),
                    "confidence": f"{confidence * 100:.2f}%",
                })
        print(rows[-1], flush=True)
    for filename, data in ((f"{name}_trials.csv", rows), (f"{name}_predictions.csv", own_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "measure"))
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="use 256 training images and one epoch")
    parser.add_argument("--fine-tune", action="store_true", help="train the entire pretrained network (default)")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="training device")
    parser.add_argument("--optimizer", choices=("all", "adam", "sgd"), default="all")
    parser.add_argument("--sgd-lr", type=float, default=0.005)
    args = parser.parse_args()
    if not np.isfinite(args.sgd_lr) or args.sgd_lr <= 0:
        parser.error("--sgd-lr must be positive and finite")
    default_output = ROOT / "results" / "squeezenet_runs"
    if args.smoke and (args.output_dir is None or args.output_dir.resolve() == default_output):
        parser.error("--smoke requires a separate --output-dir")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    args.output_dir = args.output_dir or default_output
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.stage == "train":
        train(args.model, args.output_dir, args.smoke, args.device,
              args.optimizer, args.sgd_lr)
    else:
        measure(args.model, args.output_dir, args.smoke)


if __name__ == "__main__":
    main()
