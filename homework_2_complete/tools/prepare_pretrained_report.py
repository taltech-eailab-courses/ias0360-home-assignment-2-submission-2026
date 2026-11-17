#!/usr/bin/env python3
"""Combine completed pretrained trials with the original CNN measurements."""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CNN_NAMES = ("Initial CNN 16/32/d64", "Optimized CNN 8/16/d32")
MODELS = {"squeezenet": "SqueezeNet 1.1"}
TRIALS = {
    "Reference: Adam, 0.001": ("adam", 0.001, 100),
    "Initial assignment settings": ("sgd", 0.05, 10),
    "Adam comparison": ("adam", 0.001, 10),
    "Extended training": ("adam", 0.001, 50),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def accuracy(row, field):
    value = float(row[field].removesuffix("%"))
    if not 0 <= value <= 100:
        raise ValueError(f"Invalid {field}: {row[field]}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-dir", type=Path, default=RESULTS / "pretrained_finetuned_trials")
    parser.add_argument("--cnn-results-dir", type=Path, default=RESULTS)
    parser.add_argument("--sgd-lr", type=float, default=0.005)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["squeezenet"])
    args = parser.parse_args()
    pretrained = args.trials_dir.resolve()
    output = pretrained / "report_data"
    original, result_fields = read_csv(args.cnn_results_dir / "results.csv")
    trials, trial_fields = read_csv(args.cnn_results_dir / "optimization_trials.csv")
    predictions, prediction_fields = read_csv(args.cnn_results_dir / "own_data_predictions.csv")
    reference = [row for row in original if row["model"] in CNN_NAMES]
    trials = [row for row in trials if row["model"] in CNN_NAMES]
    predictions = [row for row in predictions if row["model"] in CNN_NAMES]
    if len(reference) != 2 or len(trials) != 8 or len(predictions) != 20:
        raise ValueError("Expected the two original CNNs, their eight trials and 20 predictions")
    for name in CNN_NAMES:
        model_reference = [row for row in reference if row["model"] == name]
        model_trials = [row for row in trials if row["model"] == name]
        model_predictions = [row for row in predictions if row["model"] == name]
        if (len(model_reference) != 1 or len(model_trials) != len(TRIALS)
                or {row["optimization"] for row in model_trials} != set(TRIALS)
                or len(model_predictions) != 10
                or sorted(int(row["actual"]) for row in model_predictions) != list(range(10))):
            raise ValueError(f"Incomplete CNN results for {name}")
        for row in model_trials:
            opt, lr, epochs = TRIALS[row["optimization"]]
            if (row["optimizer"] != opt or float(row["learning_rate"]) != lr
                    or int(row["epochs"]) != epochs or int(row["batch_size"]) != 128):
                raise ValueError(f"Unexpected CNN trial configuration: {row}")
        row = model_reference[0]
        if (row["optimizer"] != "adam" or float(row["learning_rate"]) != 0.001
                or int(row["epochs"]) != 100 or int(row["batch_size"]) != 128):
            raise ValueError(f"Unexpected CNN reference configuration: {row}")

    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    for key in args.models:
        name = MODELS[key]
        rows, _ = read_csv(pretrained / f"{key}_trials.csv")
        trial_settings = dict(TRIALS)
        if args.sgd_lr != 0.05:
            trial_settings.pop("Initial assignment settings")
            trial_settings["SGD, adjusted learning rate"] = ("sgd", args.sgd_lr, 10)
        if len(rows) != len(trial_settings) or {r["optimization"] for r in rows} != set(trial_settings):
            raise ValueError(f"Incomplete trials for {name}")
        for row in rows:
            opt, lr, epochs = trial_settings[row["optimization"]]
            if (row["model"] != name or row["optimizer"] != opt
                    or float(row["learning_rate"]) != lr or int(row["epochs"]) != epochs
                    or int(row["batch_size"]) != 128):
                raise ValueError(f"Unexpected trial configuration: {row}")
            accuracy(row, "validation_accuracy")
            accuracy(row, "own_data_accuracy")
            if int(row["parameters"]) <= 0 or int(row["tflite_model_bytes"]) <= 0:
                raise ValueError(f"Invalid model size: {row}")
        own, _ = read_csv(pretrained / f"{key}_predictions.csv")
        if (len(own) != 10 or {row["model"] for row in own} != {name}
                or sorted(int(row["actual"]) for row in own) != list(range(10))):
            raise ValueError(f"Expected ten labelled predictions for {name}")
        if {row["image"] for row in own} != {
            row["image"] for row in predictions if row["model"] == CNN_NAMES[0]
        }:
            raise ValueError(f"Different evaluation images for {name}")
        reference.append(next(row for row in rows if row["optimization"] == "Reference: Adam, 0.001"))
        trials.extend(rows)
        predictions.extend(own)

    best = max(trials, key=lambda row: (
        accuracy(row, "validation_accuracy"), accuracy(row, "own_data_accuracy"),
        -int(row["tflite_model_bytes"]),
    ))
    smallest = min(trials, key=lambda row: (
        int(row["tflite_model_bytes"]), -accuracy(row, "validation_accuracy"),
        -accuracy(row, "own_data_accuracy"),
    ))
    summary = [
        {"selection": "Best validation accuracy", **best},
        {"selection": "Smallest TFLite model", **smallest},
    ]
    summary_fields = [
        "selection", "model", "optimization", "optimizer", "learning_rate", "epochs",
        "batch_size", "own_data_avg_inference_ms", "parameters", "tflite_model_bytes",
        "validation_accuracy", "own_data_accuracy",
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results.csv", reference, result_fields)
    write_csv(output / "optimization_trials.csv", trials, trial_fields)
    write_csv(output / "optimization_summary.csv", summary, summary_fields)
    write_csv(output / "own_data_predictions.csv", predictions, prediction_fields)
    print(f"Combined report data: {output}")


if __name__ == "__main__":
    main()
