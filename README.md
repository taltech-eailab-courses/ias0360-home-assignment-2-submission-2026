# Home Assignment 2: MNIST digit classification

The experiment compares three models:

1. Initial CNN: 16/32 filters and 64 dense units.
2. Optimized CNN: 8/16 filters and 32 dense units.
3. SqueezeNet 1.1: ImageNet-pretrained torchvision model, with a new
   10-class classifier and all layers fine-tuned on MNIST.

Each model uses 54,000 MNIST training images and 6,000 validation images.
The ten touchscreen drawings are used only for evaluation, not training or
quantization calibration. Their small number limits conclusions about new
handwriting.

The CNNs accept 28x28 grayscale images. SqueezeNet inputs are resized to
64x64, repeated into three channels, and normalized with ImageNet statistics.
All reported models are full-int8 TFLite models, calibrated using 200 MNIST
training images. SqueezeNet inference timings exclude input preprocessing.

## Data

Convert the SD card's touchscreen captures into 28x28 images:

```sh
python3 tools/collect_touch_digits.py sd_data
python3 tools/collect_touch_digits.py sd_data_9 --labels 9
```

The experiment requires one labelled image per digit in `own_data/`. The
conversion crops the drawing, adds a margin, resizes and centers it, and
produces a white digit on a black background.

## Training

Run the two CNN comparisons and their training configuration trials:

```sh
python3 tools/experiments.py
```

The reference uses Adam, learning rate 0.001, 100 epochs and batch size 128.
Additional trials use SGD (0.05, 10 epochs), Adam (0.001, 10 epochs), and
Adam (0.001, 50 epochs). Results are written to `results/cnn_runs/`, replacing
files with matching names there, without overwriting the historical results
in `results/`. Pass `--output-dir` to choose another directory. To rerun only
the additional trials, first generate the reference in that directory, then run:

```sh
python3 tools/experiments.py --training-sweep-only
```

For SqueezeNet, install `requirements-pretrained.txt` and
`requirements-tflite-conversion.txt` in an environment with TensorFlow. The
first run may download ImageNet weights. Full training is lengthy:

```sh
python3 tools/pretrained_trials.py train squeezenet
python3 tools/pretrained_trials.py measure squeezenet
```

The three Adam trials are checkpoints at epochs 10, 50 and 100 of the same
run. The SGD trial starts separately and uses 0.005 because 0.05 produced
non-finite weights. The measurement stage exports each checkpoint via ONNX
to int8 TFLite and evaluates the validation images and drawings. New files
are written to `results/squeezenet_runs/`, replacing matching files there.
Use `--output-dir` to choose another directory. A short test requires both
`--smoke` and a separate `--output-dir`.

The saved report uses the original CNN measurements and the completed
SqueezeNet fine-tuning results in `results/pretrained_finetuned_trials/`.
To regenerate the combined data:

```sh
python3 tools/prepare_pretrained_report.py
```

To combine new runs instead, use `--cnn-results-dir results/cnn_runs
--trials-dir results/squeezenet_runs`. This writes `report_data/` inside the
chosen SqueezeNet directory. Both sets must contain the complete reference and
trials with the report settings.

## Walkthrough notebook

`modelos_homework_2.ipynb` explains the data, the three models, int8 conversion,
and recorded results in English. CNN retraining is optional; the SqueezeNet
section loads the saved weights rather than retraining.

## Raspberry Pi Pico

To retrain the optimized CNN and export a TFLite model and C++ array:

```sh
python3 tools/export_pico_model.py
```

This overwrites `models/optimized_cnn.tflite` and
`models/optimized_cnn_model_data.cpp`. Build separately from the repository
root, then flash the firmware:

```sh
cmake -S "Home assignment 1" -B "Home assignment 1/build-homework-2" \
  -DMNIST_MODEL_SOURCE="$PWD/Home assignment 2/models/optimized_cnn_model_data.cpp"
cmake --build "Home assignment 1/build-homework-2" -j
./flash.sh "Home assignment 1/build-homework-2/digit_recognition.uf2"
```

The physical-board test requires the Pico and LCD to be connected.

## Report

```sh
home_assignment_2_report.pdf
```
