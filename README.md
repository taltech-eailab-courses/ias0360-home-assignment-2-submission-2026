# Homework 2 – MNIST digit recognition

Max Pörner · 266055IV · IAS0360

I trained a small CNN on MNIST and added my own drawings from the Pico W touchscreen.
The goal was to reduce model memory while keeping useful accuracy. I also compared
a dense MLP and a LeNet-style CNN. The report is submitted separately through Moodle.

## Included files

- `firmware/capture/`: record labelled PGM drawings on the SD card.
- `firmware/demo/`: draw a digit and run the final INT8 model on the Pico W.
- `firmware/lib/`: required LCD, font and SD drivers from the course project.
- `ml/`: preprocessing, the three architectures, training, INT8 export and evaluation.
- `data/`: my 200 TRAIN, 50 VAL and 50 TEST images, already converted to 28×28.
  The NPZ files contain `x`, `y` and the original `filenames`. There are 20/5/5
  images per digit. `example_D7_001.PGM` is one original TRAIN drawing.
- `models/final_int8.tflite`: the selected model, 4,546 parameters and 10,608 bytes.
- `results/results.csv`: saved measurements used for the report.
- `results/pico_validation.json`: saved USB comparison on the real Pico.

The final model is also stored as `firmware/demo/model_data.h`, because the Pico
firmware embeds it in flash. The `.tflite` file is used by the Python tools.
Both contain the same model. The small host/USB tests and saved results are kept
so the capture logic and board comparison can be checked.

MNIST, the external TensorFlow Micro library, intermediate models and build output
are excluded.
The 300 full-size PGM files are represented by the compact NPZ files; they are not
needed to repeat training or evaluation.

## Run the final model

Run the commands below from this repository's root. The tested environment is the
course Docker image `ias0360-2026:latest` with Python 3.12 and CPU inference:

```sh
bash scripts/python_docker.sh ml/evaluate.py --output runs/final_check
bash scripts/python_docker.sh ml/predict_lite.py --model models/final_int8.tflite --image data/example_D7_001.PGM
```

Expected output: `correct: 50`, `count: 50`, `own_test_accuracy: 1.0`; the example
is recognised as digit 7. Evaluation writes `evaluation.json` and `predictions.csv`.
Add `--mnist` to evaluate the official 10,000 MNIST test images as well. MNIST is
downloaded on first use and cached locally. Expected MNIST test accuracy: 97.01%.

Alternatively, create a Python 3.12 virtual environment, install `requirements.txt`
and replace `bash scripts/python_docker.sh` with `python`. The pinned versions
were tested in the Linux course container; native macOS installation was not tested.

## Training and comparison

```sh
bash scripts/python_docker.sh ml/train.py --output runs/retraining
```

`ml/experiments.json` lists the report configurations. The small CNN, MLP and
LeNet-style network are the three different architectures. The initial comparison
uses their `mnist_float32.tflite` exports, before adaptation to my drawings.
The 7,834-parameter `cnn_smaller_dense` is called `reference_7834` in the results.
The later CNNs have 4,546, 2,034, 1,612, 1,190 and 768 parameters.

MNIST is split into 50,000 training and 10,000 validation images using a stratified
split with seed 10. Initial models train for five epochs; the later small heads
train for ten. Adam uses learning rate 0.001 and batch size 128. Adaptation uses
learning rate 0.0001, batch size 32 and ten epochs. Each adaptation epoch contains
my 200 TRAIN images repeated five times and 1,000 fixed MNIST training images.
The best adaptation epoch maximises mean MNIST/own validation accuracy, while
limiting the MNIST validation drop to 1 percentage point for the initial models
or 2 points for the later heads. Epoch zero can remain the best choice.

INT8 calibration uses only my 200 TRAIN images and 100 MNIST training images.
The final choice is the smallest INT8 CNN with at least 95% MNIST VAL and 94% own
VAL accuracy. My 50 TEST images are used for testing. They are excluded from
training, calibration and model selection. The original TEST set had already
been evaluated in earlier runs and was reused for the final comparison.

Each run writes the selected `.keras` models, Float32/INT8 `.tflite` exports and
validation measurements into its output directory. The packaged runner combines
the original training scripts; it does not include every discarded experiment.
A new run can differ slightly across TensorFlow versions or hardware. The saved
CSV contains the original measured results, not new measurements from packaging.

For a short pipeline check or a single experiment:

```sh
bash scripts/python_docker.sh ml/train.py --only head_8_16_8 --smoke --output runs/smoke
bash scripts/python_docker.sh ml/train.py --only head_4_8_6 --output runs/one_model
bash scripts/python_docker.sh ml/evaluate.py --model runs/one_model/head_4_8_6/adapted_int8.tflite --mnist --output runs/one_model_test
```

The smoke run uses one epoch per stage and a small MNIST subset; it is not a
report experiment. Use a new output directory for each training run.

`results/results.csv` includes accuracy, parameter count, TFLite file size,
constant tensor bytes and CPU latency. Accuracy uses all 50 own TEST images.
The timing measurement uses one fixed batch with one image per digit, ten warmups
and 500 timed calls, divided by ten. It includes tensor transfer and inference;
it excludes model loading, preprocessing and input quantisation. It runs with
four TFLite CPU threads in Docker on the Mac. These are not Pico timings.

## Build and run on the Pico W

The board uses the course LCD/touchscreen and SD-card wiring. The capture and demo
programs are separate firmware images. The demo needs no SD card.

The course Docker image provides the Pico SDK and ARM toolchain. Download the
pinned course TensorFlow Micro library once before building the demo:

```sh
bash scripts/setup_dependencies.sh
bash scripts/build_firmware.sh capture
bash scripts/build_firmware.sh demo
```

The dependency is the unchanged `lab_2_4/pico_sw/lcd/lib/pico-tflmicro` directory
from course commit `9f1560597cf8cb864c0b04025b5faeeb792aeadc`.
It stays under ignored `firmware/deps/`. For an existing local copy, use CMake's
`-DTFLMICRO_PATH=/path/to/pico-tflmicro` instead. Outside Docker, set
`PICO_SDK_PATH` to a Pico SDK 2.2.0 installation and use the same CMake commands
shown in `scripts/build_firmware.sh` with an ARM toolchain available.

Hold BOOTSEL while connecting the Pico, then copy either
`firmware/capture/build/h2_digit_capture.uf2` or
`firmware/demo/build/h2_digit_demo.uf2` onto the `RPI-RP2` drive.

Capture: the heading shows the requested digit and TRAIN/VAL/TEST phase. Draw
inside the frame; CLEAR is on the left and SAVE on the right. A drawing with at
least 20 pixels is also saved after two seconds without touch input. Saving runs
in the background. Wait until saving finishes before switching off. A completed
session contains `H2DIGITS/TRAIN`, `VAL` and `TEST`, with `D0_001.PGM` etc.
If recording again, use a clean SD card or move the previous H2DIGITS folder first.
Convert a copied recording with:

```sh
bash scripts/python_docker.sh ml/prepare_data.py --data /submission/H2DIGITS --output /submission/runs/new_data
```

The output uses the same NPZ format. Training uses the included `data/` by default;
to train with a new recording, replace those three NPZ files explicitly.

Demo: draw a digit, then lift the pen. After 0.8 seconds the heading shows
`Erkannt: <digit>`. READ recognises immediately; CLEAR starts a new drawing.
The shown time covers inference only. The local Dense adapter handles the model's
per-channel INT8 scales; the old course Dense kernel did not handle these correctly.

The original demo was built, flashed and checked on the real Pico: all 50 VAL
predictions and all ten output scores per image matched the CPU reference exactly
(49/50 correct, about 30 ms inference, 8,724 bytes of tensor arena used).
Physical drawing and reading the LCD were not tested by the automated checks.
To repeat the USB check on macOS with a flashed demo and one connected Pico:

```sh
bash scripts/python_docker.sh firmware/demo/tests/prepare_usb_check.py /submission /submission/runs/usb
python3 firmware/demo/tests/check_usb.py runs/usb
```

The saved hardware results are from the earlier board test. To run the small
capture plan/state checks on a computer with a C++17 compiler:

```sh
c++ -std=c++17 -I firmware/capture firmware/capture/tests/plan_check.cpp -o /tmp/h2-plan-check
/tmp/h2-plan-check
c++ -std=c++17 -I firmware/capture firmware/capture/tests/runtime_check.cpp -o /tmp/h2-runtime-check
/tmp/h2-runtime-check
```

Both checks should print `PASS`. Preparing this repository does not flash the
board again.

## Checks of this submission folder

Both Release firmware targets compiled from a separate copy of this folder.
The capture plan/state checks passed. The short training run completed, including
Float32 and INT8 export. Final-model evaluation gave 50/50 own TEST images and
97.01% MNIST test accuracy. The included raw example matched its saved NPZ input;
the C++ preprocessing matched it too. The embedded model bytes match the TFLite file.
Full training of every model was not repeated while preparing the submission.

## Source notes

LCD/config/font code is based on the Waveshare drivers supplied with the course.
The SD driver includes FatFs and its original copyright notices. Those notices
are retained. `firmware/lib/sd_card_driver/LICENSE.Apache-2.0` contains the license
for the SD-driver files marked Apache 2.0; the other files retain their own notices.
TensorFlow Micro/CMSIS-NN comes from the external course dependency.
My additions cover guided capture, preprocessing, model experiments, the inference
demo and the Dense adapter. Development and cleanup used AI assistance; the
handwritten drawings are my own.
