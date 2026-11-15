# IAS0360 Home Assignment 2: digit classification on a Raspberry Pi Pico W
Joosep Parts, 256780IAXD. Image domain. Networks are trained on MNIST in TensorFlow, tested on 10
digits drawn on the Pico touchscreen, and the selected int8 model (MobileNetV2-Micro) runs on the Pico
itself. The optimisation goal is memory size.

Everything runs from the lab repo in the course container, with this repo in `submission/`. The
firmware uses the `lib/` folders of `lab_1_1/lcd_sd_card_example` and `lab_2_4/pico_sw/lcd` from there.

## 1. Build the firmware
```bash
cd firmware/inference && mkdir -p build && cd build && cmake ..
make -j6 main main_arena_size_test main_inference_test main_uart_inference
```

## 2. Demo and collecting the digits (Step 2)
`./start.sh` puts the demo (`main.uf2`) on the Pico and prints what it reads. On a Mac it first starts
the course VM, because the board is attached there.

Draw a digit in the box and stop. After a short countdown the Pico reads it, shows the 28x28 image
the network sees with its answer, the confidence for every digit and the inference time, saves the
drawing to the SD card as
`digit_NNN.txt` (280x240, ASCII 0 and 1), and clears the screen for the next digit. The same bitmap
goes over USB into `output/demo_log.txt`. The own data set is the last clean drawing of each digit, kept in
`data/touchscreen/`. `python3 convert_touchscreen_digits.py --sd-dir data/touchscreen` converts them
to `data/my_digits/digit_N.png`.

## 3. Train, optimise, compare (Steps 1, 3 to 6)
```bash
python3 run_ha2_experiments.py
```
Takes about 5 minutes. It retrains everything and writes the selected model to
`firmware/inference/models/mnist_model_data.cpp`, so build the firmware again afterwards.

## 4. Measure on the Pico
```bash
picotool load firmware/inference/build/main_arena_size_test.uf2 -f -x   # prints the tensor arena size
picotool load firmware/inference/build/main_inference_test.uf2 -f -x    # 10 built-in MNIST digits
picotool load firmware/inference/build/main_uart_inference.uf2 -f -x    # one image per line over USB
python3 firmware/pico_eval.py                                           # the 10 own digits, accuracy and time
```
`pico_eval.py` needs `pyserial`. Use `--port` when the board is not `/dev/pico`.

## Expected output
- `output/step5_mnist_optimizations.csv`: the Step 5 table.
- `output/step6_architectures_mnist.csv`: the three architectures of Step 6.
- `output/fig_ha2_optimizations.pdf`, `output/fig_my_digits_predictions.pdf`: the report figures.
- `output/ha2_summary.json`: all of the above in one file.
- `output/pico_results.json`: prediction and inference time of each digit, measured on the Pico.
- `data/own_digits_28x28.csv`: the 10 converted digits, one line each: label, then 784 pixels.
- `models/`: the int8 models (`.tflite`). The `.h5` float models are written here too and are not in git.

Report: `report/ha2_report.pdf`.

The converter runs with per-neuron Dense quantization switched off, because the TFLite Micro in the
lab's `pico-tflmicro` applies one scale per Dense layer. Without that the Pico still predicts, but with
wrong scores.

AI tool disclosure: Claude Code was used to write the experiment script, fill in the firmware
templates, build the demo screen and draft this README and the report.
