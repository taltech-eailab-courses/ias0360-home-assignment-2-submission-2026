# IAS0360 Home Assignment 2: IMU gesture recognition

Student code: **253436IACM**. Branch: `submissions/253436IACM`.

- **Application domain:** IMU gesture recognition, 4 classes: `negative` (stationary), `ring` (O), `slope` (L), `wing` (W).
- **Optimization goal (Step 3):** accuracy.
- **Hardware:** Raspberry Pi Pico 2 W (RP2350) on the course eval board: ICM-20948 IMU, SD card, WS2812 LED.
- **Output:** USB serial console; the LED shows the recording state. The LCD is not used.

## Repository layout

```
dataset/
  provided/<class>/*.txt        lab 2.2 dataset (25 recordings per class, raw ax,ay,az counts)
  own/<class>/*.txt             own recordings from the SD card (Step 2)
  own_converted/<class>/*.txt   own recordings converted to the dataset orientation
training/
  gesture_data.py               loading, preprocessing (same as the firmware), axis map, augmentation
  models.py                     lecture CNN, MobileNet-style, SqueezeNet-style
  run_experiments.py            Steps 1, 4, 5, 6: trains every configuration and writes the table
  convert_own_data.py           Step 2: finds the axis map and converts own data
  export_firmware.py            writes the chosen models and preprocessing constants into the firmware
  results/                      optimization_table.md/.csv, experiments.jsonl, models/*.tflite
firmware/
  common/                       ICM-20948 driver, SD card driver (FatFs), SD pin config, LED helper
  data_collection/              Step 2: records gestures to the SD card
  gesture_inference/            own_data_test (Step 5 on the device) and live_inference
```

## Requirements

The course Docker image (`ias0360-2026`, built by `./build_in_docker.sh` in the lab repository). It provides TensorFlow, tfmot, the ARM toolchain, the Pico SDK and picotool, and clones this repository into `~/submission/ias0360-home-assignment-2-submission-2026/`.

`firmware/gesture_inference` uses TensorFlow Lite Micro from the lab repository (`lab_2_4/pico_sw/imu/lib/pico-tflmicro`). If this repository is not inside the lab repository, pass `-DPICO_TFLMICRO_PATH=<lab repo>/lab_2_4/pico_sw/imu/lib/pico-tflmicro` to CMake.

All commands run inside the container from this repository's root:

```bash
cd ~/submission/ias0360-home-assignment-2-submission-2026
```

## 1. Build and flash the firmware

```bash
# Data collection (Step 2)
cmake -S firmware/data_collection -B firmware/data_collection/build
make -C firmware/data_collection/build -j$(nproc)

# Own-data test (Step 5 on the device) and live demo
cmake -S firmware/gesture_inference -B firmware/gesture_inference/build
make -C firmware/gesture_inference/build -j$(nproc)
```

| File | Purpose |
|---|---|
| `firmware/data_collection/build/data_collection.uf2` | Record gestures to the SD card |
| `firmware/gesture_inference/build/own_data_test.uf2` | Evaluate every model on the recordings on the SD card |
| `firmware/gesture_inference/build/live_inference.uf2` | Record one gesture and classify it with every model, in a loop |

Flash with the lab script (hold BOOTSEL while plugging in the first time):

```bash
~/flash.sh ~/submission/ias0360-home-assignment-2-submission-2026/firmware/gesture_inference/build/own_data_test.uf2
```

Open the serial console on `/dev/ttyACM0`, e.g. `picocom /dev/ttyACM0` (exit with Ctrl-A Ctrl-X). Each program waits up to 10 s for the console before printing.

## 2. Collect your own data (Step 2)

1. Flash `data_collection.uf2` and open the serial console.
2. Hold the board like the provided dataset: screen facing down, IMU x right, y backward, z up (see `dataset/Eval-board-imu-orientation.png`). Draw the shapes in a vertical plane, as if writing on a wall.
3. Choose the class (`0` to `3`) and the number of recordings (Enter = 25).
4. For each recording:
   - LED blue, console `get ready...`.
   - LED green, console `GO!`: perform the gesture within 2.1 s.
   - White flash: recording saved.
   - Press `q` between recordings to stop.
5. Menu option `x` deletes all recordings of a class (confirm with `y`).
6. Copy the SD card folders `negative/ ring/ slope/ wing/` into `dataset/own/`.

The SD card must be FAT32 or exFAT. Use a terminal without local echo (such as picocom).

Expected console output:

```
=== HA2 IMU gesture data collection ===
  0  negative
  1  ring
  2  slope
  3  wing
Select gesture class [0-3]: ring
Number of recordings [Enter = 25]:
Recording 25 x 'ring' (press q between recordings to stop)
[ring 1/25] get ready... GO!
    saved 0:/ring/ring_000.txt: 4200 samples (2000 Hz)
...
```

Each file has the provided dataset's format: an `ax,ay,az` header followed by raw int16 accelerometer counts, sampled at 2 kHz for 2.1 s (4200 samples).

**Converting to the training orientation:**

```bash
python training/convert_own_data.py                    # detect the orientation automatically
python training/convert_own_data.py --axis-map=-y,x,z  # or set it by hand
```

The script tries all 24 axis-aligned rotations and picks the one under which the per-class mean gestures of the own recordings best match the provided dataset. It saves `training/axis_map.json` and writes converted copies to `dataset/own_converted/`. The same axis map is compiled into the firmware, so the Pico converts recordings the same way. The axis map for this submission is `y,-x,z`.

**Collected data:** `dataset/own/` holds 25 recordings per class, 100 in total.
- `dataset/own_excluded/`: recordings dropped because the board was not moved.
- `dataset/own_horizontal/`: a session drawn in a horizontal plane, not used for training or evaluation.

## 3. Train, optimize and compare (Steps 1, 3–6)

```bash
python training/run_experiments.py --force   # about 15 min on CPU, 22 configurations
python training/export_firmware.py           # best model of each architecture -> firmware
```

Then rebuild `gesture_inference` and flash `own_data_test.uf2`.

The models train on the provided recordings plus the own recordings. Every configuration runs a stratified 5-fold cross-validation over both sets: fold k trains on 4/5 of each and is scored on the remaining fifth.

- **Validation accuracy:** held-out provided recordings, mean ± std over the folds.
- **Own data accuracy:** held-out own recordings, using the uint8 TFLite model that runs on the Pico. Every own recording is classified once, by a model that did not train on it.
- **Average inference time:** uint8 prediction time per held-out own recording, measured on the PC.
- **Memory:** parameter count, float32 weight size and deployed `.tflite` size.
- **"Provided dataset only"** rows train without own recordings.
- **Deployed models** train on all provided recordings plus own folds 1–4. The 20 own recordings in `training/results/own_test_split.json` are held out.

### Results (Step 5 table)

Full table: `training/results/optimization_table.md`.

| Optimization method | Inference avg.* (PC) | Memory** (params / float32 / uint8 TFLite) | Validation acc. (5-fold CV) | Own data acc. (100 gestures) |
|---|---|---|---|---|
| Initial model, **trained on the provided dataset only** | 0.016 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 4.9 | 65 % |
| Step 6 MobileNet-style, **trained on the provided dataset only** | 0.014 ms | 10,308 / 40.3 KB / 28.0 KB | 99.0 % ± 2.0 | 60 % |
| **Initial model** (lab 2.2 / lecture CNN: 8/16 filters, dense 16, ReLU, dropout 0.1; Adam lr 0.001, 30 epochs, batch 16) | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 96 % |
| Number of epochs = 10; learning rate = 0.05 | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 92.0 % ± 5.1 | 85 % |
| Number of epochs = 10 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 3.7 | 90 % |
| Number of epochs = 60 | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 96 % |
| Number of epochs = 100 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 94 % |
| Learning rate = 0.01 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 95.0 % ± 4.5 | 94 % |
| Learning rate = 0.0003, 60 epochs | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 95 % |
| Optimizer SGD (momentum 0.9), lr = 0.01, 60 epochs | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 95 % |
| Optimizer RMSprop, lr = 0.001 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 94 % |
| Batch size = 8 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 94 % |
| Batch size = 32, 60 epochs | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 92 % |
| Activation = tanh | 0.015 ms | 4,556 / 17.8 KB / 10.2 KB | 97.0 % ± 2.4 | 94 % |
| No Dropout | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 93.0 % ± 7.5 | 95 % |
| Dropout = 0.3 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 95 % |
| More neurons: 16/32 filters, dense 32 | 0.024 ms | 17,812 / 69.6 KB / 23.9 KB | 93.0 % ± 7.5 | 94 % |
| One more Conv2D + MaxPool layer | 0.015 ms | 6,892 / 26.9 KB / 13.9 KB | 96.0 % ± 3.7 | 95 % |
| **Data augmentation ×10** (time warp/shift, rotation ±15°, scale, noise) | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 97.0 % ± 2.4 | **99 %** |
| Combined: 16/32 filters, dense 32, dropout 0.3, augmentation ×10, 80 epochs | 0.024 ms | 17,812 / 69.6 KB / 23.9 KB | 98.0 % ± 2.4 | 98 % |
| **Step 6: MobileNet-style** depthwise-separable CNN (aug ×10, 80 epochs) | 0.014 ms | 10,308 / 40.3 KB / 28.0 KB | 98.0 % ± 2.4 | **100 %** |
| **Step 6: SqueezeNet-style** Fire-module CNN (aug ×10, 80 epochs) | 0.017 ms | 8,180 / 32.0 KB / 27.7 KB | **99.0 % ± 2.0** | **100 %** |

\* Average uint8 TFLite prediction time per held-out own recording, on the training PC.
\*\* Parameters from `model.summary()`, float32 weight size, deployed uint8 `.tflite` size.

### On-device results (Step 5 on the embedded system)

`own_data_test` on the Pico 2 W (RP2350 at 150 MHz), all 100 own recordings from the SD card:

| Model | Held-out own accuracy | All own recordings | Avg. inference / gesture | Tensor arena used | Model size |
|---|---|---|---|---|---|
| SqueezeNet-style (`squeezenet_cnn`) | 100 % (20/20) | 100 % (100/100) | 8.62 ms | 8,820 B | 28,352 B |
| MobileNet-style (`mobilenet_ds_cnn`) | 100 % (20/20) | 100 % (100/100) | 8.64 ms | 7,364 B | 28,664 B |
| Lecture CNN + augmentation ×10 (`aug_x10`) | 100 % (20/20) | 100 % (100/100) | 5.97 ms | 5,188 B | 10,152 B |

"Held-out" = the 20 recordings the deployed models did not train on.

### Observations

- **Own data is essential.** Models trained only on the provided recordings reach 94–99 % validation but only 60–65 % on own gestures. Adding own recordings raises the initial model to 96 %.
- **Step 4 (lecture CNN).** Most learning-rate, optimizer, batch, activation and dropout settings give 92–97 % own-data accuracy. Too few epochs (10) is worse: 90 %, and 85 % with lr 0.05.
- **Augmentation helps the most:** 96 % → 99 % on own data and 94 % → 97 % validation, at no cost in size or speed.
- **Step 6 (architectures).** MobileNet-style and SqueezeNet-style reach 100 % on own data and 98–99 % validation, using about 28 KB of flash (vs 10 KB) and about 2.7 ms more inference time.
- **Chosen model:** SqueezeNet-style: highest validation accuracy (99 %), 100 % on own data. It is model 1 in the firmware and drives the LED in `live_inference`.

## Design decisions

1. **Training uses the same preprocessing as the device.** `gesture_data.to_frames` matches the firmware's `DataPreprocessor` (cut/pad to 4000 samples, block-average to 128 frames, normalize).
2. **Fixed 2 kHz sampling** in data collection and live inference, matching the provided dataset.
3. **One firmware image runs all three architectures**, so they are compared on the same gestures.
4. **TFLite Micro runs on core 1** with a 16 KB stack.
5. **I2C bus recovery on start-up** (`firmware/common/icm20948/icm20948.c`) so the IMU initializes after a reset without a power cycle.

## Expected output (`own_data_test`)

```
==== HA2 own-data test: 3 model(s), window 4000 samples -> 128 frames ====
SD card mounted as 0:
Recordings on the SD card:
  negative  25 recordings
  ring      25 recordings
  slope     25 recordings
  wing      25 recordings
=== Model 1/3: squeezenet_cnn ===
Step 6: SqueezeNet-style Fire-module CNN (augmentation x10, 80 epochs)
model 28352 bytes, tensor arena used 8820 of 49152 bytes, input scale 0.051269 zero point 154
  ...
  ring      ring_050.txt       TEST  -> ring      ok    p=1.00   8.623 ms  (4200 samples)
RESULT,squeezenet_cnn,ring/ring_050.txt,test,ring,ring,8.623
  ...
squeezenet_cnn: held-out (TEST) accuracy 20/20 = 100.0 %, all own recordings 100/100 = 100.0 %
total inference 862.248 ms, average 8.622 ms/gesture
confusion matrix (rows = true, columns = predicted):
            negative     ring    slope     wing
  negative        25        0        0        0
  ring             0       25        0        0
  slope            0        0       25        0
  wing             0        0        0       25
SUMMARY,squeezenet_cnn,20,20,100.00,100,100,100.00,8.622,8820,28352
...
=== On-device summary (RP2350 @ 150 MHz, 100 own gestures) ===
| model | held-out accuracy | all own recordings | avg inference (ms) | tensor arena (B) | model size (B) |
|---|---|---|---|---|---|
| squeezenet_cnn | 100.0 % (20/20) | 100.0 % (100/100) | 8.622 | 8820 | 28352 |
| mobilenet_ds_cnn | 100.0 % (20/20) | 100.0 % (100/100) | 8.638 | 7364 | 28664 |
| aug_x10 | 100.0 % (20/20) | 100.0 % (100/100) | 5.967 | 5188 | 10152 |
```

Each recording is marked `TEST` (held out) or `train`. `RESULT,` and `SUMMARY,` lines are CSV. `SUMMARY` columns: model, held-out total, held-out correct, held-out %, all total, all correct, all %, average ms, arena bytes, model bytes.

## Expected output (`live_inference`)

```
==== HA2 live gesture recognition: 3 model(s) ====
model 1: squeezenet_cnn          28352 bytes, arena used 8820 bytes
model 2: mobilenet_ds_cnn        28664 bytes, arena used 7364 bytes
model 3: aug_x10                 10152 bytes, arena used 5188 bytes
--- gesture #1: get ready (hold the board like in the training data) ---
GO - perform the gesture now (2.1 s)
captured 4200 samples (2000 Hz), using the first 4000
[squeezenet_cnn] -> negative (9.182 ms)
    negative  0.996
    ring      0.000
    slope     0.000
    wing      0.000
[mobilenet_ds_cnn] -> negative (9.051 ms)
...
[aug_x10] -> negative (7.248 ms)
...
Gesture: negative
```

LED: blue = get ready, green = perform the gesture, yellow = classifying, then for 2 s the class predicted by model 1: white = negative, cyan = ring, magenta = slope, orange = wing.
