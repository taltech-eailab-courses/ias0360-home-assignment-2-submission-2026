# Home Assignment 2: MNIST classification with Docker

This is the project root for the image-based assignment. Code, collected digit
copies, downloaded MNIST cache, trained models and reports stay in this folder.
Dependencies run inside Docker; no local Python environment is required to
use the project. Docker's image storage remains managed by Docker itself.

Working optimization goal: **validation accuracy**. Three architectures are
implemented: a configurable custom small CNN, a LeNet-style CNN, and a fully
connected MLP. LeNet is adapted for 28x28 inputs rather than an exact historical
replica. These alternatives are appropriate for MNIST; large image classifiers
are not required by the assignment's examples.

## Current status

- The Docker configuration, models, preprocessing, experiment runner and own-data
  evaluator are implemented.
- A separate Pico touchscreen capture firmware is available in
  [`pico_digit_capture/`](pico_digit_capture/README.md); it is set up to save
  ten labeled PGM images to SD. It has compiled, but the physical screen/SD
  interaction still needs to be checked on the actual board.
- A read-only USB exporter is also available for the Waveshare Pico-Eval-Board,
  so the saved PGM files can be transferred without a separate SD-card reader.
- Ten touchscreen captures were exported and copied into `data/raw_sd_card/`
  and `data/own_digits/`. Four blank first-attempt captures were preserved in
  the raw archive; the refreshed evaluation set contains visible digits 0–9.
- Own-digit evaluation completed: LeNet classified 9/10 correctly; the custom
  CNN and MLP each classified 8/10. LeNet confused the drawn 6 with 5. With one
  example per class, these accuracy values move in 10-point increments and are
  only a small personal-data check, not a reliable generalization estimate.
- The processed preview is at
  [`results/study/own_digits/preview_0_to_9.png`](results/study/own_digits/preview_0_to_9.png).
- The native ARM64 Docker image built successfully. All three preprocessing
  tests and synthetic training checks for all three architectures passed inside
  Docker. Docker Desktop needed access to its own log/socket directories outside
  the tool sandbox; an initial application launch error did not indicate that
  TensorFlow required installation on the host.
- All eight accuracy-study runs finished inside Docker. Best validation accuracy:
  **98.47%** (custom CNN, 10 epochs). Reserved-test accuracies: custom CNN **98.56%**,
  LeNet-style CNN **98.21%**, and MLP **97.44%**. These are measured results for
  one split/seed, not guaranteed performance on new handwritten samples.
- Open [the completed comparison table](results/study/RESULTS.md).
  It now includes the measured own-digit accuracy and prediction timing for
  every trained model.
- The editable paper-style report is [`report/report.tex`](report/report.tex).
  Its plain-language companion is [`report/EXPLAINER.md`](report/EXPLAINER.md),
  and the earlier PDF preview is [`report/report_preview.pdf`](report/report_preview.pdf).

## Build and check

From Terminal:

```bash
cd "/Users/tanishqvashishth/Desktop/ML_Course/ias0360-lab-excercises-2026/submission/ias0360-home-assignment-2-submission-2026"

docker info
docker compose build
docker compose run --rm homework python -m unittest test_digits -v
docker compose run --rm homework python run_experiments.py --smoke
```

`docker info` must connect to the engine. The three unit tests check input
polarity, grayscale shape, blank-image rejection, source preservation, and
missing-data handling using temporary synthetic pictures. Those pictures are
not the assignment's collected data. The smoke command checks a finite training
loss and ten-output predictions for every architecture using synthetic tensors.
It downloads no MNIST and reports no model accuracy.

The image uses Python 3.11, TensorFlow 2.16.2 and legacy `tf_keras` 2.16.0 to
match the course notebook's API. Dependencies are installed in a native-CPU
Python container, allowing ARM64 on Apple Silicon and AMD64 on other machines.
The container is configured for CPU experiments; these commands do not use the
Mac's Apple GPU. The project is bind-mounted at `/workspace`, so outputs persist
on your Mac after each disposable container exits.

## Run the assignment experiments

First inspect the settings without training:

```bash
docker compose run --rm homework python run_experiments.py --plan
```

A short run compares three architectures at two epochs each:

```bash
docker compose run --rm homework python run_experiments.py --preset quick
```

The full accuracy study performs eight controlled runs:

```bash
docker compose run --rm homework python run_experiments.py --preset accuracy
```

| Run | Change from custom CNN baseline |
|---|---|
| baseline | Conv 8/16, dense 32, Adam 0.001, batch 64, 5 epochs |
| more_epochs | 10 epochs |
| lower_learning_rate | Adam 0.0003 |
| dropout | Dropout 0.25 |
| wider_cnn | Conv 16/32, dense 64 |
| sgd_momentum | SGD 0.01 with momentum 0.9 |
| lenet | LeNet-style architecture, 5 epochs |
| mlp | Fully connected architecture, 5 epochs |

The full preset totals 45 epochs over 54,000 training examples. It can take
substantial CPU time. The quick preset is a pipeline/architecture check, not a
completed optimization study. All presets use ten digit outputs and a 28x28x1
input. Runs get new timestamped output folders; existing runs are not overwritten.

## Data and evaluation methodology

MNIST downloads on the first real training run into `.cache/keras/`. Its 60,000
training examples are split using a fixed seed (42): 54,000 training and 6,000
validation. The official 10,000 test examples are reserved. Training images are
shuffled every epoch. All architectures receive identical splits and input
normalization; no ImageNet pretrained weights are used.

Each run saves its checkpoint with the highest validation accuracy. Once all
runs finish, the validation-selected candidate within each architecture is
evaluated on the reserved test set. Do not tune further based on these test
results or choose settings based on personal-digit test accuracy.

The generated table measures:

- **Validation accuracy:** on all 6,000 fixed validation examples.
- **Memory:** parameter count from the model plus actual weight bytes and
  serialized H5 model bytes, shown separately. The file already contains weights;
  summing the two sizes would double-count them. This is not peak runtime RAM.
- **Own-data inference time:** total warmed prediction time for ten real digits,
  divided by ten. Calls are serial, batch size one, CPU; disk reads and input
  preprocessing are excluded.
- **Own-data accuracy:** correct predictions out of the ten labelled samples.
  One sample per digit gives 10 percentage-point resolution, not a reliable
  large-sample generalization estimate.

Before personal digits are available, a separate MNIST-validation-reference
timing column is recorded. It is never substituted into the assignment's
own-data timing column. Missing own-data values remain empty/pending.

## Collect and evaluate your own digits

Draw digits 0 through 9 using the required touchscreen and save the captures
to the SD card. Preserve those raw files and collection notes under
`data/raw_sd_card/`. Put labelled copies in `data/own_digits/` following its
[instructions](data/own_digits/README.md). Hardware capture firmware is not
included in this Python training project; do not claim this step complete
until the physical collection and SD-card saving have been performed.

If the digits are already present at training time, every experiment evaluates
them automatically. If collected afterward, reuse the saved models:

```bash
docker compose run --rm homework python evaluate_own.py outputs/run_YYYYMMDD_HHMMSS
```

Replace the example run directory with a real one. This performs inference,
updates the comparison table and saves a preview of all ten preprocessed inputs.
It does not retrain or overwrite the source captures. Inspect that preview:
digits should be bright strokes on a black background and correctly labelled.

To complete the already trained, submission-ready study after collecting digits:

```bash
docker compose run --rm homework python evaluate_own.py results/study
```

`results/study/` contains a curated copy of the eight trained models, settings,
histories, summaries and measurements. It can be committed for submission;
temporary runs and caches remain ignored. Update this curated copy's own-data
table before submitting. Keep the real labelled digit samples in `data/own_digits/`
with the submission so the reported own-data results can be reproduced.

## Files and outputs

| File | Purpose |
|---|---|
| `Dockerfile`, `compose.yaml`, `requirements.txt` | Reproducible CPU runtime and mounted project |
| `models.py` | Custom CNN, LeNet-style CNN and MLP definitions |
| `digits.py` | Read-only digit preprocessing and ten-image loading |
| `run_experiments.py` | Split MNIST, train, checkpoint, measure and compare |
| `evaluate_own.py` | Apply saved models to the real ten-digit dataset |
| `test_digits.py` | Three synthetic input-processing regression tests |

Each real run saves `RESULTS.md`, `results.csv`, `results.json`, split indices,
own-data status, and a final summary. Each experiment has settings, history,
`model_summary.txt`, its best weights, and `model.h5`. The overall validation
winner is also exported as `best_saved_model/` for the legacy lab workflow.
Architecture test accuracies are in `final_summary.json`.

## Sources

- [TensorFlow Docker instructions](https://www.tensorflow.org/install/docker)
- [TensorFlow 2.16.2 platform wheels](https://pypi.org/project/tensorflow/2.16.2/)
- [Legacy tf_keras 2.16.0](https://pypi.org/project/tf-keras/2.16.0/)
- Course starter: `lab_2_2/mnist_cnn.ipynb` in the original course repository.

The measured accuracies above come from the saved Docker accuracy study.
Use that study's actual measurements to write the optimization conclusions;
no real-own-digit accuracy is claimed until the hardware samples are evaluated.
