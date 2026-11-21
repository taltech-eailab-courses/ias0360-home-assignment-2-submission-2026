# Home Assignment 2 report: plain-language guide

This file explains the claims and numbers in [`report.tex`](report.tex). The LaTeX source follows the same IEEE-style paper structure as the Home Assignment 1 report: title and abstract, introduction, implementation, experiments, results, discussion, conclusion, and references. `report_preview.pdf` is the earlier PDF preview; the submission PDF should be compiled from `report.tex` so it matches the editable source.

## What the project does

The program learns to classify a 28 × 28 grayscale image as one of the digits 0–9. It first trains models on MNIST, then runs the trained models on the ten drawings collected from the touchscreen. It does **not** train on those ten personal drawings; they are kept for a separate evaluation.

The MNIST training set has 60,000 examples. A fixed random split made 54,000 training examples and 6,000 validation examples. The official 10,000-example MNIST test set was reserved until model choices were made.

## Models in the report

- **Custom CNN:** the student-designed model. It uses two convolution and max-pooling stages, a 32-neuron dense layer, and a 10-class output. Its baseline has 14,410 trainable parameters.
- **LeNet-style CNN:** a second convolutional design using tanh activations and average pooling. It is adapted to 28 × 28 images.
- **MLP:** a simpler comparison model. It flattens the image into a vector and uses dense layers; it does not use convolution.

The wider CNN experiment is a larger version of the custom CNN, with more filters and a wider dense layer.

## What the experiments changed

All runs use the same MNIST split, seed, image preparation, and batch size (64). The baseline custom CNN trains for 5 epochs using Adam with learning rate 0.001. The other custom CNN runs change one main setting: 10 epochs, lower learning rate, dropout, a wider network, or SGD with momentum. LeNet and the MLP provide architecture comparisons.

An **epoch** means one pass through all 54,000 training images. So the baseline makes five passes; the `more_epochs` run makes ten. Training saves the checkpoint with the highest validation accuracy for each run.

## How to read the result table

- **Validation accuracy:** fraction of the 6,000 validation images classified correctly. For example, 98.47% means around 98 or 99 correct predictions per 100 validation images.
- **Own-data accuracy:** fraction of the ten touchscreen drawings classified correctly. Each digit has exactly one example, so a single error changes the score by 10 percentage points. LeNet's 90% means 9/10; the custom CNN and MLP each scored 8/10.
- **Average ms/image:** total warmed prediction time for ten touchscreen samples divided by ten. File reading and image preprocessing are excluded. This timing came from the Docker CPU environment, not the Pico hardware.
- **Parameters:** number of learned numerical values in a model.
- **H5 KiB:** size of the saved model file. It already contains the weights. Do not add the model file size and weight size together. The CSV also records the weights-only byte count.

The best validation score belongs to `more_epochs`: 98.47%, compared with 98.10% for the baseline. That is a gain of 0.37 **percentage points**. This is a small difference, not a 0.37% relative improvement claim. On the reserved MNIST test set, the validation-selected custom CNN scored 98.56%, LeNet 98.21%, and the MLP 97.44%.

## What happened on the touchscreen drawings

The SD-card PGM files were copied into the project and normalized to the model's expected input. The original captures were kept. The report's digit strip is the processed input, so it shows what the neural network actually received rather than the full touchscreen interface.

LeNet got the drawn **6** wrong and predicted **5**, giving it 9/10. The custom CNN's mistakes were 6→5 and 8→3. The MLP's mistakes were 0→2 and 8→2. This per-digit table is a small confusion summary. A large confusion-matrix analysis would not be meaningful with only one handwritten example for each class.

## Limits and what not to claim

1. Ten personal drawings are too few to make a strong claim about handwriting recognition. Collecting several drawings per digit, ideally from more than one person, would make that evaluation more useful.
2. MNIST test accuracy and touchscreen accuracy measure different data distributions. High MNIST accuracy does not guarantee high accuracy on this touchscreen style.
3. The model timing is not Raspberry Pi Pico timing. The Pico was used for capture and SD-card storage; neural-network inference was run in the Docker/TensorFlow environment.
4. Results come from one fixed split and seed. The custom CNN also had more hyperparameter experiments than LeNet and the MLP, so avoid saying the experiment proves it is the best architecture in general.
5. No quantization or other embedded inference optimization was measured in this assignment study.

## Files to look at

- `report.tex` — editable LaTeX report source.
- `report_preview.pdf` — earlier preview PDF; compile the LaTeX source for the final matching PDF.
- `figures/processed_touchscreen_digits.png` — local image used by the LaTeX source.
- `generate_report.py` — regenerates the earlier PDF preview from saved study results.
- `../results/study/RESULTS.md` — experiment comparison in Markdown.
- `../results/study/results.csv` — exact per-run settings and measured values.
- `../results/study/*/own_predictions.json` — predictions on each of the ten drawings.
- `../results/study/own_digits/preview_0_to_9.png` — processed samples shown in the report.
- `../data/own_digits/` — labelled copies used for evaluation; original SD exports are kept under `../data/raw_sd_card/`.

Before submitting, check that your name/UNI-ID and any cover-page details match the course's requirements. The report is based on the recorded experiments; do not add results or hardware claims that were not measured.

To compile the LaTeX file, open Terminal in this `report` directory and run `pdflatex report.tex` twice. The current environment does not have a LaTeX compiler installed, so the `.tex` source has not been compiled here.
