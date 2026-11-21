"""Create the Home Assignment 2 report from the saved experiment results."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ha2_report_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "study"
BLUE = "#17365D"
MID_BLUE = "#2F75B5"
PALE = "#EAF1F8"
GRAY = "#536273"
INK = "#202A35"


def load_data():
    rows = list(csv.DictReader((STUDY / "results.csv").open()))
    for row in rows:
        for key in ("epochs", "batch", "parameters", "weight_bytes", "model_file_bytes"):
            row[key] = int(row[key])
        for key in ("learning_rate", "dropout", "validation_accuracy", "own_average_ms", "own_accuracy"):
            row[key] = float(row[key])
    summary = json.loads((STUDY / "final_summary.json").read_text())
    predictions = {
        path.parent.name: json.loads(path.read_text())["predictions"]
        for path in STUDY.glob("*/own_predictions.json")
    }
    return rows, summary, predictions


def page(pdf, number, title=None, subtitle=None):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.085, 0.915], [0.94, 0.94], color=BLUE, lw=1.25)
    ax.text(0.085, 0.952, "IAS0360  |  HOME ASSIGNMENT 2", color=BLUE,
            fontsize=8, fontweight="bold", va="bottom")
    if title:
        ax.text(0.085, 0.895, title, color=BLUE, fontsize=17, fontweight="bold", va="top")
        if subtitle:
            ax.text(0.085, 0.858, subtitle, color=GRAY, fontsize=9.2, va="top")
    ax.plot([0.085, 0.915], [0.065, 0.065], color="#C7D0DA", lw=0.7)
    ax.text(0.085, 0.044, "Tanishq Vashishth  ·  Tallinn University of Technology",
            fontsize=7.3, color=GRAY, va="center")
    ax.text(0.915, 0.044, str(number), fontsize=7.3, color=GRAY,
            ha="right", va="center")
    return fig, ax


def section(ax, y, heading, body=None, width=104, size=9.0, gap=0.006):
    ax.text(0.085, y, heading, color=BLUE, fontsize=11.2, fontweight="bold", va="top")
    y -= 0.030
    if body:
        paragraphs = body if isinstance(body, list) else [body]
        for para in paragraphs:
            lines = textwrap.wrap(para, width=width, break_long_words=False,
                                  break_on_hyphens=False) or [""]
            ax.text(0.085, y, "\n".join(lines), color=INK, fontsize=size,
                    linespacing=1.35, va="top")
            y -= 0.0182 * len(lines) + gap
    return y - 0.012


def add_table(ax, y, columns, rows, widths, font=7.2, row_height=0.032,
              header_height=0.042, x=0.085, total_width=0.83):
    norm = np.array(widths, dtype=float)
    norm = norm / norm.sum() * total_width
    xpos = [x]
    for w in norm[:-1]:
        xpos.append(xpos[-1] + w)
    bottom = y - header_height - len(rows) * row_height
    ax.add_patch(plt.Rectangle((x, y - header_height), total_width, header_height,
                               color=BLUE, lw=0, zorder=0))
    for i, col in enumerate(columns):
        ax.text(xpos[i] + 0.004, y - header_height / 2, col, color="white",
                fontsize=font, fontweight="bold", va="center", ha="left")
    for r, row in enumerate(rows):
        yy = y - header_height - (r + 0.5) * row_height
        if r % 2 == 0:
            ax.add_patch(plt.Rectangle((x, y - header_height - (r + 1) * row_height),
                                       total_width, row_height, color=PALE, lw=0, zorder=0))
        for i, value in enumerate(row):
            ax.text(xpos[i] + 0.004, yy, str(value), color=INK, fontsize=font,
                    va="center", ha="left")
    ax.plot([x, x + total_width], [bottom, bottom], color=BLUE, lw=0.8)
    return bottom - 0.014


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary, predictions = load_data()
    by_name = {r["experiment"]: r for r in rows}
    own_preview = STUDY / "own_digits" / "preview_0_to_9.png"
    if not own_preview.exists() or len(predictions) != 8:
        raise RuntimeError("Expected the saved own-digit preview and all eight prediction files.")

    pdf_path = OUT / "report_preview.pdf"
    with PdfPages(pdf_path, metadata={
        "Title": "MNIST Digit Classification and Touchscreen Evaluation",
        "Author": "Tanishq Vashishth",
        "Subject": "IAS0360 Home Assignment 2",
    }) as pdf:
        # Page 1: paper-style title, abstract, objective, input pipeline and models.
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.plot([0.085, 0.915], [0.94, 0.94], color=BLUE, lw=1.25)
        ax.text(0.085, 0.952, "IAS0360  |  HOME ASSIGNMENT 2", color=BLUE,
                fontsize=8, fontweight="bold", va="bottom")
        ax.text(0.5, 0.895, "MNIST Digit Classification with\nTouchscreen Evaluation",
                color=BLUE, fontsize=20, fontweight="bold", ha="center", va="top", linespacing=1.2)
        ax.text(0.5, 0.825, "Tanishq Vashishth  ·  UNI-ID: 266058IV\nTallinn University of Technology",
                color=GRAY, fontsize=9.5, ha="center", va="top", linespacing=1.5)
        y = 0.765
        y = section(ax, y, "Abstract",
            "This work builds and evaluates a small TensorFlow classifier for handwritten digits. "
            "A custom convolutional neural network was trained on MNIST and compared with a LeNet-style "
            "network and a multilayer perceptron. The experiments varied training settings including epoch "
            "count, learning rate, dropout, network width and optimizer. The best validation result was "
            "98.47% for the custom CNN trained for 10 epochs. The selected models were then evaluated on "
            "MNIST's reserved test set and on ten digits drawn with a touchscreen and collected through the "
            "Pico SD-card workflow. LeNet classified 9 of the 10 personal samples correctly. Because there "
            "was only one personal example per class, that result is a small demonstration rather than a "
            "reliable estimate of handwriting accuracy.", width=103, size=8.7)
        ax.text(0.085, y, "Keywords—MNIST, digit classification, convolutional neural network, TensorFlow, touchscreen",
                fontsize=8.2, color=GRAY, va="top")
        y -= 0.045
        y = section(ax, y, "1. Introduction",
            "The image-based part of Home Assignment 2 asks for a neural network trained on MNIST, "
            "a set of ten touchscreen digit samples, and experiments that compare training or model "
            "choices. The goal selected for this work was classification accuracy. The implementation "
            "therefore keeps the input and data split fixed while changing one main setting at a time in "
            "the custom CNN experiments. Two additional model families were trained to provide the "
            "required architecture comparison.", width=103)
        y = section(ax, y, "2. Data and preprocessing",
            "MNIST contains 60,000 training images and 10,000 test images. The training portion was "
            "split with a fixed random seed (42) into 54,000 training and 6,000 validation examples. "
            "The official 10,000-image test set was held aside until model selection was complete. Each "
            "input is a 28 × 28 grayscale image, converted to a single-channel float tensor and scaled "
            "to [0, 1]. The touchscreen captures are read from PGM files without changing the originals. "
            "Preprocessing converts to grayscale, handles background polarity, crops around the stroke, "
            "preserves its aspect ratio, scales it into a 20 × 20 area, and centers it on a black 28 × 28 canvas.",
            width=103)
        y = section(ax, y, "3. Models",
            "The custom CNN uses two 3 × 3 convolution layers with 8 and 16 filters, each followed by "
            "max pooling, then a 32-unit dense layer and a ten-class softmax output. It has 14,410 trainable "
            "parameters. The wider CNN keeps this structure but increases the convolution widths to 16 and "
            "32 and the dense layer to 64 units. The LeNet-style model uses tanh convolutions, average pooling "
            "and dense layers; it is adapted to 28 × 28 input. The MLP flattens the image and uses dense layers "
            "of 128 and 64 units. All models output probabilities for digits 0–9.", width=103)
        ax.plot([0.085, 0.915], [0.065, 0.065], color="#C7D0DA", lw=0.7)
        ax.text(0.085, 0.044, "Tanishq Vashishth  ·  Tallinn University of Technology", fontsize=7.3, color=GRAY, va="center")
        ax.text(0.915, 0.044, "1", fontsize=7.3, color=GRAY, ha="right", va="center")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: experimental setup and requested optimization comparison.
        fig, ax = page(pdf, 2, "Training experiments", "Fixed MNIST split, CPU inference, and saved best-validation checkpoints")
        y = 0.825
        y = section(ax, y, "4. Training procedure",
            "The models were implemented in TensorFlow/Keras and run in the project's Docker environment. "
            "All runs use sparse categorical cross-entropy, a batch size of 64, the same seed and the same "
            "train/validation split. The baseline custom CNN uses Adam with learning rate 0.001 for 5 epochs. "
            "For each run, the checkpoint with the highest validation accuracy was retained. After all runs, "
            "the best validation checkpoint from each architecture family was evaluated once on the reserved "
            "MNIST test set. No test-set result was used to tune the models.", width=103)
        y = section(ax, y, "5. Optimization trials and measured results")
        changes = {
            "baseline": "CNN baseline",
            "more_epochs": "10 epochs",
            "lower_learning_rate": "Adam lr 0.0003",
            "dropout": "Dropout 0.25",
            "wider_cnn": "CNN 16/32; dense 64",
            "sgd_momentum": "SGD lr 0.01; mom. 0.9",
            "lenet": "LeNet-style",
            "mlp": "MLP",
        }
        table_rows = []
        for r in rows:
            table_rows.append([
                r["experiment"], changes[r["experiment"]], str(r["epochs"]),
                f'{r["parameters"]:,}', f'{r["model_file_bytes"] / 1024:.1f}',
                f'{100*r["validation_accuracy"]:.2f}%', f'{100*r["own_accuracy"]:.0f}%',
                f'{r["own_average_ms"]:.3f}',
            ])
        y = add_table(ax, y, ["Run", "Change", "Ep.", "Params", "H5 KiB", "Val.", "Own", "ms/img"],
                      table_rows, [1.45, 2.10, .45, .90, .70, .72, .55, .82], font=6.4,
                      row_height=0.031, header_height=0.039)
        y = section(ax, y, "How to read the table",
            "Validation accuracy is measured over 6,000 MNIST examples. Own-data accuracy is measured "
            "over ten touchscreen samples, one for each digit. The average inference time is the warmed, "
            "serial time for those ten inputs divided by ten; file reading and preprocessing are excluded. "
            "It was measured in the Docker CPU environment, not on a Raspberry Pi or Pico. The model file "
            "size includes its weights, so parameter count, weight bytes and H5 file size are different "
            "descriptions of model size and must not be added together. H5 size is shown in the table; the "
            "full results CSV also records weight bytes.", width=103, size=8.5)
        y = section(ax, y, "Architecture-level test results",
            "Reserved-test accuracy after validation-based selection: custom CNN 98.56%, LeNet 98.21%, "
            "and MLP 97.44%. This is one fixed split; the custom CNN also had more parameter trials.",
            width=103, size=8.2)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: real sample evidence, digit-wise predictions, and a small confusion matrix.
        fig, ax = page(pdf, 3, "Touchscreen evaluation", "Ten SD-card samples were converted to the same 28 × 28 model input")
        y = 0.825
        y = section(ax, y, "6. Personal digit results",
            "One digit from each class was drawn on the touchscreen, saved to the SD card, exported as a "
            "PGM file and processed by the same preprocessing code used for evaluation. The figure below "
            "shows the ten processed inputs. Each saved model predicted all ten samples. LeNet had the "
            "highest score on this small set: 9/10 (90%). The custom CNN scored 8/10 (80%) and the MLP "
            "scored 8/10 (80%). This personal sample check was performed after MNIST training; it was not "
            "used to adjust weights or choose the validation winner.", width=103, size=8.7)
        img = Image.open(own_preview).convert("RGB")
        ax.imshow(img, extent=(0.20, 0.80, 0.605, 0.735), aspect="auto", zorder=3)
        ax.text(0.5, 0.755, "Processed touchscreen samples (labels 0–9)", color=GRAY,
                fontsize=8, ha="center", va="bottom")
        ax.add_patch(plt.Rectangle((0.19, 0.598), 0.62, 0.145, fill=False, edgecolor="#CBD5E1", lw=0.8))
        y = 0.565
        y = section(ax, y, "Predictions by digit")
        arch_pred = [predictions["more_epochs"], predictions["lenet"], predictions["mlp"]]
        pred_rows = [[str(d), str(arch_pred[0][d]), str(arch_pred[1][d]), str(arch_pred[2][d]),
                      "yes" if arch_pred[0][d] == d else "no",
                      "yes" if arch_pred[1][d] == d else "no",
                      "yes" if arch_pred[2][d] == d else "no"] for d in range(10)]
        y = add_table(ax, y, ["Actual", "CNN", "LeNet", "MLP", "CNN ok", "LeNet ok", "MLP ok"],
                      pred_rows, [1.0, .85, .85, .85, 1.0, 1.1, 1.0], font=7.0,
                      row_height=0.022, header_height=0.032, total_width=0.66, x=0.17)
        ax.text(0.085, y - 0.01,
                "LeNet's single error was the drawn 6 predicted as 5. The custom CNN's errors were 6→5 and 8→3; "
                "the MLP's were 0→2 and 8→2.", color=INK, fontsize=8.1, va="top", wrap=True)
        ax.text(0.085, 0.12,
                "Figure note: the preview contains the processed model inputs, not a screenshot of the touchscreen. "
                "The original PGM captures are kept separately in data/own_digits/ and data/raw_sd_card/.",
                color=GRAY, fontsize=7.5, va="top", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: interpretation, limitations, conclusion, and sources.
        fig, ax = page(pdf, 4, "Discussion and conclusion", "What the experiments show, and what remains limited")
        y = 0.825
        y = section(ax, y, "7. Results and discussion",
            "The 10-epoch custom CNN achieved the best validation accuracy (98.47%), compared with 98.10% "
            "for the 5-epoch baseline. This is an increase of 0.37 percentage points. The reserved test "
            "accuracy for the selected custom CNN was 98.56%. The additional epochs improved validation "
            "accuracy slightly without changing the model's 14,410 parameters or 77.3 KiB serialized model "
            "file. Increasing the CNN width raised the parameter count to 56,714 and H5 size to 243.3 KiB, "
            "but validation accuracy was 98.38%, below the 10-epoch run. Dropout matched the baseline's "
            "98.10%; the lower learning rate and SGD trials were lower at 97.12% and 98.13%, respectively. "
            "The results suggest that simply making this small model wider was not useful for this run.", width=103)
        y = section(ax, y, "8. Limitations",
            "The personal-data result is based on only ten drawings, with one sample per class. A single "
            "unusual shape changes the score by ten percentage points, so 90% should not be read as a stable "
            "estimate of how the model handles future handwriting. The MNIST validation and test results "
            "measure performance on the MNIST distribution, which differs from this touchscreen writing "
            "style. The timing values are CPU measurements from a container and exclude preprocessing; they "
            "do not predict performance on the course hardware. Training trials used one seed and one split, "
            "and the architecture families did not receive equal hyperparameter-search budgets. No quantization "
            "or hardware acceleration experiment was part of this study.", width=103)
        y = section(ax, y, "9. Conclusion",
            "A complete image-classification workflow was built: MNIST preparation, custom CNN training, "
            "optimization trials, comparison with LeNet and an MLP, touchscreen capture, SD-card export, "
            "preprocessing and evaluation. The custom CNN was the best model on the reserved MNIST test set "
            "among the selected models, while LeNet classified the ten personal samples most accurately. "
            "Those findings answer different questions and should be reported separately. Collecting more "
            "touchscreen samples per digit would make the personal-data comparison more informative.", width=103)
        y = section(ax, y, "References and reproducibility",
            "[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document "
            "recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998. [2] TensorFlow, "
            "TensorFlow and Keras documentation. [3] IAS0360 course repository, Lab 2.2 MNIST CNN starter "
            "notebook, 2026. The experiment code, model summaries, run settings, CSV results, own-digit "
            "predictions and processed preview are included with the submission under results/study/.",
            width=103, size=8.3)
        ax.text(0.085, 0.105, "Report generated from the saved experiment artifacts in results/study/.",
                color=GRAY, fontsize=7.5, va="top")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
