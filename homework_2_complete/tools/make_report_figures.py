import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

root = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir", type=pathlib.Path,
    default=root / "results" / "pretrained_finetuned_trials" / "report_data",
)
results_dir = parser.parse_args().results_dir
report_images = root / "report" / "images"
report_images.mkdir(parents=True, exist_ok=True)

with (results_dir / "results.csv").open(newline="") as result_file:
    rows = list(csv.DictReader(result_file))

names = [row["model"] for row in rows]
size_kib = [int(row["tflite_model_bytes"]) / 1024 for row in rows]
validation = [float(row["validation_accuracy"].rstrip("%")) for row in rows]
own_accuracy = [
    float(row["own_data_accuracy"].rstrip("%")) for row in rows
]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
for i, row in enumerate(rows):
    ax.scatter(
        size_kib[i],
        validation[i],
        s=85,
        color=colors[i % len(colors)],
        edgecolor="black",
        linewidth=0.5,
        label=names[i],
        zorder=3,
    )
    ax.annotate(
        f"{names[i]}\nOwn data: {own_accuracy[i]:.0f}%",
        (size_kib[i], validation[i]),
        xytext=(8, 9 if i % 2 == 0 else -30),
        textcoords="offset points",
        fontsize=8,
        ha="left",
        va="bottom" if i % 2 == 0 else "top",
    )
ax.set_xscale("log")
ax.set_xlabel("TFLite model size (KiB, log scale)")
ax.set_ylabel("MNIST validation accuracy (%)")
ax.set_title("Model size and accuracy comparison")
ax.set_ylim(min(validation) - 2, 100)
ax.set_xlim(min(size_kib) / 1.7, max(size_kib) * 2.4)
ax.grid(True, which="both", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(report_images / "acc_vs_size.png", dpi=220, bbox_inches="tight")
plt.close(fig)

architecture_table = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Architecture comparison using only the Reference trial for each "
    r"model (Adam, lr=0.001, 100 epochs, batch 128; int8).}",
    r"\label{tab:results}",
    r"\small",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}ll c r r c c@{}}",
    r"\toprule",
    r"Model & Optimization & Avg. ms/image & Params & Model bytes & MNIST val. & Own acc. \\",
    r"\midrule",
]
for row in rows:
    validation_text = row["validation_accuracy"].replace("%", r"\%")
    own_text = row["own_data_accuracy"].replace("%", r"\%")
    architecture_table.append(
        f"{row['model']} & Adam, lr=0.001; 100 epochs, int8 & "
        f"{row['own_data_avg_inference_ms']} & {int(row['parameters']):,} & "
        f"{int(row['tflite_model_bytes']):,} & {validation_text} & {own_text} \\\\"
    )
architecture_table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
(root / "report" / "architecture_results_table.tex").write_text(
    "\n".join(architecture_table), encoding="utf-8"
)

short_labels = {
    "Initial CNN 16/32/d64": "Initial CNN",
    "Optimized CNN 8/16/d32": "Optimized CNN",
    "SqueezeNet 1.1": "SqueezeNet",
}
reference_summary = [
    r"\begin{table}[!ht]",
    r"\centering",
    r"\caption{Reference trial: accuracy, size and inference time.}",
    r"\label{tab:reference-summary}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}lrrrrr@{}}",
    r"\toprule",
    r"Model & Params & Bytes & Val. (\%) & Own (\%) & ms \\",
    r"\midrule",
]
for row in rows:
    reference_summary.append(
        f"{short_labels.get(row['model'], row['model'])} & "
        f"{int(row['parameters']):,} & {int(row['tflite_model_bytes']):,} & "
        f"{row['validation_accuracy'].rstrip('%')} & "
        f"{row['own_data_accuracy'].rstrip('%')} & "
        f"{row['own_data_avg_inference_ms']} \\\\"
    )
reference_summary.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    "",
])
(root / "report" / "reference_summary_table.tex").write_text(
    "\n".join(reference_summary), encoding="utf-8"
)

fig, axes = plt.subplots(2, 1, figsize=(12, 5.2))
architecture_rows = [
    (
        "Initial custom CNN — 56,714 parameters",
        [
            "Input\n28×28×1",
            "Conv2D\n3×3, 16\nReLU",
            "MaxPool\n2×2",
            "Conv2D\n3×3, 32\nReLU",
            "MaxPool\n2×2",
            "Flatten\n800 → Dense 64\nReLU",
            "Dense 10\nSoftmax",
        ],
    ),
    (
        "Memory-optimized CNN — 14,410 parameters",
        [
            "Input\n28×28×1",
            "Conv2D\n3×3, 8\nReLU",
            "MaxPool\n2×2",
            "Conv2D\n3×3, 16\nReLU",
            "MaxPool\n2×2",
            "Flatten\n400 → Dense 32\nReLU",
            "Dense 10\nSoftmax",
        ],
    ),
]
for ax, (title, stages) in zip(axes, architecture_rows):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=4)
    box_width, gap = 0.125, 0.018
    start = (1 - (7 * box_width + 6 * gap)) / 2
    for index, stage in enumerate(stages):
        x = start + index * (box_width + gap)
        ax.text(
            x + box_width / 2,
            0.48,
            stage,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#eaf2f8" if index not in (0, 6) else "#d5e8d4",
                "edgecolor": "#456b82",
                "linewidth": 1.1,
            },
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_width + gap * 0.85, 0.48),
                xytext=(x + box_width + gap * 0.12, 0.48),
                arrowprops={"arrowstyle": "->", "color": "#456b82", "lw": 1.2},
            )
fig.text(
    0.5,
    0.015,
    "Both models: trained in float32, then exported as full-int8 TFLite "
    "(post-training quantization; 200 MNIST training images for calibration).",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8)
fig.savefig(
    report_images / "custom_cnn_architectures.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)

digit_files = sorted(
    (root / "own_data").glob("*.png"), key=lambda path: int(path.name[0])
)
thumb_size = (112, 112)
margin = 16
label_height = 24
columns = 5
rows_count = (len(digit_files) + columns - 1) // columns
sheet = Image.new(
    "RGB",
    (
        margin + columns * (thumb_size[0] + margin),
        margin + rows_count * (thumb_size[1] + label_height + margin),
    ),
    "white",
)

draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default()
for index, path in enumerate(digit_files):
    image = Image.open(path).convert("L").resize(thumb_size, Image.Resampling.NEAREST)
    x = margin + (index % columns) * (thumb_size[0] + margin)
    y = margin + (index // columns) * (thumb_size[1] + label_height + margin)
    sheet.paste(ImageOps.invert(image).convert("RGB"), (x, y))
    draw.text((x, y + thumb_size[1] + 4), f"Digit {path.name[0]}", fill="black", font=font)

sheet.save(report_images / "own_data_sheet.png")

prediction_path = results_dir / "own_data_predictions.csv"
with prediction_path.open(newline="") as prediction_file:
    predictions = list(csv.DictReader(prediction_file))

model_names = list(dict.fromkeys(row["model"] for row in predictions))
by_image_model = {
    (row["image"], row["model"]): row for row in predictions
}
image_names = sorted(
    {row["image"] for row in predictions}, key=lambda name: int(name[0])
)

table_model_names = names
per_image_labels = {
    "Initial CNN 16/32/d64": "Initial CNN",
    "Optimized CNN 8/16/d32": "Optimized CNN",
}
per_image_table = [
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{Touch-screen predictions using only each model's Reference "
    r"trial (Adam, lr=0.001, 100 epochs, batch 128; int8). "
    r"Cells show prediction and confidence; bold and * denote errors.}",
    r"\label{tab:perimage}",
    r"\small",
    r"\setlength{\tabcolsep}{8pt}",
    r"\begin{tabular}{@{}" + " ".join(["c"] * (2 + len(table_model_names))) + r"@{}}",
    r"\toprule",
    r"Input & Actual & "
    + " & ".join(per_image_labels.get(name, name) for name in table_model_names)
    + r" \\",
    r"\midrule",
]
for image_name in image_names:
    first_prediction = by_image_model[(image_name, table_model_names[0])]
    cells = []
    for model_name in table_model_names:
        prediction = by_image_model[(image_name, model_name)]
        if prediction["actual"] != first_prediction["actual"]:
            raise ValueError(f"Inconsistent actual digit for {image_name}")
        confidence = prediction["confidence"].replace("%", r"\%")
        entry = f"{prediction['predicted']} ({confidence})"
        if prediction["correct"].lower() == "false":
            entry = rf"\textbf{{{entry}*}}"
        elif prediction["correct"].lower() != "true":
            raise ValueError(f"Invalid correctness value for {image_name}")
        cells.append(entry)
    image_path = f"../own_data/{image_name}"
    per_image_table.append(
        rf"\includegraphics[height=0.17in]{{\detokenize{{{image_path}}}}} & "
        f"{first_prediction['actual']} & " + " & ".join(cells) + r" \\"
    )
per_image_table.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
    "",
])
(root / "report" / "per_image_table.tex").write_text(
    "\n".join(per_image_table), encoding="utf-8"
)

font = ImageFont.load_default()
image_cell, actual_cell, model_cell = 76, 80, 190
header_height, row_height = 42, 48
comparison = Image.new(
    "RGB",
    (
        image_cell + actual_cell + len(model_names) * model_cell,
        header_height + len(image_names) * row_height,
    ),
    "white",
)
draw = ImageDraw.Draw(comparison)
draw.text((8, 14), "Image", fill="black", font=font)
draw.text((image_cell + 8, 14), "Actual", fill="black", font=font)
for model_index, model_name in enumerate(model_names):
    x = image_cell + actual_cell + model_index * model_cell
    draw.text((x + 8, 8), model_name, fill="black", font=font)
    draw.text((x + 8, 23), "Prediction / outcome", fill="black", font=font)

for row_index, image_name in enumerate(image_names):
    y = header_height + row_index * row_height
    image = Image.open(root / "own_data" / image_name).convert("L")
    image = ImageOps.invert(image.resize((36, 36), Image.Resampling.NEAREST))
    comparison.paste(image.convert("RGB"), (8, y + 5))
    draw.text((image_cell + 22, y + 17), image_name[0], fill="black", font=font)
    for model_index, model_name in enumerate(model_names):
        prediction = by_image_model[(image_name, model_name)]
        x = image_cell + actual_cell + model_index * model_cell
        is_correct = prediction["correct"].lower() == "true"
        color = "#d9f2d9" if is_correct else "#ffd9d9"
        draw.rectangle((x, y, x + model_cell - 1, y + row_height - 1), fill=color)
        outcome = "OK" if is_correct else "MISS"
        text = (
            f"Pred {prediction['predicted']} | {outcome} | "
            f"{prediction['confidence']}"
        )
        draw.text((x + 8, y + 17), text, fill="black", font=font)

comparison.save(report_images / "own_data_model_comparison.png")

optimized_name = "Optimized CNN 8/16/d32"
tile_width, tile_height = 210, 205
margin, columns = 16, 5
detail = Image.new(
    "RGB",
    (
        margin + columns * (tile_width + margin),
        margin + 2 * (tile_height + margin),
    ),
    "white",
)
draw = ImageDraw.Draw(detail)
optimized_predictions = {
    row["image"]: row
    for row in predictions
    if row["model"] == optimized_name
}
for index, image_name in enumerate(image_names):
    prediction = optimized_predictions[image_name]
    x = margin + (index % columns) * (tile_width + margin)
    y = margin + (index // columns) * (tile_height + margin)
    correct = prediction["correct"].lower() == "true"
    color = "#267326" if correct else "#b22222"
    digit = Image.open(root / "own_data" / image_name).convert("L")
    digit = ImageOps.invert(
        digit.resize((140, 140), Image.Resampling.NEAREST)
    ).convert("RGB")
    detail.paste(digit, (x + (tile_width - 140) // 2, y))
    draw.text((x + 8, y + 146), f"Actual: {prediction['actual']}", fill="black", font=font)
    draw.text((x + 8, y + 162), f"Predicted: {prediction['predicted']}", fill="black", font=font)
    draw.text((x + 8, y + 178), f"Confidence: {prediction['confidence']}", fill="black", font=font)
    draw.text((x + 135, y + 178), "CORRECT" if correct else "WRONG", fill=color, font=font)

detail.save(report_images / "optimized_cnn_own_data_details.png")

trials_path = results_dir / "optimization_trials.csv"
summary_path = results_dir / "optimization_summary.csv"
if trials_path.exists() and summary_path.exists():
    with trials_path.open(newline="") as trials_file:
        trial_rows = list(csv.DictReader(trials_file))
    with summary_path.open(newline="") as summary_file:
        summary_rows = list(csv.DictReader(summary_file))

    model_labels = {
        "Initial CNN 16/32/d64": "Initial CNN (16/32/d64)",
        "Optimized CNN 8/16/d32": "Optimized CNN (8/16/d32)",
        "SqueezeNet 1.1": "SqueezeNet 1.1",
    }
    config_labels = {
        "Reference: Adam, 0.001": "Reference",
        "Initial assignment settings": "Assignment",
        "SGD, adjusted learning rate": "SGD adjusted",
        "Adam comparison": "Adam",
        "Extended training": "Extended",
    }

    def tex_percentage(value):
        return value.replace("%", r"\%")

    def trial_row_tex(row):
        return (
            f"{model_labels[row['model']]} & "
            f"{config_labels[row['optimization']]} & "
            f"{row['optimizer'].upper()}, {float(row['learning_rate']):g} & "
            f"{row['epochs']} / {row['batch_size']} & "
            f"{row['own_data_avg_inference_ms']} & "
            f"{int(row['parameters']):,} & "
            f"{int(row['tflite_model_bytes']):,} & "
            f"{tex_percentage(row['validation_accuracy'])} & "
            f"{tex_percentage(row['own_data_accuracy'])} \\\\"
        )

    matrix_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Training configuration trials on all models.}",
        r"\label{tab:training-trials}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll l c c r r c c@{}}",
        r"\toprule",
        r"Model & Trial & Optimizer, lr & Epochs / batch & Avg. ms & Params & "
        r"TFLite bytes & MNIST val. & Own acc. \\",
        r"\midrule",
        *[trial_row_tex(row) for row in trial_rows],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Final selections from the training and architecture trials.}",
        r"\label{tab:optimization-summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll l c r r c c@{}}",
        r"\toprule",
        r"Selection & Model & Training setup & Avg. ms & Params & TFLite bytes "
        r"& MNIST val. & Own acc. \\",
        r"\midrule",
    ]
    for row in summary_rows:
        matrix_lines.append(
            f"{row['selection']} & {model_labels[row['model']]} & "
            f"{row['optimizer'].upper()}, lr={float(row['learning_rate']):g}, "
            f"{row['epochs']} epochs & {row['own_data_avg_inference_ms']} & "
            f"{int(row['parameters']):,} & {int(row['tflite_model_bytes']):,} & "
            f"{tex_percentage(row['validation_accuracy'])} & "
            f"{tex_percentage(row['own_data_accuracy'])} \\\\"
        )
    matrix_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    (root / "report" / "optimization_tables.tex").write_text(
        "\n".join(matrix_lines), encoding="utf-8"
    )
