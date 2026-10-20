#!/usr/bin/env python3
import copy
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import ai_edge_litert.interpreter as tflite
import litert_torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from ai_edge_quantizer import Quantizer, recipe
from matplotlib.axes import Axes
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

matplotlib.use("Agg")

torch.manual_seed(10)
np.random.seed(10)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 6.4,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
})

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DATA_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "data"))
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
TABLES_DIR = os.path.join(PROJECT_DIR, "tables")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

for directory in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


@dataclass(frozen=True)
class Fp32Benchmark:
    val_accuracy: float
    touch_accuracy: float
    avg_speed_ms: float
    num_params: int
    num_nonzero_params: int
    size_kb: float


@dataclass(frozen=True)
class Int8Benchmark:
    val_accuracy: float
    touch_accuracy: float
    touch_predictions: np.ndarray
    avg_speed_ms: float
    size_kb: float


@dataclass(frozen=True)
class ExperimentResult:
    table_label: str
    memory_tex: str
    avg_speed_ms: float
    size_kb: float
    val_accuracy: float
    touch_accuracy: float


@dataclass(frozen=True)
class ArchitectureResult:
    name: str
    num_params: str
    fp32_speed_ms: float
    int8_size_kb: float
    int8_speed_ms: float
    int8_val_accuracy: float
    int8_touch_accuracy: float
    touch_predictions: np.ndarray


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    has_repo_mnist = os.path.exists(os.path.join(REPO_DATA_DIR, "MNIST"))
    data_root, download = (REPO_DATA_DIR, False) if has_repo_mnist else ("/tmp", True)

    def to_arrays(train: bool) -> tuple[np.ndarray, np.ndarray]:
        dataset = datasets.MNIST(root=data_root, train=train, download=download)
        images = (dataset.data.numpy() / 255.0).astype(np.float32)[:, np.newaxis, :, :]
        labels = dataset.targets.numpy().astype(np.int64)
        return images, labels

    return to_arrays(True), to_arrays(False)


def load_touch_samples():
    normalized_images = []
    raw_matrices = []
    labels = []

    for digit in range(10):
        sample_id = 22 + digit
        filepath = os.path.join(SAMPLES_DIR, f"sample_{sample_id}.txt")
        rows = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    row = [int(x.strip()) for x in line.split(",") if x.strip().isdigit()]
                    if len(row) == 28:
                        rows.append(row)

        matrix = np.array(rows, dtype=np.uint8)
        raw_matrices.append(matrix)
        peak_value = float(matrix.max()) if matrix.max() > 0 else 1.0
        normalized_img = (matrix.astype(np.float32) / peak_value)[np.newaxis, :, :]
        normalized_images.append(normalized_img)
        labels.append(digit)

    touch_x = np.array(normalized_images, dtype=np.float32)
    touch_y = np.array(labels, dtype=np.int64)
    return touch_x, touch_y, raw_matrices


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero_params(model: nn.Module) -> int:
    return sum(int((p != 0).sum().item()) for p in model.parameters() if p.requires_grad)


def build_baseline_cnn(
    num_filters1: int = 8,
    num_filters2: int = 16,
    dense_units: int = 32,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Module:
    def make_activation() -> nn.Module:
        return nn.LeakyReLU(0.1) if activation.lower() == "leaky_relu" else nn.ReLU()

    layers: list[nn.Module] = [
        nn.Conv2d(1, num_filters1, kernel_size=3),
        make_activation(),
        nn.MaxPool2d(2),
        nn.Conv2d(num_filters1, num_filters2, kernel_size=3),
        make_activation(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(num_filters2 * 5 * 5, dense_units),
        make_activation(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dense_units, 10))
    return nn.Sequential(*layers)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.bn2(self.pw(self.act1(self.bn1(self.dw(x))))))


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(8, 16, stride=1),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(16, 24, stride=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(24 * 3 * 3, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FireModule(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int, expand_channels: int):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(),
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(),
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = self.squeeze(x)
        return torch.cat([self.expand1x1(squeezed), self.expand3x3(squeezed)], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            FireModule(8, squeeze_channels=4, expand_channels=8),
            nn.MaxPool2d(2),
            FireModule(16, squeeze_channels=6, expand_channels=12),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(24 * 3 * 3, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    table_label: str
    builder: Callable[[], nn.Module] = build_baseline_cnn
    optimizer: str = "adam"
    lr: float = 0.001
    epochs: int = 8
    batch_size: int = 64


@dataclass(frozen=True)
class ArchitectureConfig:
    id: str
    checkpoint: str
    name: str
    builder: Callable[[], nn.Module] = build_baseline_cnn
    epochs: int = 8
    lr: float = 0.001
    batch_size: int = 64


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 0.001,
    optimizer_name: str = "adam",
) -> nn.Module:
    torch.set_num_threads(4)
    criterion = nn.CrossEntropyLoss()

    optimizer: optim.Optimizer
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def measure_avg_latency_ms(
    step_fn: Callable[[], None],
    num_samples: int,
    warmup: int = 50,
    repeats: int = 1000,
) -> float:
    for _ in range(warmup):
        step_fn()
    start_time = time.perf_counter()
    for _ in range(repeats):
        step_fn()
    elapsed = time.perf_counter() - start_time
    return (elapsed / (repeats * num_samples)) * 1000.0


def load_or_train(
    model: nn.Module,
    checkpoint_path: str,
    label: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer_name: str = "adam",
    l1_prune_amount: float = 0.0,
) -> nn.Module:
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model.eval()
        return model

    print(f"Training {label}...", flush=True)
    prunable = (
        [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        if l1_prune_amount > 0.0
        else []
    )
    for module in prunable:
        prune.l1_unstructured(module, name="weight", amount=l1_prune_amount)

    torch.manual_seed(10)
    model = train_model(
        model, x_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr, optimizer_name=optimizer_name,
    )
    for module in prunable:
        prune.remove(module, "weight")

    torch.save(model.state_dict(), checkpoint_path)
    return model


def benchmark_fp32_model(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    touch_x: np.ndarray,
    touch_y: np.ndarray,
    model_file_path: str,
) -> Fp32Benchmark:
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()

    touch_tensor = torch.from_numpy(touch_x)
    test_tensor = torch.from_numpy(x_test)

    with torch.no_grad():
        val_predictions = model(test_tensor).argmax(dim=1).numpy()
        val_accuracy = float(np.mean(val_predictions == y_test))

        touch_predictions = model(touch_tensor).argmax(dim=1).numpy()
        touch_accuracy = float(np.mean(touch_predictions == touch_y))

        def infer_touch() -> None:
            _ = model(touch_tensor)

        avg_speed_ms = measure_avg_latency_ms(infer_touch, len(touch_x))

    torch.set_num_threads(prev_threads)

    return Fp32Benchmark(
        val_accuracy=val_accuracy,
        touch_accuracy=touch_accuracy,
        avg_speed_ms=float(avg_speed_ms),
        num_params=count_trainable_params(model),
        num_nonzero_params=count_nonzero_params(model),
        size_kb=os.path.getsize(model_file_path) / 1024.0,
    )


def quantize_model_to_int8(model: nn.Module, x_train: np.ndarray, output_path: str) -> None:
    if os.path.exists(output_path):
        return

    model.eval()
    sample_inputs = (torch.randn(1, 1, 28, 28),)
    edge_model = litert_torch.convert(model, sample_inputs)
    temp_float_path = output_path + ".float.tflite"
    edge_model.export(temp_float_path)

    with open(temp_float_path, "rb") as f:
        float_model_bytes = f.read()

    quantizer = Quantizer(float_model_bytes)
    quantizer.load_quantization_recipe(recipe.static_wi8_ai8())
    calibration_samples: list[dict[str, Any]] = [{"args_0": x_train[i:i + 1]} for i in range(100)]
    calibration_data: dict[str, Iterable[dict[str, Any]]] = {"serving_default": calibration_samples}
    calibration_result = quantizer.calibrate(calibration_data)
    quantized_result = quantizer.quantize(calibration_result)

    quantized_bytes = quantized_result.quantized_model
    if quantized_bytes is None:
        raise RuntimeError("Quantization failed to produce model bytes")

    with open(output_path, "wb") as f:
        f.write(quantized_bytes)

    if os.path.exists(temp_float_path):
        os.remove(temp_float_path)


def benchmark_int8_model(
    tflite_path: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    touch_x: np.ndarray,
    touch_y: np.ndarray,
) -> Int8Benchmark:
    interpreter = tflite.Interpreter(model_path=tflite_path, num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index: int = input_details["index"]
    output_index: int = output_details["index"]
    scale, zero_point = input_details["quantization"]

    def quantize_input(image: np.ndarray) -> np.ndarray:
        if scale != 0.0:
            return np.round(image / scale + zero_point).clip(-128, 127).astype(np.int8)
        return image.astype(input_details["dtype"])

    def predict(images: np.ndarray) -> np.ndarray:
        preds: list[int] = []
        for i in range(len(images)):
            interpreter.set_tensor(input_index, quantize_input(images[i:i + 1]))
            interpreter.invoke()
            preds.append(int(np.argmax(interpreter.get_tensor(output_index))))
        return np.array(preds, dtype=np.int64)

    val_accuracy = float(np.mean(predict(x_test) == y_test))
    touch_predictions = predict(touch_x)
    touch_accuracy = float(np.mean(touch_predictions == touch_y))

    quantized_touch = [quantize_input(touch_x[i:i + 1]) for i in range(len(touch_x))]

    def infer_touch() -> None:
        for img in quantized_touch:
            interpreter.set_tensor(input_index, img)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_index)

    avg_speed_ms = measure_avg_latency_ms(infer_touch, len(touch_x))
    size_kb = os.path.getsize(tflite_path) / 1024.0

    return Int8Benchmark(
        val_accuracy=val_accuracy,
        touch_accuracy=touch_accuracy,
        touch_predictions=touch_predictions,
        avg_speed_ms=float(avg_speed_ms),
        size_kb=float(size_kb),
    )


def save_figure(filename: str) -> None:
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close()


def write_latex_table(
    filename: str,
    comment: str,
    tabcolsep: str,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[str],
    footnote: str,
) -> None:
    col_spec = "l" + "c" * (len(headers) - 1)
    header_line = " & ".join(f"\\textbf{{{col}}}" for col in headers) + " \\\\"
    lines = [
        f"% {comment}",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        "\\renewcommand{\\arraystretch}{0.88}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header_line,
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}\\par",
        "\\vspace{2pt}",
        f"{{\\raggedright \\tiny {footnote}\\par}}",
        "\\end{table}",
    ]
    with open(os.path.join(TABLES_DIR, filename), "w") as f:
        f.write("\n".join(lines))


def plot_figure1_digits(touch_x: np.ndarray, raw_matrices: list[np.ndarray]) -> None:
    num_digits = len(raw_matrices)
    fig, axes = plt.subplots(2, num_digits, figsize=(7.2, 1.72), gridspec_kw={"hspace": 0.14, "wspace": 0.12})
    for digit in range(num_digits):
        axes[0, digit].imshow(raw_matrices[digit], cmap="gray_r", aspect=240.0 / 280.0, interpolation="nearest")
        axes[0, digit].set_title(f"'{digit}'", fontsize=11.5, pad=3)
        axes[1, digit].imshow(touch_x[digit, 0, :, :], cmap="gray", aspect="equal", interpolation="nearest")

        for row_idx, spine_color in ((0, "#888888"), (1, "#444444")):
            ax: Axes = axes[row_idx, digit]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(spine_color)
                spine.set_linewidth(1.0)

    fig.text(0.010, 0.61, "LCD Canvas\n(280x240)", va="center", ha="left", fontsize=11.2, fontweight="bold")
    fig.text(0.010, 0.22, "Normalized\n(28x28)", va="center", ha="left", fontsize=11.2, fontweight="bold")
    plt.subplots_adjust(left=0.185, right=0.99, top=0.79, bottom=0.04)
    save_figure("fig1_pipeline_and_digits.png")


def plot_figure2_pareto(experiment_results: list[ExperimentResult]) -> None:
    _, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax1: Axes = axes[0]
    ax2: Axes = axes[1]

    sizes: list[float] = [r.size_kb for r in experiment_results]
    val_accuracies: list[float] = [r.val_accuracy * 100.0 for r in experiment_results]
    latencies_ms: list[float] = [r.avg_speed_ms for r in experiment_results]
    touch_accuracies: list[float] = [r.touch_accuracy * 100.0 for r in experiment_results]

    ax1.scatter(sizes[:7], val_accuracies[:7], color="#1f77b4", s=60, alpha=0.85,
                edgecolors="k", linewidth=0.7, label="Exp 1-7 (H-Params)", zorder=3)
    ax1.scatter([sizes[7]], [val_accuracies[7]], color="#2ca02c", s=70, marker="o",
                edgecolors="k", linewidth=0.7, label="Exp 8 (Ch. Pruned)", zorder=3)
    ax1.scatter([sizes[8]], [val_accuracies[8]], color="#ff7f0e", s=70, marker="s",
                edgecolors="k", linewidth=0.7, label="Exp 9 (50% Sparse)", zorder=4)
    ax1.scatter([sizes[9]], [val_accuracies[9]], color="#d62728", s=125, marker="*",
                edgecolors="k", linewidth=0.7, label="Exp 10 (INT8 PTQ)", zorder=5)

    ax1.legend(loc="lower right", framealpha=0.92, fontsize=7.8,
               borderpad=0.28, labelspacing=0.25, handletextpad=0.3)
    arrow_kw: dict[str, Any] = {"arrowstyle": "->", "lw": 0.9, "shrinkA": 2, "shrinkB": 3}

    def add_callout(
        ax: Axes, text: str, xy: tuple[float, float], xytext: tuple[float, float], color: str, ha: str = "center"
    ) -> None:
        ax.annotate(
            text, xy=xy, xytext=xytext, ha=ha, arrowprops={**arrow_kw, "color": color}, fontsize=8.0, color=color,
        )

    baseline_mean_size = float(np.mean(sizes[:7]))
    baseline_mean_acc = float(np.mean(val_accuracies[:7]))
    add_callout(ax1, f"Exp 1-7\n({baseline_mean_size:.1f} KB)",
                (baseline_mean_size, baseline_mean_acc), (53.0, 99.65), "#1f77b4")
    add_callout(ax1, f"Exp 8: {val_accuracies[7]:.1f}%\n({sizes[7]:.1f} KB)",
                (sizes[7], val_accuracies[7]), (16.0, 99.65), "#2ca02c")
    add_callout(ax1, f"Exp 9: {val_accuracies[8]:.1f}%\n({sizes[8]:.1f} KB)",
                (sizes[8], val_accuracies[8]), (24.5, 96.35), "#d95f02", ha="left")

    compression_ratio = sizes[0] / (sizes[9] if sizes[9] > 0.1 else 0.1)
    add_callout(ax1, f"INT8 ({sizes[9]:.1f} KB)\n{compression_ratio:.1f}x smaller",
                (sizes[9], val_accuracies[9]), (4.5, 94.2), "#d62728", ha="left")

    min_val_acc = float(min(val_accuracies))
    ax1.set_xlim(2, 70)
    ax1.set_ylim(min(92.5, min_val_acc - 1.5), 101.5)
    ax1.set_yticks([94, 96, 98, 100])
    ax1.tick_params(axis="both", labelsize=10.5)
    ax1.set_xlabel("Model Storage Footprint (KB)", fontsize=11.5)
    ax1.set_ylabel("MNIST Val. Accuracy (%)", fontsize=11.5)
    ax1.set_title("(a) Footprint vs. Validation Accuracy", fontsize=12.2, pad=6)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.scatter(latencies_ms[:7], touch_accuracies[:7], color="#1f77b4", s=60, alpha=0.85,
                edgecolors="k", linewidth=0.7, label="Exp 1-7 (FP32)", zorder=3)
    ax2.scatter(latencies_ms[7:9], touch_accuracies[7:9], color="#2ca02c", s=70, marker="s",
                edgecolors="k", linewidth=0.7, label="Exp 8-9 (Pruned)", zorder=4)
    ax2.scatter([latencies_ms[9]], [touch_accuracies[9]], color="#d62728", s=125, marker="*",
                edgecolors="k", linewidth=0.7, label="Exp 10 (INT8 PTQ)", zorder=5)

    ax2.legend(loc="lower left", framealpha=0.92, fontsize=7.8,
               borderpad=0.28, labelspacing=0.25, handletextpad=0.3)

    max_latency = float(max(latencies_ms))
    x_max = max(0.038, max_latency * 1.45)

    add_callout(ax2, f"INT8: {latencies_ms[9]:.3f} ms\n({touch_accuracies[9]:.0f}% touch)",
                (latencies_ms[9], touch_accuracies[9]), (x_max * 0.03, 71), "#d62728", ha="left")

    def format_speed_range(values: list[float]) -> str:
        low, high = float(min(values)), float(max(values))
        return f"{low:.3f} ms" if round(low, 3) == round(high, 3) else f"{low:.3f}-{high:.3f} ms"

    add_callout(ax2, f"Exp 8-9: {touch_accuracies[8]:.0f}%\n({format_speed_range(latencies_ms[7:9])})",
                (latencies_ms[8], touch_accuracies[8]), (x_max * 0.28, 108), "#2ca02c")

    best_baseline_acc = float(max(touch_accuracies[:7]))
    best_baseline_idx = touch_accuracies[:7].index(best_baseline_acc)
    add_callout(ax2, f"Exp 1-7\n({format_speed_range(latencies_ms[:7])})",
                (latencies_ms[best_baseline_idx], touch_accuracies[best_baseline_idx]), (x_max * 0.76, 108), "#1f77b4")

    worst_baseline_idx = int(np.argmin(touch_accuracies[:7]))
    worst_label_x = min(max(latencies_ms[worst_baseline_idx] + x_max * 0.12, x_max * 0.62), x_max * 0.88)
    add_callout(ax2, f"Exp {worst_baseline_idx + 1}\n({touch_accuracies[worst_baseline_idx]:.0f}%)",
                (latencies_ms[worst_baseline_idx], touch_accuracies[worst_baseline_idx]),
                (worst_label_x, 51), "#1f77b4")

    ax2.set_xlim(0.0, x_max)
    ax2.set_ylim(38, 120)
    ax2.set_yticks([50, 60, 70, 80, 90, 100])
    ax2.tick_params(axis="both", labelsize=10.5)
    ax2.set_xlabel("Host CPU Latency (ms / image)", fontsize=11.5)
    ax2.set_ylabel("Touch Screen Accuracy (%)", fontsize=11.5)
    ax2.set_title("(b) Latency vs. Touch Accuracy", fontsize=12.2, pad=6)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure("fig2_optimization_pareto.png")


def plot_figure3_architectures(arch_results: list[ArchitectureResult]) -> None:
    _, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    ax0: Axes = axes[0]
    ax1: Axes = axes[1]
    ax2: Axes = axes[2]
    names = [r.name.replace(" CNN", "") for r in arch_results]
    x = np.arange(len(names))
    single_bar_width = 0.38
    grouped_width = 0.30

    param_counts: list[int] = [int(r.num_params.replace(",", "")) for r in arch_results]
    int8_sizes_kb: list[float] = [r.int8_size_kb for r in arch_results]

    param_bars = ax0.bar(x, param_counts, single_bar_width, color="#1f77b4", edgecolor="k", linewidth=0.5)
    ax0.set_ylim(0, max(param_counts) * 1.38)
    for bar, param_count, size_kb in zip(param_bars.patches, param_counts, int8_sizes_kb):
        ax0.text(
            bar.get_x() + bar.get_width() / 2, param_count + max(param_counts) * 0.03,
            f"{param_count:,}\n({size_kb:.1f} KB)", ha="center", va="bottom", fontsize=6.4, color="#1f77b4",
        )

    fp32_latencies: list[float] = [r.fp32_speed_ms for r in arch_results]
    int8_latencies: list[float] = [r.int8_speed_ms for r in arch_results]

    fp32_bars = ax1.bar(
        x - grouped_width / 2, fp32_latencies, grouped_width,
        label="PyTorch FP32", color="#2ca02c", edgecolor="k", linewidth=0.5,
    )
    int8_bars = ax1.bar(
        x + grouped_width / 2, int8_latencies, grouped_width,
        label="LiteRT INT8", color="#d62728", edgecolor="k", linewidth=0.5,
    )
    max_latency = float(max(fp32_latencies + int8_latencies))
    ax1.set_ylim(0, max_latency * 1.42)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=7.0)
    for fp32_bar, int8_bar, fp32_height, int8_height in zip(
        fp32_bars.patches, int8_bars.patches, fp32_latencies, int8_latencies
    ):
        ax1.text(
            fp32_bar.get_x() + fp32_bar.get_width() / 2 - 0.03, fp32_height + max_latency * 0.02,
            f"{fp32_height:.3f}", ha="center", va="bottom", fontsize=6.3, color="#1b7837",
        )
        speedup: float = fp32_height / (int8_height if int8_height > 1e-5 else 1e-5)
        ax1.text(
            int8_bar.get_x() + int8_bar.get_width() / 2 + 0.04, int8_height + max_latency * 0.025,
            f"{speedup:.1f}x", ha="center", va="bottom", fontsize=6.3, color="#d62728",
        )

    val_accuracies: list[float] = [r.int8_val_accuracy * 100.0 for r in arch_results]
    touch_accuracies: list[float] = [r.int8_touch_accuracy * 100.0 for r in arch_results]

    val_bars = ax2.bar(
        x - grouped_width / 2, val_accuracies, grouped_width,
        label="MNIST Val", color="#9467bd", edgecolor="k", linewidth=0.5,
    )
    touch_bars = ax2.bar(
        x + grouped_width / 2, touch_accuracies, grouped_width,
        label="Touch Digits", color="#8c564b", edgecolor="k", linewidth=0.5,
    )
    ax2.set_ylim(0, 136)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=7.0)
    for val_bar, touch_bar, val_height, touch_height in zip(
        val_bars.patches, touch_bars.patches, val_accuracies, touch_accuracies
    ):
        ax2.text(
            val_bar.get_x() + val_bar.get_width() / 2 - 0.02, val_height + 2.5,
            f"{val_height:.1f}%", ha="center", va="bottom", fontsize=6.3, color="#542788",
        )
        ax2.text(
            touch_bar.get_x() + touch_bar.get_width() / 2 + 0.02, touch_height + 2.5,
            f"{touch_height:.0f}%", ha="center", va="bottom", fontsize=6.3, color="#8c564b",
        )

    panel_specs: list[tuple[Axes, str, str]] = [
        (ax0, "Trainable Parameters", "(a) Parameters & Storage"),
        (ax1, "Host CPU Latency (ms/img)", "(b) Latency & Speedup"),
        (ax2, "INT8 Accuracy (%)", "(c) Classification Accuracy"),
    ]
    for ax, ylabel, title in panel_specs:
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.55, len(arch_results) - 0.45)
        ax.tick_params(axis="y", labelsize=7.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.0)
        ax.set_title(title, fontsize=8.8, pad=6)
        ax.grid(True, linestyle="--", axis="y", alpha=0.5)

    plt.tight_layout()
    save_figure("fig3_architectures_comparison.png")


def plot_figure4_predictions(arch_results: list[ArchitectureResult], touch_x: np.ndarray) -> None:
    num_digits = len(touch_x)
    fig, axes = plt.subplots(
        len(arch_results), num_digits, figsize=(7.2, 2.75), gridspec_kw={"hspace": 0.42, "wspace": 0.12}
    )

    for row_idx, result in enumerate(arch_results):
        for digit in range(num_digits):
            ax: Axes = axes[row_idx, digit]
            ax.imshow(touch_x[digit, 0, :, :], cmap="gray", interpolation="nearest")
            pred = int(result.touch_predictions[digit])
            is_correct = pred == digit
            color = "#1b7837" if is_correct else "#d62728"
            status = "OK" if is_correct else "ERR"
            ax.set_title(f"T:{digit} P:{pred}\n{status}", color=color, fontsize=7.6, fontweight="bold", pad=2)
            ax.axis("off")

    plt.subplots_adjust(left=0.16, right=0.99, top=0.88, bottom=0.04)
    for row_idx, result in enumerate(arch_results):
        acc_pct = int(round(result.int8_touch_accuracy * 100.0))
        row_ax: Axes = axes[row_idx, 0]
        bbox = row_ax.get_position()
        row_center_y = 0.5 * (bbox.ymin + bbox.ymax)
        fig.text(0.012, row_center_y, f"{result.name}\n({acc_pct}% Acc)",
                 va="center", ha="left", fontsize=8.2, fontweight="bold")

    save_figure("fig4_confusion_matrices.png")


def main() -> None:
    (x_train, y_train), (x_test, y_test) = load_mnist()
    touch_x, touch_y, raw_matrices = load_touch_samples()

    experiments = [
        ExperimentConfig(
            name="exp1_initial",
            table_label="Baseline (SGD, $\\eta=0.05$)",
            optimizer="sgd",
            lr=0.05,
            epochs=10,
        ),
        ExperimentConfig(name="exp2_adam", table_label="Adam optimizer ($\\eta=0.001$)"),
        ExperimentConfig(name="exp3_lr_0005", table_label="LR tuning ($\\eta=0.0005$)", lr=0.0005),
        ExperimentConfig(name="exp4_ep12", table_label="Extended training (12 ep)", epochs=12),
        ExperimentConfig(name="exp5_bs32", table_label="Batch size reduction ($B=32$)", epochs=6, batch_size=32),
        ExperimentConfig(
            name="exp6_leaky",
            table_label="LeakyReLU activation",
            builder=lambda: build_baseline_cnn(activation="leaky_relu"),
            epochs=6,
        ),
        ExperimentConfig(
            name="exp7_dropout",
            table_label="Dropout regularization",
            builder=lambda: build_baseline_cnn(dropout=0.25),
            epochs=6,
        ),
        ExperimentConfig(
            name="exp8_pruned_width",
            table_label="Channel pruning (filters 4, 8)",
            builder=lambda: build_baseline_cnn(num_filters1=4, num_filters2=8, dense_units=16),
        ),
    ]

    experiment_results: list[ExperimentResult] = []
    exp8_model: nn.Module | None = None

    for exp in experiments:
        checkpoint_path = os.path.join(MODELS_DIR, f"{exp.name}.pth")
        model = load_or_train(
            exp.builder(), checkpoint_path, exp.name, x_train, y_train,
            epochs=exp.epochs, batch_size=exp.batch_size, lr=exp.lr, optimizer_name=exp.optimizer,
        )
        if exp.name == "exp8_pruned_width":
            exp8_model = model
        fp32_bench = benchmark_fp32_model(model, x_test, y_test, touch_x, touch_y, checkpoint_path)
        experiment_results.append(ExperimentResult(
            table_label=exp.table_label,
            memory_tex=f"{fp32_bench.num_params:,} ({fp32_bench.size_kb:.1f} KB)",
            avg_speed_ms=fp32_bench.avg_speed_ms,
            size_kb=fp32_bench.size_kb,
            val_accuracy=fp32_bench.val_accuracy,
            touch_accuracy=fp32_bench.touch_accuracy,
        ))

    if exp8_model is None:
        raise RuntimeError("exp8_pruned_width model was not created")

    exp9_path = os.path.join(MODELS_DIR, "exp9_l1_pruned.pth")
    pruned_model = load_or_train(
        copy.deepcopy(exp8_model), exp9_path, "exp9_l1_pruned", x_train, y_train,
        epochs=3, batch_size=64, lr=0.0005, l1_prune_amount=0.5,
    )
    exp9_bench = benchmark_fp32_model(pruned_model, x_test, y_test, touch_x, touch_y, exp9_path)
    experiment_results.append(ExperimentResult(
        table_label="L1 weight pruning (50\\%)",
        memory_tex=f"{exp9_bench.num_nonzero_params:,} ({exp9_bench.size_kb:.1f} KB)",
        avg_speed_ms=exp9_bench.avg_speed_ms,
        size_kb=exp9_bench.size_kb,
        val_accuracy=exp9_bench.val_accuracy,
        touch_accuracy=exp9_bench.touch_accuracy,
    ))

    exp10_path = os.path.join(MODELS_DIR, "exp10_pruned_int8.tflite")
    quantize_model_to_int8(pruned_model, x_train, exp10_path)
    exp10_bench = benchmark_int8_model(exp10_path, x_test, y_test, touch_x, touch_y)
    experiment_results.append(ExperimentResult(
        table_label="INT8 PTQ quantization",
        memory_tex=f"INT8 ({exp10_bench.size_kb:.1f} KB)",
        avg_speed_ms=exp10_bench.avg_speed_ms,
        size_kb=exp10_bench.size_kb,
        val_accuracy=exp10_bench.val_accuracy,
        touch_accuracy=exp10_bench.touch_accuracy,
    ))

    table1_rows = [
        f"{idx}) {row.table_label} & {row.avg_speed_ms:.3f} ms & {row.memory_tex}"
        f" & {row.val_accuracy * 100.0:.2f}\\% & {int(round(row.touch_accuracy * 100.0))}\\% \\\\"
        for idx, row in enumerate(experiment_results, 1)
    ]
    write_latex_table(
        filename="table1_optimizations.tex",
        comment="Table I: Optimization and Quantization Progression",
        tabcolsep="1.4pt",
        caption="Optimization and Quantization Progression",
        label="tab:optimizations",
        headers=["Optimization Method", "Avg. Speed*", "Memory", "Val. Acc.", "Touch"],
        rows=table1_rows,
        footnote=(
            "*10-digit prediction time divided by 10 on single-thread host CPU. "
            f"Row 9 keeps {exp9_bench.num_nonzero_params:,} non-zero parameters "
            f"out of {exp9_bench.num_params:,} total."
        ),
    )

    architectures = [
        ArchitectureConfig(id="arch1_baseline", checkpoint="exp2_adam.pth", name="Baseline CNN"),
        ArchitectureConfig(
            id="arch2_mobilenet",
            checkpoint="arch2_mobilenet.pth",
            name="MobileNet",
            builder=MobileNet,
            lr=0.002,
        ),
        ArchitectureConfig(
            id="arch3_squeezenet",
            checkpoint="arch3_squeezenet.pth",
            name="SqueezeNet",
            builder=SqueezeNet,
            epochs=6,
            batch_size=128,
        ),
    ]

    arch_results: list[ArchitectureResult] = []
    for arch in architectures:
        checkpoint_path = os.path.join(MODELS_DIR, arch.checkpoint)
        model = load_or_train(
            arch.builder(), checkpoint_path, arch.name, x_train, y_train,
            epochs=arch.epochs, batch_size=arch.batch_size, lr=arch.lr,
        )
        fp32_bench = benchmark_fp32_model(model, x_test, y_test, touch_x, touch_y, checkpoint_path)

        tflite_path = os.path.join(MODELS_DIR, f"{arch.id}_int8.tflite")
        quantize_model_to_int8(model, x_train, tflite_path)
        int8_bench = benchmark_int8_model(tflite_path, x_test, y_test, touch_x, touch_y)

        arch_results.append(ArchitectureResult(
            name=arch.name,
            num_params=f"{fp32_bench.num_params:,}",
            fp32_speed_ms=fp32_bench.avg_speed_ms,
            int8_size_kb=int8_bench.size_kb,
            int8_speed_ms=int8_bench.avg_speed_ms,
            int8_val_accuracy=int8_bench.val_accuracy,
            int8_touch_accuracy=int8_bench.touch_accuracy,
            touch_predictions=int8_bench.touch_predictions,
        ))

    table2_rows = [
        f"{row.name} & {row.num_params} & {row.int8_size_kb:.1f} KB"
        f" & {row.int8_speed_ms:.3f} ms & {row.int8_val_accuracy * 100.0:.2f}\\%"
        f" & {int(round(row.int8_touch_accuracy * 100.0))}\\% \\\\"
        for row in arch_results
    ]
    write_latex_table(
        filename="table2_architectures.tex",
        comment="Table II: Architectural Benchmark (Host CPU)",
        tabcolsep="2.8pt",
        caption="Multi-Architecture INT8 Benchmark on Host CPU",
        label="tab:architectures",
        headers=["Architecture", "Params", "INT8 Size", "Latency*", "Val. Acc.", "Touch"],
        rows=table2_rows,
        footnote="*10-digit prediction time divided by 10 on single-thread host CPU (INT8 LiteRT).",
    )

    plot_figure1_digits(touch_x, raw_matrices)
    plot_figure2_pareto(experiment_results)
    plot_figure3_architectures(arch_results)
    plot_figure4_predictions(arch_results, touch_x)

    print("Done. Saved outputs to tables/ and figures/.", flush=True)


if __name__ == "__main__":
    main()
