# MNIST CNN Optimization — Homework 2

**IAS0360 · Embedded Machine Learning · TalTech**

This project explores handwritten digit classification with convolutional neural networks (CNNs), focusing on **reducing model memory usage while maintaining good accuracy**.

## Project tasks

- Train a baseline CNN on the MNIST dataset.
- Test different hyperparameters: number of filters, convolutional and dense layers, neurons, epochs, batch size, optimizer, and dropout.
- Evaluate models on MNIST and **10 personally handwritten digits**.
- Compare three architectures: **Baseline CNN**, **Mini-SqueezeNet**, and **Mini-MobileNet**.
- Compare accuracy, parameter count, estimated weight memory, and inference time.

## Requirements

- Python 3
- TensorFlow and `tf-keras`
- NumPy, pandas, Matplotlib, Pillow, Jupyter Notebook

Install dependencies:

```bash
pip install tensorflow tf-keras numpy pandas matplotlib pillow notebook
```

## How to run

1. Open `mnist_cnn(2).ipynb` in Jupyter Notebook.
2. Make sure your 10 handwritten digit images are available at the paths expected by the notebook.
3. Run the cells in order to load data, train models, and compare results.

```bash
jupyter notebook
```

## Models

| Architecture | Description |
|---|---|
| Baseline CNN | Two convolutional layers followed by a dense classifier |
| Mini-SqueezeNet | Lightweight CNN with a Fire module |
| Mini-MobileNet | CNN using depthwise separable convolutions |

The latter two are simplified, MNIST-specific adaptations of the original architectures.

## Evaluation

The notebook reports MNIST accuracy, accuracy on the 10 handwritten digits, average inference time, and estimated float32 weight memory.

This estimate covers model weights, not total runtime memory. Results can vary between training runs.

## Goal

Identify a compact CNN architecture with a good trade-off between **memory usage, inference time, and classification accuracy**.
