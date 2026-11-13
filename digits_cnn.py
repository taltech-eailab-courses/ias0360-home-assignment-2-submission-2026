from pathlib import Path
import os
import time
from xml.parsers.expat import model
import numpy as np
import tensorflow as tf
import tf_keras
import matplotlib.pyplot as plt


tf.random.set_seed(10)
np.random.seed(10)

# Hides TensorFlow INFO messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")



def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Load MNIST and prepare tensors for Conv2D.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[..., np.newaxis]
    x_test = (x_test / 255.0).astype("float32")[..., np.newaxis]
    return x_train, y_train, x_test, y_test


def build_model(num_filters1: int = 8, num_filters2: int = 16, dense_units: int = 32, activation: str = "relu", dropout: float = 0.0) -> tf_keras.Model:
    """
        Build the compact CNN used as the assignment baseline.
    """
    layers = [

        # INPUT LAYER
        tf_keras.layers.InputLayer(input_shape=(28, 28, 1)),
        # 28*28*1

        # FIRST LAYER
        tf_keras.layers.Conv2D(num_filters1, (3, 3), activation=activation),
        # 26*26*8
        tf_keras.layers.MaxPooling2D((2, 2)),
        # 13*13*8

        # SECOND LAYER
        tf_keras.layers.Conv2D(num_filters2, (3, 3), activation=activation),
        # 11*11*16
        tf_keras.layers.MaxPooling2D((2, 2)),
        # 5*5*16
        tf_keras.layers.Flatten(),
        # 400
        
        # THIRD LAYER
        tf_keras.layers.Dense(dense_units, activation=activation),
        # 32
    ]
    if dropout > 0:
        layers.append(tf_keras.layers.Dropout(dropout))
    # OUTPUT LAYER
    layers.append(tf_keras.layers.Dense(10, activation="softmax"))
    return tf_keras.models.Sequential(layers, name="mnist_baseline_cnn")


def plot_training_curve(model: tf_keras.Model, history: tf_keras.callbacks.History, test_loss: float, test_accuracy: float) -> None:
    """
        Plot the training curve and evaluate the model.
    """
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Trainable parameters: {model.count_params()}")

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Day 1 MNIST baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/baseline_training_curve.png", dpi=110)
    plt.close()

    model.save("results/mnist_cnn.h5")
    model.export("results/mnist_cnn_savedmodel")
    print("Saved outputs in:", "results/")


def main() -> None:
    
    x_train, y_train, x_test, y_test = load_mnist()

    model = build_model()
    model.summary()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    plot_training_curve(model, history, test_loss, test_accuracy)

    


if __name__ == "__main__":
    main()