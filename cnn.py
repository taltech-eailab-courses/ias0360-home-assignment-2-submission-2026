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

def load_my_digits() -> tuple[np.ndarray, np.ndarray]:
    """
        Load the custom digits dataset.
    """
    images = []
    labels = []

    for label in range(10):
        img_path = Path("my_digits") / f"digit_{label}.txt"

        image_rows = []
        with img_path.open("r") as file:
            for line in file:
                row = []
                for character in line.strip():
                    row.append(float(character))
                image_rows.append(row)

        image = np.array(image_rows, dtype=np.float32)
        images.append(image)
        labels.append(label)

    x = np.array(images, dtype=np.float32)[..., np.newaxis]
    y = np.array(labels, dtype=np.int64)

    return x, y

def plot_training_curve(model: tf_keras.Model, history: tf_keras.callbacks.History, test_loss: float, test_accuracy: float) -> None:
    """
        Plot the training curve and evaluate the model.
    """

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
    x_my_test, y_my_test = load_my_digits()

    model = build_model(acctivation="tanh")

    print("\n--------------")
    print("MODEL SUMMARY:")
    print("--------------")
    model.summary()

    optimizer=tf_keras.optimizers.Adam(learning_rate=0.001)
    # optimizer=tf_keras.optimizers.SGD(learning_rate=0.001)
    # optimizer=tf_keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n---------------")
    print("MODEL TRAINING:")
    print("---------------")
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=2,
    )

    # plot_training_curve(model, history, test_loss, test_accuracy)

    print("\n-----------------")
    print("MODEL EVALUATION:")
    print("-----------------")
    
    mnist_test_loss, mnist_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {mnist_test_loss:.4f}")
    print(f"Test accuracy: {mnist_test_accuracy:.4f}")

    my_test_loss, my_test_accuracy = model.evaluate(x_my_test, y_my_test, verbose=0)
    print(f"Custom digits test loss: {my_test_loss:.4f}")
    print(f"Custom digits test accuracy: {my_test_accuracy:.4f}")



    print("\n-----------------")
    print("MODEL PREDICTION:")
    print("-----------------")

    start = time.perf_counter()
    predictions = model.predict(x_my_test, verbose=0)
    elapsed = time.perf_counter() - start
    average_time_ms = elapsed / len(x_my_test) * 1000

    # print(f"Predictions: {predictions}")    
    print(f"Average prediction time: {average_time_ms:.2f} ms")


if __name__ == "__main__":
    main()