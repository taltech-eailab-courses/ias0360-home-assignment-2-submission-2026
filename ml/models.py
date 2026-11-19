"""Three simple architectures and configurable small CNN heads."""
import tensorflow as tf


def build_model(config):
    layers = tf.keras.layers
    model = tf.keras.Sequential(name=config["name"])
    model.add(layers.Input((28, 28, 1)))
    architecture = config["architecture"]
    if architecture == "mlp":
        model.add(layers.Flatten())
        for units in config["dense"]:
            model.add(layers.Dense(units, activation="relu"))
    elif architecture == "lenet_style":
        # Adapted to 28x28; this is not an exact reproduction of LeNet-5.
        model.add(layers.Conv2D(6, 5, padding="same", activation="tanh"))
        model.add(layers.AveragePooling2D(pool_size=2))
        model.add(layers.Conv2D(16, 5, activation="tanh"))
        model.add(layers.AveragePooling2D(pool_size=2))
        model.add(layers.Flatten())
        for units in config["dense"]:
            model.add(layers.Dense(units, activation="tanh"))
    elif architecture == "small_cnn":
        for filters in config["filters"]:
            model.add(layers.Conv2D(filters, 3, activation="relu"))
            model.add(layers.MaxPooling2D())
        if config["head"] == "global_average":
            model.add(layers.GlobalAveragePooling2D())
        else:
            model.add(layers.Flatten())
        model.add(layers.Dense(config["dense"], activation="relu"))
    else:
        raise ValueError(architecture)
    model.add(layers.Dense(10, activation="softmax"))
    return model
