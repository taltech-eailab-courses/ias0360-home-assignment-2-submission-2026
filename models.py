try:
    import tf_keras as keras          # what the lab 2.2 notebook used
except ImportError:
    from tensorflow import keras
layers = keras.layers


def baseline_cnn(f1=8, f2=16, dense=32, dropout=0.0, gap=False):
    """gap=True replaces Flatten with GlobalAveragePooling (far fewer dense params)."""
    m = keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(f1, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(f2, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D() if gap else layers.Flatten(),
        layers.Dense(dense, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax"),
    ])
    return m


def mobilenet_style(f=16, dropout=0.0):
    """Depthwise-separable convs, the core idea of MobileNet, sized for 28x28."""
    inp = layers.Input((28, 28, 1))
    x = layers.Conv2D(f, 3, activation="relu")(inp)
    x = layers.SeparableConv2D(2 * f, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(4 * f, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out)


def _fire(x, s, e):
    sq = layers.Conv2D(s, 1, activation="relu")(x)
    e1 = layers.Conv2D(e, 1, activation="relu")(sq)
    e3 = layers.Conv2D(e, 3, padding="same", activation="relu")(sq)
    return layers.concatenate([e1, e3])


def squeezenet_style(s=8, e=16):
    """Small SqueezeNet: Fire modules (1x1 squeeze -> 1x1 + 3x3 expand)."""
    inp = layers.Input((28, 28, 1))
    x = layers.Conv2D(16, 3, activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = _fire(x, s, e)
    x = layers.MaxPooling2D()(x)
    x = _fire(x, s, e)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out)