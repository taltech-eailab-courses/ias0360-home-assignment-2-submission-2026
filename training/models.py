"""Model architectures for IMU gesture classification.

Input: (128, 3, 1) = 128 time frames x 3 acceleration axes x 1 channel.
The ops used are registered in firmware/gesture_inference/src/model_runner.cpp.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

INPUT_SHAPE = (128, 3, 1)


def lecture_cnn(num_classes: int, num_filters1=8, num_filters2=16, dense_units=16,
                activation="relu", dropout=0.1, extra_conv_block=False,
                input_shape=INPUT_SHAPE, batch_size=None) -> tf.keras.Model:
    """Architecture from lecture 08 slide 64 / lab 2.2 (Magic Wand CNN).

    Conv2D(4x3) -> MaxPool(3x3) -> Conv2D(4x1) -> MaxPool(3x1) -> Flatten
    -> Dropout -> Dense -> Dropout -> Dense(softmax)
    ``extra_conv_block`` inserts one more Conv2D(4x1)+MaxPool(2x1) stage.
    """
    model = tf.keras.Sequential(name="lecture_cnn")
    model.add(layers.Input(shape=input_shape, batch_size=batch_size))
    model.add(layers.Conv2D(num_filters1, (4, 3), padding="same", activation=activation))
    model.add(layers.MaxPooling2D((3, 3), strides=(3, 3), padding="same"))
    model.add(layers.Conv2D(num_filters2, (4, 1), padding="same", activation=activation))
    model.add(layers.MaxPooling2D((3, 1), strides=(3, 1), padding="same"))
    if extra_conv_block:
        model.add(layers.Conv2D(num_filters2 * 2, (4, 1), padding="same", activation=activation))
        model.add(layers.MaxPooling2D((2, 1), strides=(2, 1), padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units, activation=activation))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def _ds_block(x, filters, stride, name):
    """MobileNet depthwise-separable block: DW(5x1) -> BN -> ReLU6 -> PW(1x1) -> BN -> ReLU6."""
    # Width is 1 here, so the square stride only downsamples time.
    x = layers.DepthwiseConv2D((5, 1), strides=(stride, stride), padding="same", use_bias=False,
                               name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(6.0, name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(filters, (1, 1), use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    return layers.ReLU(6.0, name=f"{name}_pw_relu")(x)


def mobilenet_ds_cnn(num_classes: int, width=16, dropout=0.2,
                     input_shape=INPUT_SHAPE, batch_size=None) -> tf.keras.Model:
    """MobileNet-style network: Conv2D(5x3) stem fusing the 3 axes,
    depthwise-separable blocks along time, global average pooling."""
    inputs = layers.Input(shape=input_shape, batch_size=batch_size)
    x = layers.Conv2D(width, (5, 3), strides=(2, 1), padding="valid", use_bias=False,
                      name="stem")(inputs)                      # -> (62, 1, width)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(6.0, name="stem_relu")(x)
    x = _ds_block(x, width * 2, 2, "b1")                        # 31
    x = _ds_block(x, width * 2, 1, "b2")
    x = _ds_block(x, width * 4, 2, "b3")                        # 16
    x = _ds_block(x, width * 4, 2, "b4")                        # 8
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenet_ds_cnn")


def _fire(x, squeeze, expand, name):
    """SqueezeNet Fire module: 1x1 squeeze -> concat(1x1 expand, 3x1 expand)."""
    s = layers.Conv2D(squeeze, (1, 1), activation="relu", name=f"{name}_squeeze")(x)
    e1 = layers.Conv2D(expand, (1, 1), activation="relu", name=f"{name}_e1x1")(s)
    e3 = layers.Conv2D(expand, (3, 1), padding="same", activation="relu", name=f"{name}_e3x1")(s)
    return layers.Concatenate(axis=-1, name=f"{name}_concat")([e1, e3])


def squeezenet_cnn(num_classes: int, width=16, dropout=0.2,
                   input_shape=INPUT_SHAPE, batch_size=None) -> tf.keras.Model:
    """SqueezeNet-style 1-D network: conv stem, Fire modules, conv classifier + GAP."""
    inputs = layers.Input(shape=input_shape, batch_size=batch_size)
    x = layers.Conv2D(width * 2, (5, 3), strides=(2, 1), padding="valid", activation="relu",
                      name="stem")(inputs)                      # -> (62, 1, 2w)
    x = layers.MaxPooling2D((3, 1), strides=(2, 1), padding="same", name="pool1")(x)   # 31
    x = _fire(x, width // 2, width, "fire2")
    x = _fire(x, width // 2, width, "fire3")
    x = layers.MaxPooling2D((3, 1), strides=(2, 1), padding="same", name="pool3")(x)   # 16
    x = _fire(x, width, width * 2, "fire4")
    x = _fire(x, width, width * 2, "fire5")
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(num_classes, (1, 1), name="conv_classifier")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Softmax(name="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="squeezenet_cnn")


ARCHITECTURES = {
    "lecture_cnn": lecture_cnn,
    "mobilenet_ds_cnn": mobilenet_ds_cnn,
    "squeezenet_cnn": squeezenet_cnn,
}


def build(arch: str, num_classes: int, **kwargs) -> tf.keras.Model:
    return ARCHITECTURES[arch](num_classes, **kwargs)
