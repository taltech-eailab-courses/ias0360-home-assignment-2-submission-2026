import tf_keras

def depth_point(x, filters, activation="relu"):
    x = tf_keras.layers.DepthwiseConv2D((3, 3), activation=activation, padding="same")(x)
    x = tf_keras.layers.BatchNormalization()(x)
    x = tf_keras.layers.Activation(activation=activation)(x)

    x = tf_keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x)
    x = tf_keras.layers.BatchNormalization()(x)
    x = tf_keras.layers.Activation(activation=activation)(x)

    return x

def build_mobilenet(activation: str = "relu", dense_units: int = 64) -> tf_keras.Model:
    """
        Build the MobileNet network.
    """
    inputs = tf_keras.layers.Input(shape=(28, 28, 1))

    # Initial light stem (keep spatial dimensions safe for 28x28)
    x = tf_keras.layers.Conv2D(16, (3, 3), strides=1, activation=activation, padding="same")(inputs) # 28x28x16
    x = tf_keras.layers.MaxPooling2D((2, 2))(x) # -> 14x14x16

    # Depthwise Separable Convolution Blocks
    x = depth_point(x, filters=32, activation=activation) # 14x14x32
    x = tf_keras.layers.MaxPooling2D((2, 2))(x) # -> 7x7x32
    x = depth_point(x, filters=64, activation=activation) # 7x7x64
    x = tf_keras.layers.MaxPooling2D((2, 2))(x) # -> 3x3x64
    
    # Classification head
    x = tf_keras.layers.Flatten()(x) # 1x1x576
    x = tf_keras.layers.Dense(dense_units, activation=activation)(x) # 1x1x64
    outputs = tf_keras.layers.Dense(10, activation="softmax")(x) # 1x1x10
    
    return tf_keras.Model(inputs, outputs, name="mobilenet")