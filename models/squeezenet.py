import tf_keras





def fire_module(x, squeeze_channels: int = 8, expand_channels: int = 16, activation: str = "relu"):
    """
        Helper to build a SqueezeNet Fire Module
    """

    # Squeeze layer
    squeeze = tf_keras.layers.Conv2D(squeeze_channels, (1, 1), activation=activation, padding="same")(x)
    
    # Expand layer
    expand_1x1 = tf_keras.layers.Conv2D(expand_channels, (1, 1), activation=activation, padding="same")(squeeze)
    expand_3x3 = tf_keras.layers.Conv2D(expand_channels, (3, 3), activation=activation, padding="same")(squeeze)
    
    
    return tf_keras.layers.Concatenate()([expand_1x1, expand_3x3])


def build_squeezenet(activation: str = "relu", dense_units: int = 64) -> tf_keras.Model:
    """
        Build the SqueezeNet network.
    """
    
    inputs = tf_keras.layers.Input(shape=(28, 28, 1))

    
    # Initial light stem (keep spatial dimensions safe for 28x28)
    x = tf_keras.layers.Conv2D(16, (3, 3), strides=1, activation=activation, padding="same")(inputs) # 28x28x16
    x = tf_keras.layers.MaxPooling2D((2, 2))(x) # -> 14x14x16
    
    # Fire Modules (SqueezeNet core)
    x = fire_module(x, squeeze_channels=8, expand_channels=16, activation=activation) # 14x14x32
    x = fire_module(x, squeeze_channels=8, expand_channels=16, activation=activation) # 14x14x32
    
    x = tf_keras.layers.MaxPooling2D((2, 2))(x) # -> 7x7x32
    
    # Classification head
    x = tf_keras.layers.Flatten()(x) # 1x1x1568
    x = tf_keras.layers.Dense(dense_units, activation=activation)(x) # 1x1x64
    outputs = tf_keras.layers.Dense(10, activation="softmax")(x) # 1x1x10
    
    return tf_keras.Model(inputs, outputs, name="squeezenet")