import tf_keras

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
    return tf_keras.models.Sequential(layers, name="cnn")