import tensorflow as tf

def build_model(input_shape, num_classes, num_filters1=8, num_filters2=16,
                 dense_units=16, activation="relu", dropout=0.1, l2_lambda=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(num_filters1, kernel_size=(4, 3), padding="same", activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        # TODO: MaxPooling2D — pool_size=(3, 3), strides=(3, 3), padding="same"
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3), padding="same"),
        # TODO: Conv2D — num_filters2, kernel_size=(4, 1), padding="same" addition of kernel_regularizer
        tf.keras.layers.Conv2D(num_filters2, kernel_size=(4,1), padding="same", activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        # TODO: MaxPooling2D — pool_size=(3, 1), strides=(3, 1), padding="same"
        tf.keras.layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), padding="same"),
        # TODO: Flatten was replaced by GlobalAveragePooling to increase accuracy, but it cut down the trainable parameters together with the accuracy.
        tf.keras.layers.Flatten(),
        # TODO: Dropout — rate=dropout
        tf.keras.layers.Dropout(rate=dropout),
        # TODO: Dense — dense_units, activation=activation
        tf.keras.layers.Dense(dense_units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        # TODO: Dropout — rate=dropout
        tf.keras.layers.Dropout(rate=dropout),
        # TODO: final Dense — num_classes, activation="softmax" (why sof back to what softmax outputs)
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model



