"""Three TensorFlow/tf_keras classifiers, all with 28x28x1 input and ten outputs."""
import tf_keras as keras


def build_model(architecture='custom_cnn',filters=(8,16),dense=32,dropout=0.0):
    layers=keras.layers
    model=keras.Sequential(name=architecture)
    model.add(layers.Input(shape=(28,28,1)))
    if architecture=='custom_cnn':
        # Student baseline derived from the lab diagram; widths are configurable.
        for width in filters:
            model.add(layers.Conv2D(width,3,activation='relu'))
            model.add(layers.MaxPooling2D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(dense,activation='relu'))
    elif architecture=='lenet':
        # LeNet-style adaptation for 28x28 input, not an exact historical replica.
        model.add(layers.Conv2D(6,5,padding='same',activation='tanh'))
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Conv2D(16,5,activation='tanh'))
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(120,activation='tanh'))
        model.add(layers.Dense(84,activation='tanh'))
    elif architecture=='mlp':
        model.add(layers.Flatten())
        model.add(layers.Dense(128,activation='relu'))
        model.add(layers.Dense(64,activation='relu'))
    else:raise ValueError('Unknown architecture: '+architecture)
    if dropout:model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10,activation='softmax'))
    return model
