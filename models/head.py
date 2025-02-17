import tensorflow as tf
from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten,
                          LayerNormalization, Layer,
                          BatchNormalization, Activation, Concatenate)
from keras.models import Model
from keras import initializers

def attach_single_head(encoder: Model, n_units: int, hidden_layers = [], name: str = 'full_model') -> Model:
    input = encoder.input
    x = encoder.output

    for i, hidden in enumerate(hidden_layers):
        x = Dense(hidden)(x)
        x = Activation('relu')(x)

    x = Dense(n_units, kernel_initializer = initializers.glorot_uniform())(x)
    outputs = Activation(activation='sigmoid')(x)
    full_model = Model(input, outputs, name = name)

    return full_model

def attach_double_head(encoder: Model, n_units: int, hidden_layers = [], name: str = 'full_model'):
    input = encoder.input
    encoding = encoder.output
    outputs = []

    for _ in range(n_units):
        X = encoding
        for i, hidden in enumerate(hidden_layers):
            X = Dense(hidden)(X)
            X = Activation('relu')(X)

        x = Dense(1)(X)
        output = Activation(activation='sigmoid')(x)
        outputs.append(output)

    outputs = Concatenate()(outputs)
    full_model = Model(input, outputs, name = name)

    return full_model