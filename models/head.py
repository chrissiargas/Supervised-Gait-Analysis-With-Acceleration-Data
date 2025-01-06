import tensorflow as tf
from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten, LayerNormalization, Layer,
                          BatchNormalization, Activation)
from keras.models import Model

def attach_head(encoder: Model, n_units: int, hidden_layers = [], name: str = 'full_model') -> Model:
    input = encoder.input
    x = encoder.output

    for i, hidden in enumerate(hidden_layers):
        x = Dense(hidden)(x)
        x = Activation('relu')(x)

    x = Dense(n_units)(x)
    outputs = Activation(activation='sigmoid')(x)

    full_model = Model(input, outputs, name = name)

    return full_model