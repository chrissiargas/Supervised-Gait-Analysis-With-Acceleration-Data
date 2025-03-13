import tensorflow as tf
from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten,
                          LayerNormalization, Layer,
                          BatchNormalization, Activation, Concatenate,)
from keras.models import Model
from keras import initializers
from typing import Optional
from keras import Sequential

from keras.src.layers import TimeDistributed

class squeeze(Layer):
    def call(self, x):
        return tf.squeeze(x, axis=-1)

class single_head(Layer):
    def __init__(self,  n_units: int, hidden_units = [], bias: Optional[int] = None):
        super().__init__()

        layers = []

        layers.append(Flatten())
        for hidden_unit in hidden_units:
            layers.append(Dense(hidden_unit, kernel_initializer=initializers.he_normal()))
            layers.append(ReLU())

        layers.append(Dense(n_units,
                            activation='sigmoid',
                            bias_initializer=bias,
                            kernel_initializer=initializers.glorot_normal()))
        layers.append(squeeze())

        self.head_net = Sequential(layers)

    def call(self, inputs):
        y = self.head_net(inputs)
        return y

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

def attach_temporal_head(encoder: Model, n_units: int, hidden_layers = [], name: str = 'full_model', bias: Optional[int] = None) -> Model:
    input = encoder.input
    x = encoder.output

    for i, hidden in enumerate(hidden_layers):
        x = TimeDistributed(Dense(hidden, activation='relu'))(x)

    outputs = TimeDistributed(Dense(n_units, activation='sigmoid', bias_initializer=bias))(x)
    outputs = squeeze()(outputs)

    full_model = Model(input, outputs, name = name)

    return full_model