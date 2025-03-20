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
        # layers.append(squeeze())

        self.head_net = Sequential(layers)

    def call(self, inputs):
        y = self.head_net(inputs)
        return y

class multiple_head(Layer):
    def __init__(self, n_units: int, hidden_units = [], bias: Optional[int] = None):
        super().__init__()
        self.n_units = n_units

        layers = []
        layers.append(Flatten())
        for hidden_unit in hidden_units:
            layers.append(Dense(hidden_unit, kernel_initializer=initializers.he_normal()))
            layers.append(ReLU())

        layers.append(Dense(1, activation='sigmoid',
                            bias_initializer=bias,
                            kernel_initializer=initializers.glorot_normal()))
        layers.append(squeeze())

        self.head_nets = [Sequential(layers) for _ in range(n_units)]

    def call(self, inputs):
        y = []

        for unit in range(self.n_units):
            single_y = self.head_nets[unit](inputs)
            y.append(single_y)

        y = Concatenate()(y)
        return y

class temporal_head(Layer):
    def __init__(self, n_units: int, hidden_units = [], bias: Optional[int] = None, length: Optional[int] = None):
        super().__init__()
        self.n_units = n_units

        layers = []
        for hidden_unit in hidden_units:
            layers.append(
                TimeDistributed(
                    Dense(hidden_unit, kernel_initializer=initializers.he_normal(), activation='relu')
                )
            )

        layers.append(
            TimeDistributed(
                Dense(n_units, activation='sigmoid', bias_initializer=bias),
            )
        )

        layers.append(squeeze())

        self.head_net = Sequential(layers)

    def call(self, inputs):
        y = self.head_net(inputs)
        return y