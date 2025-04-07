import tensorflow as tf

from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten, LayerNormalization, Layer,
                          BatchNormalization, Multiply, Bidirectional, Permute, Lambda, RepeatVector,
                          GRU, ZeroPadding2D, MaxPooling2D)
from keras.models import Model
from keras import initializers, Sequential
import keras.backend as K
from keras.src.layers import concatenate
from tcn import TCN
import keras
from typing import Optional
from pre_processing.utils import impute

class conv1d_block(Layer):
    def __init__(self,
                 use_dropout: bool = False,
                 kernel_size: int = 3,
                 filters: int = 64,
                 use_pooling: bool = True,
                 use_norm: bool = False):
        super().__init__()

        layers = []

        if use_norm:
            layers.append(BatchNormalization())
        layers.append(ZeroPadding1D(padding=1))
        layers.append(Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             strides=1,
                             padding='valid',
                             kernel_initializer=initializers.he_normal()
        ))
        layers.append(ReLU())
        if use_dropout:
            layers.append(Dropout(0.4))
        if use_pooling:
            layers.append(MaxPooling1D(pool_size=2, strides=2))

        self.conv_net = Sequential(layers)

    def call(self, inputs):
        y = self.conv_net(inputs)
        return y


class CNNGRU_encoder(Layer):
    def __init__(self,  n_units: int):
        super().__init__()

        self.conv1d_blocks = [conv1d_block(
            filters=n_units,
            kernel_size=3,
            use_pooling=False,
            use_norm=True,
            use_dropout=False
        ) for _ in range(3)]

        self.grus = [
            Bidirectional(GRU(n_units//2,
                              activation='tanh',
                              return_sequences=True))
            for _ in range(2)
        ]

    def call(self, inputs):
        x = inputs

        if self.conv1d_blocks:
            for conv1d in self.conv1d_blocks:
                x = conv1d(x)

        if self.grus:
            for gru in self.grus:
                x = gru(x)

        return x
