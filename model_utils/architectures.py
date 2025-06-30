import tensorflow as tf

from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten, LayerNormalization, Layer,
                          BatchNormalization, Multiply, Bidirectional, Permute, Lambda, RepeatVector,
                          GRU, ZeroPadding2D, MaxPooling2D, MultiHeadAttention, Concatenate)
from keras.models import Model
from keras import initializers, Sequential
import keras.backend as K
from keras.src.layers import concatenate, Conv1DTranspose, UpSampling1D
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

class transformer_block(Layer):
    def __init__(self,
                 units: int = 64,
                 last_units: int = 64):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(num_heads=4, key_dim=16)
        self.layer_norm = LayerNormalization()
        self.first_dense = Dense(units=units, activation='relu')
        self.second_dense = Dense(units=last_units)

    def call(self, inputs):
        attn = self.multihead_attention(inputs, inputs)
        x = self.layer_norm(inputs + attn)
        ffn = self.first_dense(x)
        ffn = self.second_dense(ffn)
        return self.layer_norm(x + ffn)


def encoder_block(x, filters, kernel_size):
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = ReLU()(x)
    return MaxPooling1D(2)(x)

def decoder_block(x, skip, filters, kernel_size):
    x = UpSampling1D(2)(x)
    x = Conv1D(filters, 3, padding='same')(x)
    x = Concatenate()([x, skip])
    x = Conv1D(filters, kernel_size, padding='same')(x)
    return ReLU()(x)

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


class transformer_encoder(Layer):
    def __init__(self, n_units: int):
        super().__init__()

        self.dense = Dense(units=n_units)

        self.transformer_blocks = [
            transformer_block(
                units=n_units*2,
                last_units=n_units
            ) for _ in range(4)
        ]

    def call(self, inputs):
        x = inputs

        x = self.dense(x)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        return x

def unet_encoder(n_units):
        inputs = Input(shape=(128, 3))

        x = encoder_block(inputs, filters=n_units, kernel_size=5)
        x1 = x

        x = encoder_block(x, filters=n_units * 2, kernel_size=5)
        x2 = x

        x = encoder_block(x, filters=n_units * 4, kernel_size=5)
        x3 = x

        x = encoder_block(x, filters=n_units * 4, kernel_size=5)

        x = decoder_block(x, x3, filters=n_units * 4, kernel_size=5)

        x = decoder_block(x, x2, filters=n_units * 2, kernel_size=5)

        x = decoder_block(x, x1, filters=n_units, kernel_size=5)

        x = decoder_block(x, inputs, filters=n_units, kernel_size=5)

        return Model(inputs=inputs, outputs=x)




