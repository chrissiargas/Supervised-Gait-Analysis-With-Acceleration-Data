import tensorflow as tf
from keras import Layer
from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten,
                          LayerNormalization, Layer,
                          BatchNormalization, Activation, Concatenate, )
from keras import Sequential
from keras.src.ops import Squeeze
from tensorflow.python.framework.test_ops import kernel_label
from keras import initializers
from tensorflow_graphics.geometry.transformation import quaternion, rotation_matrix_3d
import math

from preprocessing.building import activity

class get_mean(Layer):
    def call(self, x):
        return tf.math.reduce_mean(x, axis=1)


class feature_extractor(Layer):
    def __init__(self):
        super(feature_extractor, self).__init__()

    def __call__(self, inputs):
        mean = get_mean()(inputs)
        var = tf.math.reduce_variance(inputs, axis=1)
        fft_input = tf.transpose(inputs, perm=[0, 2, 1])
        fft = tf.signal.fft(tf.cast(fft_input, tf.complex64))
        fft_mag = tf.abs(fft)
        fft_mean = tf.reduce_mean(fft_mag, axis=2)
        fft_max = tf.reduce_max(fft_mag, axis=2)

        features = tf.concat([
            mean,
            var,
            fft_mean,
            fft_max,
        ], axis=-1)

        return features

class rotateByAxis(Layer):
    def __init__(self, input_shape, by: str = 'raw'):
        super().__init__()
        self.input_shape = input_shape
        self.length = self.input_shape[0]
        self.channels = self.input_shape[1]
        self.pi = tf.constant(math.pi)

        if by == 'raw':
            self.extract_features = Sequential([
                Conv1D(filters=32,
                       kernel_size=3,
                       strides=1,
                       padding="same",
                       activation="relu",
                       kernel_initializer=initializers.he_normal()),
                MaxPooling1D(pool_size=2, strides=2),
                Conv1D(filters=32,
                       kernel_size=3,
                       strides=1,
                       padding="same",
                       activation="relu",
                       kernel_initializer=initializers.he_normal()),
                MaxPooling1D(pool_size=2, strides=2),
                Conv1D(filters=32,
                       kernel_size=3,
                       strides=1,
                       padding="same",
                       activation="relu",
                       kernel_initializer=initializers.he_normal()),
                MaxPooling1D(pool_size=2, strides=2)
            ])

            self.regression_net = Sequential([
                Flatten(),
                Dense(64, activation='relu', kernel_initializer=initializers.he_normal()),
                Dense(32, activation='relu', kernel_initializer=initializers.he_normal()),
                Dense(1, activation='tanh', kernel_initializer=initializers.glorot_normal())
            ])

        if by == 'features':
            self.extract_features = feature_extractor()

            self.regression_net = Sequential([
                Flatten(),
                Dense(64, activation='relu', kernel_initializer=initializers.he_normal()),
                Dense(32, activation='relu', kernel_initializer=initializers.he_normal()),
                Dense(1, activation='tanh', kernel_initializer=initializers.glorot_normal())
            ])


    def call(self, inputs):
        x = inputs
        features = self.extract_features(x)
        angle = self.regression_net(features)
        return tf.math.multiply(angle, self.pi / 2.)









