import numpy as np
from config_parser import Parser
from preprocessing.augmentations import add_noise, random_scale, time_mask, random_rotate
import tensorflow as tf

class specter:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

        self.height = self.conf.nperseg // 2
        self.width = self.conf.length // self.conf.nstride
        self.channels = len(self.conf.features)

        self.type = np.float32

    def get_shape(self):
        return self.height, self.width, 3

    def get_type(self):
        return self.type

    def __call__(self, spectrogram: np.ndarray, augment: bool = False) -> tf.Tensor:
        spectrogram = spectrogram.astype(self.type)
        output = spectrogram.transpose((1, 2, 0))
        output = tf.convert_to_tensor(output, dtype=tf.float32)
        return output

class fourier:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

        self.initial_features = {'acc_x': 0, 'acc_y': 1, 'acc_z': 2}
        self.initial_features.update({k: v+3 for v, k in enumerate(self.conf.new_features)})

        self.length = self.conf.length
        self.n_features = len(self.conf.features)
        self.type = np.float32

    def get_shape(self):
        return self.length, self.n_features

    def get_type(self):
        return self.type

    def __call__(self, window: np.ndarray, augment: bool = False):
        window = window.astype(self.type)
        output = None

        for feature in self.conf.features:
            f_idx = self.initial_features[feature]
            signal = window[:, f_idx]

            complex_fft = np.fft.fft(signal)
            power_fft = np.abs(complex_fft)
            power_fft[0] = 0.

            centered_power_fft = np.fft.fftshift(power_fft)

            if output is None:
                output = centered_power_fft[:, np.newaxis]

            else:
                output = np.concatenate(
                    (output, centered_power_fft[:, np.newaxis]),
                    axis=1
                )

        output = tf.convert_to_tensor(output, dtype = tf.float32)

        return output

from typing import Dict
class transformer:
    def __init__(self, channels: Dict):
        config = Parser()
        config.get_args()
        self.conf = config

        self.channels = channels
        self.available_augmentations =  ['jitter', 'scale', 'mask', 'rotate']
        self.augmentations = self.conf.augmentations

        if self.augmentations is None:
            self.augmentations = []
        else:
            assert all(augment in self.available_augmentations for augment in self.augmentations)

        self.length = self.conf.length
        self.n_features = len(self.conf.features)
        self.type = np.float32

        self.x = self.channels['acc_x']
        self.y = self.channels['acc_y']
        self.z = self.channels['acc_z']

    def get_shape(self):
        return self.length, self.n_features

    def get_type(self):
        return self.type

    def __call__(self, window: np.ndarray, augment: bool = False) -> tf.Tensor:
        window = window.astype(self.type)
        output = None

        if augment:
            for aug in self.augmentations:
                if aug == 'jitter':
                    window = add_noise(window)
                elif aug == 'scale':
                    window = random_scale(window)
                elif aug == 'mask':
                    window = time_mask(window)
                elif aug == 'rotate':
                    window[:, [self.x, self.y, self.z]] = random_rotate(window[:, [self.x, self.y, self.z]])

        for feature in self.conf.features:
            f_idx = self.channels[feature]
            feature_window = window[:, f_idx]

            if output is None:
                output = feature_window[:, np.newaxis]

            else:
                output = np.concatenate(
                    (output, feature_window[:, np.newaxis]),
                    axis=1
                )

        output = tf.convert_to_tensor(output, dtype = tf.float32)

        return output

