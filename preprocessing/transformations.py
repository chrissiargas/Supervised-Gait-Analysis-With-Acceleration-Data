import matplotlib.pyplot as plt
import numpy as np
from config_parser import Parser
import tensorflow as tf
from preprocessing.augmentations import random_rotate
from typing import Dict

class transformer:
    def __init__(self, channels: Dict):
        config = Parser()
        config.get_args()
        self.conf = config

        self.channels = channels
        self.length = self.conf.length
        self.n_features = len(self.conf.features)
        self.type = np.float32
        self.xyz = self.conf.xyz

        self.x = self.channels['acc_x']
        self.y = self.channels['acc_y']
        self.z = self.channels['acc_z']

        if self.conf.augmentations is None:
            self.augmentations = []
        else:
            self.augmentations = self.conf.augmentations

    def get_shape(self):
        if self.xyz:
            return self.length, 3
        else:
            return self.length, self.n_features

    def get_type(self):
        return self.type

    def __call__(self, window: np.ndarray, augment: bool = False) -> tf.Tensor:
        window = window.astype(self.type)
        output = None

        if augment:
            for aug in self.augmentations:
                if aug == 'rotate':
                    xyz = window[:, [self.x, self.y, self.z]]
                    window[:, [self.x, self.y, self.z]] = random_rotate(xyz, around='x')

        if self.xyz:
            output = window[:, [self.x, self.y, self.z]]

        else:
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

