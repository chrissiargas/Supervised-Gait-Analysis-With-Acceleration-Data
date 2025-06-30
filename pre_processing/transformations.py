import numpy as np
from config.config_parser import Parser
import tensorflow as tf
from pre_processing.augmentations import random_rotate
from typing import Dict

class transformer:
    def __init__(self, channels: Dict, batch: bool = False):
        config = Parser()
        config.get_args()
        self.conf = config

        self.channels = channels
        self.batch_size = self.conf.batch_size
        self.length = self.conf.length
        self.n_features = len(self.conf.features)
        self.type = np.float32
        self.xyz = self.conf.xyz
        self.batch = batch

        self.x = self.channels['acc_x']
        self.y = self.channels['acc_y']
        self.z = self.channels['acc_z']

        if self.conf.augmentations is None:
            self.augmentations = []
        else:
            self.augmentations = self.conf.augmentations

    def get_shape(self):
        if self.batch:
            if self.xyz:
                return self.batch_size, self.length, 3
            else:
                return self.batch_size, self.length, self.n_features

        else:
            if self.xyz:
                return self.length, 3
            else:
                return self.length, self.n_features

    def get_type(self):
        return self.type

    def __call__(self, windows: np.ndarray, augment: bool = False) -> tf.Tensor:
        windows = windows.astype(self.type)
        output = None

        if self.batch:
            if augment:
                for aug in self.augmentations:
                    if aug == 'rotate':
                        xyz = windows[..., [self.x, self.y, self.z]]
                        windows[..., [self.x, self.y, self.z]] = random_rotate(xyz, around='x')

            if self.xyz:
                output = windows[..., [self.x, self.y, self.z]]

            else:
                for feature in self.conf.features:
                    f_idx = self.channels[feature]
                    feature_windows = windows[..., f_idx]

                    if output is None:
                        output = feature_windows[..., np.newaxis]

                    else:
                        output = np.concatenate(
                            (output, feature_windows[..., np.newaxis]),
                            axis=-1
                        )

        else:
            if augment:
                for aug in self.augmentations:
                    if aug == 'rotate':
                        xyz = windows[:, [self.x, self.y, self.z]]
                        windows[:, [self.x, self.y, self.z]] = random_rotate(xyz, around='x')

            if self.xyz:
                output = windows[:, [self.x, self.y, self.z]]

            else:
                for feature in self.conf.features:
                    f_idx = self.channels[feature]
                    feature_window = windows[:, f_idx]

                    if output is None:
                        output = feature_window[:, np.newaxis]

                    else:
                        output = np.concatenate(
                            (output, feature_window[:, np.newaxis]),
                            axis=1
                        )

        output = tf.convert_to_tensor(output, dtype = tf.float32)
        return output

