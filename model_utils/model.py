
import keras
import tensorflow as tf

from model_utils.architectures import CNNGRU_encoder
from preprocessing.building import builder
from config_parser import Parser
from model_utils.metrics import binary_accuracies, Metrics
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, Optimizer
from keras.losses import Loss, Dice, Tversky, BinaryFocalCrossentropy
import keras.backend as K
from model_utils.losses import get_weighted_BCE, get_BCE
from model_utils.head import single_head, multiple_head, temporal_head
from model_utils.rotation import rotateByAxis


class alligaitor(Model):
    def __init__(self, data: builder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = Parser()
        config.get_args()
        self.conf = config

        self.class_weights = data.class_weights if config.class_weights else None

        if config.rotation_layer is None:
            self.rotation_layer = None
        elif config.rotation_layer == 'rotate_by_axis':
            self.rotation_layer = rotateByAxis(data.input_shape)
        else:
            self.rotation_layer = None

        units = 64
        if config.architecture == 'cnn-gru':
            self.encoder_layer = CNNGRU_encoder(units)
        else:
            self.encoder_layer = None

        if config.head == 'single':
            self.head_layer = single_head(len(config.labels), [units // 2])
        elif config.head == 'multi':
            self.head_layer = multiple_head(len(config.labels), [units // 2])
        elif config.head == 'temporal':
            self.head_layer = temporal_head(len(config.labels), [units // 2])
        elif config.head == 'temporal_single':
            self.head_layer = single_head(self.conf.length, [units // 2])
        else:
            self.head_layer = None

        self.classes = data.classes
        self.optimizer = Optimizer(learning_rate=config.learning_rate)
        self.target_loss = Loss()
        self.BCE_loss_tracker = None
        self.accuracy_trackers = None

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)

        if self.conf.optimizer == 'adam':
            self.optimizer = Adam(learning_rate=self.conf.learning_rate)

        if self.conf.loss == 'bce':
            if self.class_weights:
                self.target_loss = get_weighted_BCE(self.class_weights, self.conf)
            elif self.class_weights:
                self.target_loss = get_BCE(self.conf)

        elif self.conf.loss == 'tversky':
            self.target_loss = Tversky(alpha=0.7, beta=0.3)
        elif self.conf.loss == 'dice':
            self.target_loss = Dice()
        elif self.conf.loss == 'focal_bce':
            self.target_loss = BinaryFocalCrossentropy()

        self.BCE_loss_tracker = keras.metrics.Mean(name='loss')
        self.accuracy_trackers = binary_accuracies(self.classes, self.conf)

    def build_model(self, inputs_shape):
        self.build(inputs_shape)
        _ = self(tf.keras.Input(shape=inputs_shape))

    def rotate(self, x, angle):
        if self.conf.rotation_layer == 'rotate_by_axis':
            cos_theta = tf.math.cos(angle)
            sin_theta = tf.math.sin(angle)
            zero = tf.zeros_like(angle)
            one = tf.ones_like(angle)

            R = tf.stack([
                tf.concat([cos_theta, zero, sin_theta], axis=-1),
                tf.concat([zero, one, zero], axis=-1),
                tf.concat([-sin_theta, zero, cos_theta], axis=-1)
            ], axis=1)

            x_rotated = tf.einsum('bij,bkj->bki', R, x)

            return x_rotated, R

    @property
    def metrics(self):
        return [
            self.BCE_loss_tracker,
            *self.accuracy_trackers
        ]

    @tf.function
    def call(self, inputs, training=None):
        x_ = inputs
        if self.rotation_layer:
            angle = self.rotation_layer(x_)
            x_, _ = self.rotate(x_, angle)
        if self.encoder_layer:
            x_ = self.encoder_layer(x_)
        if self.head_layer:
            x_ = self.head_layer(x_)
        y_ = x_

        return y_

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_ = x

            if self.rotation_layer:
                angle = self.rotation_layer(x_)
                x_, R = self.rotate(x_, angle)
            if self.encoder_layer:
                x_ = self.encoder_layer(x_)
            if self.head_layer:
                x_ = self.head_layer(x_)

            y_ = x_

            total_loss = self.target_loss(y, y_)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.BCE_loss_tracker.update_state(total_loss)
        for accuracy_tracker in self.accuracy_trackers:
            accuracy_tracker.update_state(y, y_)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y = data

        x_ = x
        if self.rotation_layer:
            angle = self.rotation_layer(x_)
            x_, _ = self.rotate(x_, angle)
        if self.encoder_layer:
            x_ = self.encoder_layer(x_)
        if self.head_layer:
            x_ = self.head_layer(x_)
        y_ = x_

        loss = self.target_loss(y, y_)

        self.BCE_loss_tracker.update_state(loss)
        for accuracy_tracker in self.accuracy_trackers:
            accuracy_tracker.update_state(y, y_)

        return {m.name: m.result() for m in self.metrics}

