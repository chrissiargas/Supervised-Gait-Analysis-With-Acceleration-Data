
import keras
import tensorflow as tf

from model_utils.architectures import CNNGRU_encoder, transformer_encoder, unet_encoder
from pre_processing.building import sl_builder, ssl_builder
from config.config_parser import Parser
from model_utils.metrics import binary_accuracies, f1_scores
from keras.models import Model
from keras.optimizers import Adam, Optimizer
from keras.losses import Loss, Dice, Tversky, BinaryFocalCrossentropy, cosine_similarity
from tensorflow.math import l2_normalize
from model_utils.losses import get_weighted_BCE, get_BCE
from model_utils.head import single_head, multiple_head, temporal_head, projector_head
from model_utils.rotation import rotateByAxis
from typing import Optional
import numpy as np

class sl_model(Model):
    def __init__(self, data: sl_builder,
                 rotation_layer: Optional[str] = None,
                 architecture: Optional[str] = None,
                 head: Optional[str] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = Parser()
        config.get_args()
        self.conf = config

        self.class_weights = data.class_weights if config.class_weights else None

        rotation_layer = rotation_layer if rotation_layer else config.rotation_layer
        if rotation_layer is None:
            self.rotation_layer = None
        elif rotation_layer == 'rotate_by_axis':
            self.rotation_layer = rotateByAxis(data.input_shape)
        else:
            self.rotation_layer = None

        units = 32

        architecture = architecture if architecture else config.architecture
        if architecture == 'cnn-gru':
            self.encoder_layer = CNNGRU_encoder(units)
        elif architecture == 'transformer':
            self.encoder_layer = transformer_encoder(units)
        elif architecture == 'unet':
            self.encoder_layer = unet_encoder(units)
        else:
            self.encoder_layer = None

        head = head if head else config.head
        if head == 'single':
            self.head_layer = single_head(len(config.labels), [units // 2], temporal=False)
        elif head == 'multi':
            self.head_layer = multiple_head(len(config.labels), [units // 2])
        elif head == 'temporal':
            self.head_layer = temporal_head(len(config.labels), [units])
        elif head == 'temporal_single':
            self.head_layer = single_head(self.conf.length, [units // 2], temporal=True)
        else:
            self.head_layer = None

        self.classes = data.classes
        self.optimizer = Optimizer(learning_rate=config.learning_rate)
        self.target_loss = Loss()
        self.BCE_loss_tracker = None
        self.metric_trackers = None

    def compile(self, loss: Optional[str] = None, *args, **kwargs):
        super().compile(*args, **kwargs)

        if self.conf.optimizer == 'adam':
            self.optimizer = Adam(learning_rate=self.conf.learning_rate)

        loss = loss if loss is not None else self.conf.loss
        if loss == 'bce':
            if self.class_weights:
                self.target_loss = get_weighted_BCE(self.class_weights, self.conf)
            elif self.class_weights:
                self.target_loss = get_BCE(self.conf)

        elif loss == 'tversky':
            self.target_loss = Tversky(alpha=0.7, beta=0.3)
        elif loss == 'dice':
            self.target_loss = Dice()
        elif loss == 'focal_bce':
            self.target_loss = BinaryFocalCrossentropy()

        self.BCE_loss_tracker = keras.metrics.Mean(name='loss')

        if self.conf.metric == 'accuracy':
            self.metric_trackers = binary_accuracies(self.classes, self.conf)
        elif self.conf.metric == 'f1_score':
            self.metric_trackers = f1_scores(self.classes, self.conf)


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
            *self.metric_trackers
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
        for metric_tracker in self.metric_trackers:
            metric_tracker.update_state(y, y_)

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
        for metric_tracker in self.metric_trackers:
            metric_tracker.update_state(y, y_)

        return {m.name: m.result() for m in self.metrics}


class contrastive_model(Model):
    def __init__(self, data: ssl_builder,
                 architecture: Optional[str] = None,
                 attach_head: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = Parser()
        config.get_args()
        self.conf = config

        self.L = 128

        architecture = architecture if architecture else config.architecture
        if architecture == 'cnn-gru':
            self.anchor_encoder = CNNGRU_encoder(self.L)
            self.target_encoder = CNNGRU_encoder(self.L)
        else:
            self.anchor_encoder = None
            self.target_encoder = None

        attach_head = attach_head if attach_head else config.attach_head
        if attach_head:
            self.anchor_head = projector_head(self.L, [])
            self.target_head = projector_head(self.L, [])

        self.input_shape = data.input_shape
        self.contrastive_optimizer = Optimizer(learning_rate=config.learning_rate)
        self.batch_size = config.batch_size
        self.temperature = 0.3

        if self.conf.neg_pos == 'other':
            self.negative_mask = np.ones((2 * self.batch_size, 2 * self.batch_size))
            self.negative_mask[:self.batch_size, :self.batch_size] = 0
            self.negative_mask[-self.batch_size:, -self.batch_size:] = 0
            self.negative_mask = tf.constant(self.negative_mask, dtype=tf.float32)
        elif self.conf.neg_pos == 'all':
            self.negative_mask = tf.cast(~tf.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool), tf.float32)

    def compile(self, loss: Optional[str] = None, *args, **kwargs):
        super().compile(*args, **kwargs)

        if self.conf.optimizer == 'adam':
            self.contrastive_optimizer = Adam(learning_rate=self.conf.learning_rate)

        self.contrastive_loss_tracker = keras.metrics.Mean(name='loss')

    def get_contrastive_loss(self, anchor_embeddings, target_embeddings):
        anchor_embeddings = l2_normalize(anchor_embeddings, axis=1)
        target_embeddings = l2_normalize(target_embeddings, axis=1)

        representations = tf.concat([anchor_embeddings, target_embeddings], axis=0)

        sim_matrix = -cosine_similarity(tf.expand_dims(representations, axis=1),
                                        tf.expand_dims(representations, axis=0),
                                        axis=2)

        sim_ij = tf.linalg.diag_part(sim_matrix, k=self.batch_size)
        sim_ji = tf.linalg.diag_part(sim_matrix, k=-self.batch_size)
        positives = tf.concat([sim_ij, sim_ji], axis=0)

        nominator = tf.math.exp(positives / self.temperature)
        denominator = self.negative_mask * tf.math.exp(sim_matrix / self.temperature)

        loss_partial = -tf.math.log(nominator / tf.reduce_sum(denominator, axis=1))
        loss = tf.reduce_sum(loss_partial) / (2 * self.batch_size)

        return loss

    def build_model(self, inputs_shape):
        self.build(inputs_shape)
        _ = self(tf.keras.Input(shape=inputs_shape), which='anchor')
        _ = self(tf.keras.Input(shape=inputs_shape), which='target')

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker
        ]

    @tf.function
    def call(self, inputs, which: str = 'anchor'):
        x_ = inputs

        if which == 'anchor':
            if self.anchor_encoder:
                x_ = self.anchor_encoder(x_)
            if self.anchor_head:
                x_ = self.anchor_head(x_)

        elif which == 'target':
            if self.target_encoder:
                x_ = self.target_encoder(x_)
            if self.target_head:
                x_ = self.target_head(x_)

        y_ = x_
        return y_

    @tf.function
    def train_step(self, data):
        anchor_inputs, target_inputs = data
        with tf.GradientTape() as tape:
            anchor_embeddings = self.anchor_encoder(anchor_inputs)
            target_embeddings = self.target_encoder(target_inputs)

            if self.anchor_head:
                anchor_embeddings = self.anchor_head(anchor_embeddings)
            if self.target_head:
                target_embeddings = self.target_head(target_embeddings)

            contrastive_loss = self.get_contrastive_loss(anchor_embeddings, target_embeddings)

        gradients = tape.gradient(contrastive_loss, self.trainable_variables)
        self.contrastive_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        anchor_inputs, target_inputs = data

        anchor_embeddings = self.anchor_encoder(anchor_inputs)
        target_embeddings = self.target_encoder(target_inputs)

        if self.anchor_head:
            anchor_embeddings = self.anchor_head(anchor_embeddings)
        if self.target_head:
            target_embeddings = self.target_head(target_embeddings)

        contrastive_loss = self.get_contrastive_loss(anchor_embeddings, target_embeddings)

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

