import os.path

import keras
from keras.src.saving.saving_lib import save_weights_only
from tensorflow.python.eager.profiler_client import monitor
from tensorflow.python.keras.models import save_model
from tensorflow.python.ops.distributions.categorical import Categorical
from tensorflow_graphics.geometry.transformation.rotation_matrix_3d import rotate

from models.architectures import CNNGRU_encoder
from models.head import single_head
from preprocessing.building import builder
from config_parser import Parser
from models.metrics import binary_accuracies, Metrics

import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Optimizer
from keras.losses import Loss
from keras.losses import BinaryCrossentropy, MeanSquaredError, CategoricalCrossentropy
from keras.metrics import binary_accuracy, categorical_accuracy
import keras.backend as K
import shutil
import numpy as np
from typing import Optional, Dict
from models.losses import get_weighted_BCE, get_BCE, get_dice_loss
from models.rotation import quaternionLayer, rotateByAxisLayer
from plots import plot_signal


class predictor(Model):
    def __init__(self, data: builder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = Parser()
        config.get_args()
        self.conf = config

        self.class_weights = data.class_weights if config.class_weights else None

        if config.rotation_layer is None:
            self.rotation_layer = None
        elif config.rotation_layer == 'quaternion':
            self.rotation_layer = quaternionLayer(data.input_shape)
        elif config.rotation_layer == 'rotate_by_axis':
            self.rotation_layer = rotateByAxisLayer(data.input_shape)

        units = 64
        if config.architecture == 'cnn-gru':
            self.encoder_layer = CNNGRU_encoder(units)
        else:
            self.encoder_layer = None

        if config.head == 'single':
            self.head_layer = single_head(len(config.labels), [units // 2])
        else:
            self.head_layer = None

        self.data = data
        self.optimizer = Optimizer(learning_rate=config.learning_rate)
        self.loss_function = Loss()
        self.binary_accuracy = None
        self.BCE_loss_tracker = None

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)

        if self.conf.optimizer == 'adam':
            self.optimizer = Adam(learning_rate=self.conf.learning_rate)

        if self.conf.task == 'gait_phases' or self.conf.task == 'gait_events':
            if self.class_weights is None:
                self.loss_function = get_BCE(self.conf)
            else:
                self.loss_function = get_weighted_BCE(self.class_weights, self.conf)

        elif self.conf.task == 'gait_parameters':
            self.loss_function = keras.losses.MeanSquaredError()

        self.BCE_loss_tracker = keras.metrics.Mean(name='loss')
        self.binary_accuracy = binary_accuracies(self.data, self.conf)

    def build_model(self, inputs_shape):
        self.build(inputs_shape)
        _ = self(tf.keras.Input(shape=inputs_shape))

    @property
    def metrics(self):
        return [
            self.BCE_loss_tracker
        ]

    @tf.function
    def call(self, inputs, training=None):
        x = inputs
        if self.rotation_layer:
            x = self.rotation_layer(x)
        if self.encoder_layer:
            x = self.encoder_layer(x)
        if self.head_layer:
            x = self.head_layer(x)
        outputs = x

        return outputs

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_ = x
            if self.rotation_layer:
                x_ = self.rotation_layer(x_)
            if self.encoder_layer:
                x_ = self.encoder_layer(x_)
            if self.head_layer:
                x_ = self.head_layer(x_)
            y_ = x_

            loss = self.loss_function(y, y_)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.BCE_loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y = data

        x_ = x
        if self.rotation_layer:
            x_ = self.rotation_layer(x_)
        if self.encoder_layer:
            x_ = self.encoder_layer(x_)
        if self.head_layer:
            x_ = self.head_layer(x_)
        y_ = x_

        loss = self.loss_function(y, y_)
        self.BCE_loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}


def train_evaluate(data: builder, summary: bool = False, verbose: bool = False):
    config = Parser()
    config.get_args()

    train, test, val = data()

    model = predictor(data)
    model.compile()
    model.build_model(data.input_shape)
    model.summary()

    train_steps = data.train_size // config.batch_size
    test_steps = data.test_size // config.batch_size
    val_steps = data.val_size // config.batch_size

    log_dir = os.path.join('logs', config.architecture + '_TB')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    try:
        shutil.rmtree(log_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    model_dir = os.path.join('archive', 'models', config.architecture)
    model_file = '%s.weights.h5' % config.architecture
    model_file = os.path.join(model_dir, model_file)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    try:
        os.remove(model_file)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard = TensorBoard(log_dir, histogram_freq=1)

    save_model = ModelCheckpoint(
        filepath = model_file,
        monitor = 'val_loss',
        verbose = 0,
        save_best_only = True,
        mode = 'min',
        save_weights_only = True
    )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0,
        patience = 5,
        mode = 'min',
        verbose = 1
    )

    # scores = ['accuracy', 'binary_crossentropy', 'f1_score']
    # train_metrics = Metrics(train, train_steps, log_dir, on='epoch_end', scores=scores, verbose=0)
    # val_metrics = Metrics(val, val_steps, log_dir, on='epoch_end', scores=scores, verbose=0)
    callbacks = [
        tensorboard,
        save_model,
        early_stopping
    ]

    model.fit(
        train,
        epochs = config.epochs,
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = callbacks,
        verbose = 1
    )

    model.load_weights(model_file)

    scores = ['accuracy','f1_score', 'precision', 'recall']
    test_metrics = Metrics(test, test_steps, log_dir, on='test_end', scores=scores, verbose=0)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics], verbose=0)

    set = 'test'
    subject = 3
    activity = 3

    df, _, _ = data.compare_yy_(model, which=set, subject=subject, activity=activity)

    show_events = True if config.task == 'gait_events' else False
    show_phases = True if config.task == 'gait_phases' else False

    for i, start in enumerate(range(0, df.shape[0], 1000)):
        plot_signal(df, 'left_lower_arm', subject=subject,
                    activity=activity, start=start, length=1000,
                    show_events=show_events, features='acc', turn=i,
                    show_phases=show_phases, raw=True, figpath=None)









