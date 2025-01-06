import os.path

import keras
from keras.src.saving.saving_lib import save_weights_only
from tensorflow.python.eager.profiler_client import monitor
from tensorflow.python.keras.models import save_model
from tensorflow.python.ops.distributions.categorical import Categorical

from models.architectures import get_attention_encoder, get_tcn_encoder, get_CNN_encoder

from models.architectures import get_tcn_encoder
from models.head import attach_head
from preprocessing.building import builder
from config_parser import Parser
from models.metrics import binary_accuracies, Metrics

import tensorflow as tf
from keras.layers import Input, LSTM, Conv2D, Conv1D, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Optimizer
from keras.losses import BinaryCrossentropy, MeanSquaredError, CategoricalCrossentropy
from keras.metrics import binary_accuracy, categorical_accuracy
import keras.backend as K
import shutil


def train_evaluate(data: builder, summary: bool = False, verbose: bool = False):
    config = Parser()
    config.get_args()

    train, test, val = data(verbose=False)

    if config.optimizer == 'adam':
        optimizer = Adam(learning_rate=config.learning_rate)
    else:
        optimizer = Optimizer()

    loss = BinaryCrossentropy()
    metrics = binary_accuracies(data)

    if config.architecture == 'cnn':
        model = get_CNN_encoder(data.input_shape)

    if config.architecture == 'attention':
        model = get_attention_encoder(data.input_shape)

    if config.architecture == 'tcn':
        model = get_tcn_encoder(data.input_shape)

    model = attach_head(model, data.output_shape, [])

    if summary:
        print(model.summary())

    model.compile(optimizer, loss = loss, metrics = metrics)

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
        patience = 10,
        mode = 'auto',
        verbose = 1
    )

    scores = ['accuracy', 'binary_crossentropy', 'f1_score']
    val_metrics = Metrics(val, val_steps, log_dir, on='epoch_end',
                                 scores=scores, verbose=0)
    callbacks = [
        tensorboard,
        save_model,
        early_stopping,
        val_metrics
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






