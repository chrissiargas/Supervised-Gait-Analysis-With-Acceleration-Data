import copy

import pandas as pd
import ruamel.yaml
import os
import time
import gc

from parameters import sl_params
from preprocessing.building import builder
from models.sl_training import train_evaluate

import matplotlib.pyplot as plt
generate = False

def reset_tensorflow_keras_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    _ = gc.collect()

def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:
            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)


def config_save(paramsFile):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        parameters = yaml.load(fp)

    with open(paramsFile, 'w') as fb:
        yaml.dump(parameters, fb)

def save(path, history, hparams=None):
    if not hparams:
        try:
            os.makedirs(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    else:
        try:
            path = os.path.join(path, hparams)
            os.makedirs(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    params_file = os.path.join(path, "parameters.yaml")
    config_save(params_file)

    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Estimation performance')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')

    historyFile = os.path.join(path, hparams + "_history.png")
    plt.savefig(historyFile)

    plt.close(fig)

def sl_experiment():
    data = builder(generate)
    train_evaluate(data, summary=True, verbose=True)
    del data

    return None, None

def supervised(archive_path):
    parameters = sl_params

    for param_name, param_value in parameters.items():
        config_edit('main_args', param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))

    reset_tensorflow_keras_backend()

    xp_model, xp_history = sl_experiment()

