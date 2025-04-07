import ruamel.yaml
import os
import gc

from config.parameters import sl_params
from pre_processing.building import builder
from model_utils.supervised import train_evaluate
from post_processing.visualize import visualize
from config.config_utils import config_edit, config_save
import shutil
import matplotlib.pyplot as plt

def reset_tensorflow_keras_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    _ = gc.collect()

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

def sl_experiment(generate):
    data = builder(generate)
    model = train_evaluate(data, summary=True, verbose=True)
    del data

def supervised():
    generate = False
    parameters = sl_params

    tasks_params = ['gait_events']
    targets_params = ['all']
    head_params = ['temporal_single']

    for task_param in tasks_params:
        parameters['task'] = task_param
        if task_param == 'gait_events':
            parameters['class_weights'] = True
            parameters['loss'] = 'tversky'
            labels_params = [['LF_HS'], ['LF_TO'], ['RF_HS'], ['RF_TO']]
        if task_param == 'gait_phases':
            parameters['class_weights'] = False
            parameters['loss'] = 'bce'
            labels_params = ['LF_stance', 'RF_stance']

        for target_param, head_param in zip(targets_params, head_params):
            parameters['targets'] = target_param
            parameters['head'] = head_param

            for label_param in labels_params:
                parameters['labels'] = label_param

                for param_name, param_value in parameters.items():
                    config_edit('main_args', param_name, param_value)

                reset_tensorflow_keras_backend()

                sl_experiment(generate)
                generate = False

                # path = os.path.join(
                #     os.path.expanduser('~'),
                #     'Pictures',
                #     task_param + '-' + target_param + '-' + label_param,
                # )
                #
                # if os.path.exists(path):
                #     shutil.rmtree(path)

                # if vis:
                #     os.mkdir(path)
                #     visualize(path)


