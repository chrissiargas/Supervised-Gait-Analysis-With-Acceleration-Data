import ruamel.yaml
import os
import gc

from config.parameters import sl_params, ssl_params
from post_processing.results import get_results
from pre_processing.building import sl_builder, ssl_builder
from model_utils.supervised import sl_train_evaluate, ssl_train_evaluate
from post_processing.visualize import visualize
from config.config_utils import config_edit, config_save
import shutil
import matplotlib.pyplot as plt
from config.config_parser import Parser
from typing import Optional, Dict
import json
from datetime import datetime
from typing import List

def get_dataset_subjects(dataset: str):
    if dataset == 'MMgait':
        all_subs = [1001, 1002, 1003, 1004, 1005, 1007, 1008,
                    1010, 1012, 1013, 1014, 1015, 1016, 1017,
                    1018, 1019, 1022, 1023, 1024, 1025]
    elif dataset == 'nonan':
        all_subs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15,
                    16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 63,
                    9, 13, 19, 40, 81, 102, 103, 104, 107, 109,
                    110, 112, 113, 116, 119, 120, 123, 125, 126,
                    127, 128, 129, 130, 131, 133, 134, 135, 136,
                    137, 138, 139, 140, 141, 142, 143, 145, 146,
                    154, 156, 157, 158]
    elif dataset == 'nonan_young':
        all_subs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15,
                    16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 63]
    elif dataset == 'nonan_old':
        all_subs = [9, 13, 19, 40, 81, 102, 103, 104, 107, 109,
                    110, 112, 113, 116, 119, 120, 123, 125, 126,
                    127, 128, 129, 130, 131, 133, 134, 135, 136,
                    137, 138, 139, 140, 141, 142, 143, 145, 146,
                    154, 156, 157, 158]

    return all_subs

def reset_tensorflow_keras_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    _ = gc.collect()

def save_results(results: Dict, theme: Optional[str] = None):
    config = Parser()
    config.get_args()

    args = config.as_str(theme, 'sl', with_event=False)
    placement = config.position

    result_dir = os.path.join('archive', 'results', placement, args)
    result_file = '%s.json' % datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    result_file = os.path.join(result_dir, result_file)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    try:
        os.remove(result_file)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    with open(result_file, 'w') as f:
       json.dump(results, f)

def save_plots(errors: Dict, windows: Dict, cycles: Dict, cumms: Dict, theme: Optional[str] = None):
    config = Parser()
    config.get_args()

    args = config.as_str(theme, 'sl', with_event=False)
    placement = config.position

    windows_dir = os.path.join('archive', 'plots', 'windows', placement, args)
    cycles_dir = os.path.join('archive', 'plots', 'cycles', placement, args)
    errors_dir = os.path.join('archive', 'plots', 'errors', placement, args)
    cummulative_dir = os.path.join('archive', 'plots', 'cummulative', placement, args)

    if not os.path.isdir(windows_dir):
        os.makedirs(windows_dir)

    if not os.path.isdir(cycles_dir):
        os.makedirs(cycles_dir)

    if not os.path.isdir(errors_dir):
        os.makedirs(errors_dir)

    if not os.path.isdir(cummulative_dir):
        os.makedirs(cummulative_dir)

    for id in windows.keys():
        file_name = '%s|%s.png' % (str(id), datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        window_file = os.path.join(windows_dir, file_name)
        cycle_file = os.path.join(cycles_dir, file_name)
        error_file = os.path.join(errors_dir, file_name)
        cumm_file = os.path.join(cummulative_dir, file_name)

        try:
            os.remove(window_file)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            os.remove(cycle_file)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            os.remove(error_file)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            os.remove(cumm_file)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        windows[id].savefig(window_file, format="png", bbox_inches="tight")
        plt.close(windows[id])

        cycles[id].savefig(cycle_file, format="png", bbox_inches="tight")
        plt.close(cycles[id])

        errors[id].savefig(error_file, format="png", bbox_inches="tight")
        plt.close(errors[id])

        cumms[id].savefig(cumm_file, format="png", bbox_inches="tight")
        plt.close(cumms[id])


def save_history(config: Parser, history, theme: Optional[str] = None):
    args = config.as_str(theme, 'sl')
    placement = config.position

    history_dir = os.path.join('archive', 'history', placement, args)
    history_file = '%s.png' % (str(config.labels) + '-' + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    history_file = os.path.join(history_dir, history_file)

    if not os.path.isdir(history_dir):
        os.makedirs(history_dir)
    try:
        os.remove(history_file)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Estimation performance')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.savefig(history_file)
    plt.close(fig)

def sl_experiment(generate, subjects: List[int], theme: Optional[str] = None):
    data = sl_builder(generate, subjects)
    history = sl_train_evaluate(data, summary=True, verbose=True, theme=theme)
    save_history(data.conf, history, theme)

    del data

def ssl_experiment(generate):
    data = ssl_builder(generate)
    model = ssl_train_evaluate(data, summary=True, verbose=True)
    del data

def supervised(dataset: str, position: str, subjects: List[int], compare: Optional[str] = None):
    generate = False

    if not compare:
        config_edit('data_args', 'position', position)


        for subject in subjects:
            parameters = sl_params
            parameters['test_hold_out'] = [subject]
            parameters['val_hold_out'] = 0.2

            if parameters['task'] == 'gait_events':
                labels = [['LF_HS'], ['LF_TO'], ['RF_HS'], ['RF_TO']]

            if parameters['task'] == 'gait_phases':
                labels = ['LF_stance', 'RF_stance']

            for label in labels:
                parameters['labels'] = label

                for param_name, param_value in parameters.items():
                    config_edit('main_args', param_name, param_value)

                reset_tensorflow_keras_backend()

                sl_experiment(generate, theme=None)
                generate = False

            (score_results, error_results, cumm_results,
             error_plots, window_plots, cycle_plots, cumm_plots) = get_results(subjects=[1007, 1008, 1012, 1015],
                                                                               with_window=True,
                                                                               with_cycle=True,
                                                                               with_error=True,
                                                                               with_cumulative=True,
                                                                               theme=None)

            save_results(score_results, None)
            save_plots(error_plots, window_plots, cycle_plots, cumm_plots, None)

    if compare == 'preprocessing':
        task = 'gait_events'

        experiment_args = {
            'include_gravity': [True],
            # 'new_features': [['norm_xyz'],
            #                  ['norm_yz'],
            #                  ['norm_xy'],
            #                  ['norm_xz'],
            #                  ['jerk'],
            #                  ['x_yz_angle'],
            #                  ['y_x_angle']],
            # 'filter': ['lowpass']
        }

        if task == 'gait_events':
            labels = [['LF_HS'], ['LF_TO'], ['RF_HS'], ['RF_TO']]

        if task == 'gait_phases':
            labels = ['LF_stance', 'RF_stance']

        for pm_name, pm_values in experiment_args.items():
            for pm_value in pm_values:
                parameters = sl_params

                if pm_name == 'new_features':
                    parameters['features'] = ['acc_x', 'acc_y', 'acc_z', *pm_value]

                parameters[pm_name] = pm_value

                # for label in labels:
                #     parameters['labels'] = label
                #
                #     for param_name, param_value in parameters.items():
                #         config_edit('main_args', param_name, param_value)
                #
                #     reset_tensorflow_keras_backend()
                #
                #     sl_experiment(generate, theme='preprocessing')
                #     generate = False

                (score_results, error_results,  cumm_results,
                 error_plots, window_plots, cycle_plots, cumm_plots) = get_results(subjects=[1007, 1008, 1012, 1015],
                                                                                with_window=True,
                                                                                with_cycle=True,
                                                                                with_error=True,
                                                                                with_cumulative=True,
                                                                                theme='preprocessing')

                save_results(score_results, 'preprocessing')
                save_plots(error_plots, window_plots, cycle_plots, cumm_plots, 'preprocessing')

def self_supervised():
    generate = True
    parameters = ssl_params

    for param_name, param_value in parameters.items():
        config_edit('main_args', param_name, param_value)

    reset_tensorflow_keras_backend()

    ssl_experiment(generate)
    generate = False



