import os
import pprint

import pandas as pd

from config.config_parser import Parser
from model_utils.supervised import alligaitor
from post_processing.analytics import get_conf, get_time_error, get_scores, get_parameters
from post_processing.analytics_plots import plot_events, plot_parameters, plot_confusion
from post_processing.results import reconstruct_y
from pre_processing.building import builder
from plot_utils.plots import plot_signal, plot_results
from typing import Optional, List
import numpy as np
from rotation_utils import rotation_by_axis
from scipy.spatial.transform import Rotation

def visualize(path: Optional[str] = None,
              set: str = 'test',
              subject: Optional[int] = None,
              activity: int = 3):
    data = builder()
    data()

    config = Parser()
    config.get_args()

    model_args = f'{config.task}-{config.targets}-{str(config.labels)}'
    model_dir = f'archive/model_weights/{model_args}'
    model_file = '%s.weights.h5' % config.architecture
    model_file = f'{model_dir}/{model_file}'

    model = alligaitor(data)
    model.compile()
    model.build_model(data.input_shape)
    model.load_weights(model_file)

    if subject is None:
        subject = config.test_hold_out[0]

    df, _, _ = data.get_yy_(model, which=set, subject=subject, activity=activity, oversample=True)

    show_events = True if config.task == 'gait_events' else False
    show_phases = True if config.task == 'gait_phases' else False

    for i, start in enumerate(range(0, df.shape[0], 1000)):
        plot_signal(df, 'left_lower_arm', subject=subject,
                    activity=activity, start=start, length=1000,
                    show_events=show_events, features='acc', turn=i,
                    show_phases=show_phases, raw=True, real=False, figpath=path)


def visualize_rot(path: Optional[str] = None,
                  set: str = 'test',
                  subject: Optional[int] = None,
                  activity: int = 3):

    data = builder()
    data()

    config = Parser()
    config.get_args()

    model_dir = f'archive/model_weights/{config.architecture}'
    model_file = '%s.weights.h5' % config.architecture
    model_file = f'{model_dir}/{model_file}'

    model = alligaitor(data)
    model.compile()
    model.build_model(data.input_shape)
    model.load_weights(model_file)

    if subject is None:
        subject = config.test_hold_out[0]

    start = 4000
    window = 1000

    group = Rotation.create_group('O')
    Rs = group.as_matrix()

    for subject in config.test_hold_out:
        for r, R in enumerate(Rs):
            df, _, _ = data.get_yy_(model, which=set, subject=subject, activity=activity,
                                    start=start, end=start+window, rotation=R, oversample=True)

            show_events = True if config.task == 'gait_events' else False
            show_phases = True if config.task == 'gait_phases' else False

            for i, start_ in enumerate(range(0, df.shape[0], window)):
                plot_signal(df, 'left_lower_arm', subject=subject,
                            activity=activity, start=start_, length=window,
                            show_events=show_events, features='acc', turn=f'-{angle}-',
                            show_phases=show_phases, raw=True, figpath=path, R=R)

def plot_all(subjects: Optional[List[int]] = None,
             activities: Optional[List[int]] = None):
    figpath = os.path.join('archive', 'figures')

    config = Parser()
    config.get_args()

    all_tasks = ['gait_events']
    all_targets = ['all']
    heads = ['temporal_single']
    events = ['LF_HS']

    with_parameters = False
    with_window = True

    for task in all_tasks:
        for targets, head in zip(all_targets, heads):
            stats = {}

            R = None
            results = reconstruct_y(task, targets, config.architecture, subjects,
                                    activities, events, True, rotation=R)
            time_errors = None

            for event in events:
                event_tp, event_fp, event_fn = 0, 0, 0
                event_error = []

                for subject_id in results.subject_id.unique():
                    sub_results = results[results['subject_id'] == subject_id]
                    for activity_id in results.activity_id.unique():
                        act_results = sub_results[sub_results['activity_id'] == activity_id]

                        tp, fp, fn = get_conf(act_results, event+'_pred', event+'_raw', n=5)
                        time_dists = get_time_error(act_results, event+'_pred', event+'_raw', n=5)
                        time_error = time_dists / config.fs

                        event_tp += tp
                        event_fp += fp
                        event_fn += fn

                        event_error.extend(time_error)

                precision, recall, f1_score = get_scores(event_tp, event_fp, event_fn)
                stats[event] = {
                    'true positives': event_tp,
                    'false positives': event_fp,
                    'false negatives': event_fn,
                    'precision': precision,
                    'recall': recall,
                    'f1 score': f1_score,
                    'time error mean': np.mean(event_error),
                    'time error std': np.std(event_error),
                }

                event_error = pd.DataFrame({'event': event, 'time_error': event_error})

                if time_errors is None:
                    time_errors = event_error
                else:
                    time_errors = pd.concat([time_errors, event_error], axis=0)

            plot_confusion(stats, figpath)

            HS_errors = time_errors[time_errors['event'].str.contains('HS')]
            TO_errors = time_errors[time_errors['event'].str.contains('TO')]
            plot_events(HS_errors, TO_errors, figpath)

            if with_parameters:
                gait_params = None
                for subject_id in results.subject_id.unique():
                    sub_results = results[results['subject_id'] == subject_id]
                    for activity_id in results.activity_id.unique():
                        act_results = sub_results[sub_results['activity_id'] == activity_id]
                        act_params = get_parameters(results)

                        act_params['true_value'] = act_params['true_value'] / config.fs
                        act_params['pred_value'] = act_params['pred_value'] / config.fs

                        if gait_params is None:
                            gait_params = act_params
                        else:
                            gait_params = pd.concat([gait_params, act_params], axis=0)

                plot_parameters(gait_params, True, figpath)

            if with_window:
                show_events = True if task == 'gait_events' else False
                show_phases = True if task == 'gait_phases' else False
                window = 500

                for subject in results.subject_id.unique():
                    sub_results = results[results['subject_id'] == subject]
                    for activity in sub_results.activity_id.unique():
                        act_results = sub_results[sub_results['activity_id'] == activity]

                        for i, start in enumerate(range(0, act_results.shape[0], window)):
                            plot_results(act_results, start=start, length=window,
                                        show_events=show_events, show_phases=show_phases,
                                        turn=f'{subject}-{activity}-{i}', raw=True, real=False, signal=True)

if __name__ == '__main__':
    plot_all(subjects=[2001], activities=None)
