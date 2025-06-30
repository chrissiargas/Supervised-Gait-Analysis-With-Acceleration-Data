import matplotlib.pyplot as plt
import numpy as np

from config.config_utils import config_edit
from pre_processing.building import sl_builder
from model_utils.model import sl_model
import pandas as pd
from post_processing.postprocess import find_peak_positions
from typing import Optional, List, Tuple
from rotation_utils import rotation_by_axis
from config.config_parser import Parser

from post_processing.analytics import get_conf, get_time_error, get_scores, get_parameters
from post_processing.analytics_plots import plot_events, plot_parameters, plot_confusion
from plot_utils.plots import plot_results, plot_cycle, plot_cummulative

ft_cols = [
    'acc_x',
    'acc_y',
    'acc_z'
]

id_cols = [
    'subject_id',
    'activity_id',
    'timestamp'
]

def split_with_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    res_df = df.copy()

    res_df['id'] = res_df[id_cols].astype(str).agg('-'.join, axis=1)
    data_df = res_df[['id', *ft_cols, *id_cols]]
    res_df = res_df.drop(columns=[*ft_cols, *id_cols])

    return res_df, data_df

def get_preds(x: pd.DataFrame, label: str) -> np.ndarray:
    x = x.copy()

    event_indices = find_peak_positions(x[label + '_prob'].values)
    predictions = np.zeros(x.shape[0])
    predictions[event_indices] = 1.0

    return predictions

def filter_events(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()

    groups = df.groupby(['subject_id', 'activity_id'])
    preds = groups.apply(lambda gr: get_preds(gr, label))
    df[label + '_pred'] = np.concatenate(preds.values)

    return df

def reconstruct_y(task: str, targets: str, arch: str,
                  subjects: List[int], activities: Optional[List[int]] = None,
                  filter: bool = True, rotation: Optional[np.ndarray] = None,
                  theme: Optional[str] = None) -> pd.DataFrame:

    data = sl_builder(subjects=subjects)
    data(selected=subjects, only_test=True)

    if task == 'gait_events':
        all_labels = [['LF_HS'], ['RF_HS'], ['LF_TO'], ['RF_TO']]
    elif task == 'gait_phases':
        all_labels = [['LF_stance'], ['RF_stance']]
    else:
        all_labels = []

    results = None
    for labels in all_labels:
        result_cols = [*labels, *ft_cols, *id_cols]

        data.conf.task = task
        data.conf.targets = targets
        data.conf.labels = labels

        model_args = data.conf.as_str(theme, 'sl')
        placement = data.conf.position
        model_dir = f'archive/model_weights/{placement}/{model_args}'
        model_file = '%s.weights.h5' % arch
        model_file = f'{model_dir}/{model_file}'

        model = sl_model(data, head=data.conf.head)
        model.compile()
        model.build_model(data.input_shape)
        model.load_weights(model_file)

        yy_ = data.get_yy_(model, subjects, activities, rotation=rotation, oversample=True)

        y_cols = yy_.columns[yy_.columns.str.contains('|'.join(result_cols))]
        yy_ = yy_[y_cols].copy()

        yy_, same = split_with_id(yy_)

        if results is None:
            results = yy_
        else:
            results = pd.merge(results, yy_, on='id', how='left')

        del model
        del yy_

    results = pd.merge(results, same, on='id', how='left')

    results.drop(columns=['id'])

    if filter:
        for labels in all_labels:
            for label in labels:
                results = filter_events(results, label)

    return results

def get_results(subjects: Optional[List[int]] = None,
                activities: Optional[List[int]] = None,
                theme: Optional[str] = None,
                with_window: bool = False,
                with_cycle: bool = False,
                with_error: bool = False,
                with_cumulative: bool = False):

    config = Parser()
    config.get_args()

    events = ['LF_HS', 'LF_TO', 'RF_HS', 'RF_TO']

    results = reconstruct_y(config.task, config.targets, config.architecture,
                            subjects, activities, True, theme=theme)

    sub_stats = {}
    sub_time_errors = {}
    sub_window = {}
    sub_cycle = {}
    sub_errors = {}
    sub_dist_stats = {}
    sub_cumms = {}

    for subject_id in results.subject_id.unique():
        sub_results = results[results['subject_id'] == subject_id].copy()

        id = int(subject_id)
        sub_stats[id] = {}
        sub_time_errors[id] = None
        sub_dist_stats[id] = {}

        for event in events:
            event_tp, event_fp, event_fn = 0, 0, 0
            event_error = []

            if with_cumulative:
                sub_dist_stats[id][event] = {}
                dist_tp = {dist: 0 for dist in range(0,11)}
                dist_fp = {dist: 0 for dist in range(0,11)}
                dist_fn = {dist: 0 for dist in range(0,11)}

            for activity_id in sub_results.activity_id.unique():
                act_results = sub_results[sub_results['activity_id'] == activity_id]

                tp, fp, fn = get_conf(act_results, event + '_pred', event + '_raw', n=5)
                time_dists = get_time_error(act_results, event + '_pred', event + '_raw', n=5)
                time_error = time_dists / config.fs

                event_tp += tp
                event_fp += fp
                event_fn += fn

                event_error.extend(time_error)

                if with_cumulative:
                    for dist in range(10):
                        tp, fp, fn = get_conf(act_results, event + '_pred', event + '_raw', n=dist)
                        dist_tp[dist] += tp
                        dist_fp[dist] += fp
                        dist_fn[dist] += fn

            precision, recall, f1_score = get_scores(event_tp, event_fp, event_fn)

            sub_stats[id][event] = {
                'true positives': int(event_tp),
                'false positives': int(event_fp),
                'false negatives': int(event_fn),
                'precision': float(precision),
                'recall': float(recall),
                'f1 score': float(f1_score),
                'time error mean': float(np.mean(event_error)),
                'time error std': float(np.std(event_error)),
            }

            if with_cumulative:
                for dist in range(10):
                    precision, recall, f1_score = get_scores(dist_tp[dist],
                                                             dist_fp[dist],
                                                             dist_fn[dist])

                    sub_dist_stats[id][event][dist] = {
                        'true positives': int(dist_tp[dist]),
                        'false positives': int(dist_fp[dist]),
                        'false negatives': int(dist_fn[dist]),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1 score': float(f1_score)
                    }

            event_error = pd.DataFrame({'event': event, 'time_error': event_error})

            if sub_time_errors[id] is None:
                sub_time_errors[id] = event_error
            else:
                sub_time_errors[id] = pd.concat([sub_time_errors[id], event_error], axis=0)

        if with_error:
            HS_errors = sub_time_errors[id][sub_time_errors[id]['event'].str.contains('HS')]
            TO_errors = sub_time_errors[id][sub_time_errors[id]['event'].str.contains('TO')]
            plot = plot_events(HS_errors, TO_errors, return_plot=True)
            sub_errors[id] = plot

    if with_cumulative:
        for subject_id in results.subject_id.unique():
            id = int(subject_id)
            plot = plot_cummulative(sub_dist_stats[id], return_plot=True)
            sub_cumms[id] = plot


    if with_window:
        show_events = True if config.task == 'gait_events' else False
        show_phases = True if config.task == 'gait_phases' else False
        window = 500

        for subject_id in results.subject_id.unique():
            sub_results = results[results['subject_id'] == subject_id].copy()
            id = int(subject_id)

            for activity_id in sub_results.activity_id.unique():
                if activity_id > 1:
                    continue

                act_results = sub_results[sub_results['activity_id'] == activity_id]

                plot = plot_results(act_results, start=10*window, length=window,
                                    show_events=show_events, show_phases=show_phases,
                                    turn=f'{subject_id}-{activity_id}', raw=True, real=False,
                                    signal=True, return_plot=True)

                sub_window[id] = plot

    if with_cycle:
        limit = 'LF_HS_raw'
        show_events = True if config.task == 'gait_events' else False
        show_phases = True if config.task == 'gait_phases' else False
        n_cycles = 50

        for subject_id in results.subject_id.unique():
            sub_results = results[results['subject_id'] == subject_id].copy()
            id = int(subject_id)

            for activity_id in sub_results.activity_id.unique():
                if activity_id > 1:
                    continue

                act_results = sub_results[sub_results['activity_id'] == activity_id]

                act_results['cycle'] = act_results[limit].cumsum()
                act_results = act_results[act_results['cycle'] > 0]
                act_results['cycle_group'] = act_results.cycle // n_cycles

                cycle_results = act_results[act_results['cycle_group'] == 2]

                plot = plot_cycle(cycle_results,
                                  show_events=show_events, show_phases=show_phases,
                                  turn=f'{subject_id}-{activity_id}', raw=True,
                                  real=False, return_plot=True)

                sub_cycle[id] = plot


    return sub_stats, sub_time_errors, sub_dist_stats, sub_errors, sub_window, sub_cycle, sub_cumms
