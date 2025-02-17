import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import math


def segment(X: pd.DataFrame, length: int, step: int, start: int) -> np.ndarray:
    X = X.values[start:]

    n_windows = math.ceil((X.shape[0] - length + 1) / step)
    n_windows = max(0, n_windows)

    X = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, length, X.shape[1]),
        strides=(step * X.strides[0], X.strides[0], X.strides[1]))

    return X

def form_y(x: pd.DataFrame, length: int, step: int, start: int, task: str) -> np.ndarray:

    if task == 'gait_parameters':
        y_cols = x.columns[x.columns.str.contains('swing|stance|support|step')]
    elif task == 'gait_events':
        y_cols = x.columns[x.columns.str.contains('LF|RF|HS|TO')]
    elif task == 'gait_phases':
        y_cols = x.columns[x.columns.str.contains('swing|stance|support')]
    else:
        y_cols = x.columns[x.columns.str.contains('LF|RF|HS|TO')]

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    y_segs = groups.apply(lambda g: segment(g[y_cols], length, step, start))
    y_segs = np.concatenate(y_segs.values)

    if task == 'gait_parameters':
        y_targets = np.nanmean(y_segs, axis=1)
    elif task == 'gait_events':
        y_targets = y_segs[:, length//2, :]
    elif task == 'gait_phases':
        y_segs[np.isnan(y_segs)] = 0
        y_targets = np.array(y_segs[:, length//2, :] != 0, dtype=int)
    else:
        y_targets = y_segs

    return y_targets

def form_t(x: pd.DataFrame, length: int, step: int, start: int) -> np.ndarray:
    t_cols = ['dataset', 'subject_id', 'activity_id', 'timestamp']

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    t_segs = groups.apply(lambda g: segment(g[t_cols], length, step, start))
    t_segs = np.concatenate(t_segs.values)

    t_info = np.concatenate((t_segs[:, 0, :], t_segs[:, -1, [-1]]), axis=1)

    return t_info

def form(x: pd.DataFrame, length: int, step: int, start: int, task: str, get_events: bool = False):
    x = x.copy()

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    valid_segs = groups.apply(lambda g: segment(g[['irregular']], length, step, start))
    valid_segs = np.concatenate(valid_segs.values).squeeze()
    valid_indices = np.argwhere(np.sum(valid_segs, axis=1) == 0).squeeze()

    x_cols = x.columns[x.columns.str.contains('acc|jerk|low|norm|angle|gyr')]
    channels = {k: v for v, k in enumerate(x_cols)}
    x_segs = groups.apply(lambda g: segment(g[x_cols], length, step, start))
    sizes = x_segs.apply(lambda g: len(g)).unstack('activity_id').to_dict('index')
    x_segs = np.concatenate(x_segs.values)

    y_targets = form_y(x, length, step, start, task)
    t_info = form_t(x, length, step, start)

    if get_events:
        y_segs = form_y(x, length, step, 'event_series')
        return x_segs, y_targets, y_segs, t_info

    x_segs = x_segs[valid_indices]
    y_targets = y_targets[valid_indices]
    t_info = t_info[valid_indices]

    return x_segs, y_targets, t_info, sizes, channels

def finalize(x: pd.DataFrame, length: int, step: int, task: str, get_events: bool = False):
    x = x.copy()

    start = 0
    if get_events:
        X, Y, events, T, lens, channels = form(x, length, step, start, task, get_events=True)
        return (X, Y, events, T), lens, channels

    else:
        X, Y, T, lens, channels = form(x, length, step, start, task, get_events=False)
        return (X, Y, T), lens, channels
