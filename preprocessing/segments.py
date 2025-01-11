import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import math


def to_categorical(x: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    x = x.copy()

    act_factor = pd.factorize(x['activity'])

    x['subject_id'] = x['subject'].astype(int)
    x['activity_id'], act_dict = act_factor[0], act_factor[1].values
    act_dict = {k: v for v, k in enumerate(act_dict)}

    return x, act_dict

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

    groups = x.groupby(['subject', 'activity'])
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
    t_cols = ['subject_id', 'activity_id', 'timestamp']

    groups = x.groupby(['subject', 'activity'])
    t_segs = groups.apply(lambda g: segment(g[t_cols], length, step, start))
    t_segs = np.concatenate(t_segs.values)

    t_info = np.concatenate((t_segs[:, 0, :], t_segs[:, -1, [-1]]), axis=1)

    return t_info

def form(x: pd.DataFrame, length: int, step: int, start: int, task: str, get_events: bool = False):
    x = x.copy()

    x_cols = x.columns[x.columns.str.contains('acc|jerk|low|norm|angle')]
    groups = x.groupby(['subject', 'activity'])
    x_segs = groups.apply(lambda g: segment(g[x_cols], length, step, start))
    x_segs = np.concatenate(x_segs.values)

    y_targets = form_y(x, length, step, start, task)
    t_info = form_t(x, length, step, start)

    if get_events:
        y_segs = form_y(x, length, step, 'event_series')
        return x_segs, y_targets, y_segs, t_info

    return x_segs, y_targets, t_info

def finalize(x: pd.DataFrame, length: int, step: int, task: str, get_events: bool = False):
    x = x.copy()

    nan_col = ['is_NaN']
    moved_cols = [col for col in x.columns if col not in nan_col] + nan_col
    x = x[moved_cols]

    x, act_dict = to_categorical(x)

    start = 1
    if get_events:
        X, Y, events, T = form(x, length, step, start, task, get_events=True)
        return (X, Y, events, T), act_dict

    else:
        X, Y, T = form(x, length, step, start, task, get_events=False)
        return (X, Y, T), act_dict
