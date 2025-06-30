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
        strides=(step * X.strides[0], X.strides[0], X.strides[1])
    )

    return X

def form_y(x: pd.DataFrame, length: int, step: int, start: int, task: str,
           targets: str, target_pos: str) -> np.ndarray:

    if task == 'gait_parameters':
        y_cols = x.columns[x.columns.str.contains('swing|stance|support|step')]
    elif task == 'gait_events':
        y_cols = x.columns[x.columns.str.contains('HS|TO') & ~x.columns.str.contains('raw')]
    elif task == 'gait_phases':
        y_cols = x.columns[x.columns.str.contains('stance')]

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    y_segs = groups.apply(lambda g: segment(g[y_cols], length, step, start))
    y_segs = np.concatenate(y_segs.values)

    if task == 'gait_parameters':
        y_targets = np.nanmean(y_segs, axis=1)

    elif task == 'gait_events' or task == 'gait_phases':
        if targets == 'one':
            if target_pos == 'center':
                y_targets = y_segs[:, length//2, :]
            elif target_pos == 'start':
                y_targets = y_segs[:, 0, :]
            elif target_pos == 'end':
                y_targets = y_segs[:, -1, :]
        elif targets == 'all':
            y_targets = y_segs

    else:
        y_targets = y_segs

    return y_targets.squeeze()

def form_t(x: pd.DataFrame, length: int, step: int, start: int, targets: str, targets_pos: str) -> np.ndarray:
    t_cols = ['dataset', 'subject_id', 'activity_id', 'timestamp']

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    t_segs = groups.apply(lambda g: segment(g[t_cols], length, step, start))
    t_segs = np.concatenate(t_segs.values)

    if targets == 'one':
        if targets_pos == 'center':
            t_info = t_segs[:, length // 2, :]
        elif targets_pos == 'start':
            t_info = t_segs[:, 0, :]
        elif targets_pos == 'end':
            t_info = t_segs[:, -1, :]

    elif targets == 'all':
        t_info = np.concatenate((t_segs[:, 0, :-1], t_segs[:, :, -1]), axis=1)

    return t_info

def form(x: pd.DataFrame, length: int, step: int, task: str,
         targets: str, target_pos: str, start: int = 0):

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

    y_targets = form_y(x, length, step, start, task, targets, target_pos)
    t_info = form_t(x, length, step, start, targets, target_pos)

    x_segs = x_segs[valid_indices]
    y_targets = y_targets[valid_indices]
    t_info = t_info[valid_indices]

    return x_segs, y_targets, t_info, sizes, channels

def sl_finalize(x: pd.DataFrame, length: int, step: int, task: str,
             targets: str, target_pos: str):

    if x.shape[0] == 0:
        return None, None, None

    x = x.copy()
    X, Y, T, sizes, channels = form(x, length, step, task, targets, target_pos)
    return (X, Y, T), sizes, channels

def stack(x: pd.DataFrame, length: int, step: int, anchor: str, target: str,
          t_target: str, t_pos: str, start: int = 0):
    x = x.copy()

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    valid_segs = groups.apply(lambda g: segment(g[['irregular']], length, step, start))
    valid_segs = np.concatenate(valid_segs.values).squeeze()
    valid_indices = np.argwhere(np.sum(valid_segs, axis=1) == 0).squeeze()

    features = x.columns[x.columns.str.contains('acc|jerk|low|norm|angle|gyr')]

    anchor_feats = features[features.str.contains(anchor)]
    target_feats = features[features.str.contains(target)]

    if len(anchor_feats) == 0 or len(target_feats) == 0:
        return None, None, None, None

    channels = {k.replace('_' + anchor, "") : v for v, k in enumerate(anchor_feats)}

    anchor_segs = groups.apply(lambda g: segment(g[anchor_feats], length, step, start))
    sizes = anchor_segs.apply(lambda g: len(g)).unstack('activity_id').to_dict('index')
    anchor_segs = np.concatenate(anchor_segs.values)

    target_segs = groups.apply(lambda g: segment(g[target_feats], length, step, start))
    target_segs = np.concatenate(target_segs.values)

    t_info = form_t(x, length, step, start, t_target, t_pos)

    anchor_segs = anchor_segs[valid_indices]
    target_segs = target_segs[valid_indices]
    t_info = t_info[valid_indices]

    anchor_ws = anchor_segs[:, np.newaxis, ...]
    target_ws = target_segs[:, np.newaxis, ...]
    pair_segs = np.concatenate((anchor_ws, target_ws), axis=1)

    return pair_segs, t_info, sizes, channels

def ssl_finalize(x: pd.DataFrame, length: int, step: int, anchor: str, target: str,
                 t_target: str, t_pos: str):
    if x.shape[0] == 0:
        return None, None, None

    x = x.copy()
    X, T, sizes, channels = stack(x, length, step, anchor, target, t_target, t_pos)
    return (X, T), sizes, channels


