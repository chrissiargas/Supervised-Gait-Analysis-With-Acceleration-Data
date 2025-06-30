
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.ops.numpy_ops.np_array_ops import around

from pre_processing.features import (add_norm_xy, add_norm_xz, add_norm_xyz, add_norm_yz,
                                     add_jerk, add_angle_y_xz, add_angle_z_xy, add_angle_y_x,
                                     add_angle_x_yz)
from pre_processing.filters import (butter_lowpass_filter, lowpass_smoothing, butter_highpass_filter)
from pre_processing.gait_parameters import oversample_events
from pre_processing.rotate import rotate_by_gravity, rotate_by_pca
import matplotlib.pyplot as plt
import os
from pre_processing.irregularities import *
from pre_processing.augmentations import random_rotate

figpath = os.path.join('archive', 'figures')

def fill_nan(x: pd.DataFrame, features: List[str], how: str) -> pd.DataFrame:
    x = x.copy()

    groups = x.groupby(['subject', 'activity_id'])
    for feature in features:
        if how == 'linear':
            cleaned_ft = groups[feature].transform(lambda g: g.interpolate(method='linear'))
        elif how == 'spline':
            cleaned_ft = groups[feature].transform(lambda g: g.interpolate(method='spline'))
        elif how == 'bfill':
            cleaned_ft = groups[feature].transform(lambda g: g.fillna(method='bfill'))
        elif how == 'ffill':
            cleaned_ft = groups[feature].transform(lambda g: g.fillna(method='ffill'))

        x[feature] = cleaned_ft

    print(x[(x['is_NaN'] == True)].to_string())
    return x

def impute(x: pd.DataFrame, how: str) -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()
    features = x.columns[x.columns.str.contains("acc|gyr")]
    x = fill_nan(x, features, how)

    return x

def trim(x: pd.DataFrame, length: int = 0) -> pd.DataFrame:
    x = x.copy()

    if length != 0:
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        x = x.drop(groups.tail(length).index, axis=0)
        x = x.drop(groups.head(length).index, axis=0)
        return x

    return x

def get_period_id(x: pd.DataFrame, period: int = 0) -> pd.DataFrame:
    x = x.copy().reset_index()
    N = x.shape[0]
    total_periods = np.ceil(N / period).astype(int)

    period_id = np.floor(np.array(list(x.index)) / period).astype(int)
    period_id[period_id == total_periods - 1] -= 1

    return period_id

def is_irregular(x: pd.DataFrame, period: int, checks: List[str]) -> pd.DataFrame:
    x = x.copy()
    x['irregular'] = 0

    if checks is None:
        return x

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    period_id = groups.apply(lambda g: get_period_id(g, period))
    x['period_id'] = np.concatenate(period_id.values)

    for ds, ds_df in x.groupby('dataset'):
        for sub, sub_df in ds_df.groupby('subject_id'):
            for act, act_df in sub_df.groupby('activity_id'):
                Flags = []

                for check in checks:
                    flags = []
                    if check == 'rotation':
                        flags = check_rotation(act_df)

                    Flags.append(set(flags))

                all_flags = set.union(*Flags)
                x.loc[(x.dataset == ds) &
                       (x.subject_id == sub) &
                       (x.activity_id == act) &
                       (x.period_id.isin(all_flags)), 'irregular'] = 1

    return x


def orient(x: pd.DataFrame,
           fs: float,
           how: Optional[str] = None,
           placement: bool = False) -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()

    features = x.columns[x.columns.str.contains('acc')]

    if placement:
        positions = list(set(features.str.split('_').str[2]))
    else:
        positions = ['']

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    period_id = groups.apply(lambda g: get_period_id(g, period=2000))
    x['rotation_period'] = np.concatenate(period_id.values)

    groups = x.groupby(['dataset', 'subject_id', 'activity_id', 'rotation_period'])

    for position in positions:
        pos_features = features[features.str.contains(position)]

        if 'gravity' in how:
            rotated = groups.apply(lambda gr: rotate_by_gravity(gr, pos_features, fs))
            x[pos_features] = np.concatenate(rotated.values)

        if 'pca' in how:
            rotated = groups.apply(lambda gr: rotate_by_pca(gr, pos_features))
            x[pos_features] = np.concatenate(rotated.values)

    x = x.drop(columns=['rotation_period'])

    return x

def remove_g(x: pd.DataFrame, fs: int, include_g:bool, g_cutoff: float, how: str = 'subtract') -> pd.DataFrame:
    if include_g:
        return x

    x = x.copy()

    features = x.columns[x.columns.str.contains("acc|norm")]
    for acc_feat in features:
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        if how == 'lowpass':
            grav = groups[acc_feat].transform(lambda g: butter_lowpass_filter(g, g_cutoff, fs / 2))
            x[acc_feat] = x[acc_feat] - grav

    return x

def produce(x: pd.DataFrame, features: List[str], placement: bool = False) -> pd.DataFrame:
    if features is None:
        return x

    x = x.copy()

    if placement:
        acc_features = x.columns[x.columns.str.contains('acc')]
        positions = list(set(acc_features.str.split('_').str[2]))
    else:
        positions = [None]

    for position in positions:
        if 'norm_xyz' in features:
            x = add_norm_xyz(x, position)

        if 'norm_xy' in features:
            x = add_norm_xy(x, position)

        if 'norm_xz' in features:
            x = add_norm_xz(x, position)

        if 'norm_yz' in features:
            x = add_norm_yz(x, position)

        if 'jerk' in features:
            x = add_jerk(x, position)

        if 'y_x_angle' in features:
            x = add_angle_y_x(x, position)

        if 'z_xy_angle' in features:
            x = add_angle_z_xy(x, position)

        if 'y_xz_angle' in features:
            x = add_angle_y_xz(x, position)

        if 'x_yz_angle' in features:
            x = add_angle_x_yz(x, position)

    return x

def smooth(x, filter_type, fs=0, cutoff=0):
    if filter_type is None:
        return x

    x = x.copy()

    if filter_type == 'lowpass':
        x = lowpass_smoothing(x, fs, cutoff)

    return x

def get_parameters(x: pd.DataFrame, labels: List[str], task: str, window: int = 0) -> pd.DataFrame:
    x = x.copy()

    if labels is None or task is None:
        return x

    if task == 'gait_phases':
        pm_cols = x.columns[x.columns.str.contains('stance|swing')]

        cols_to_drop = list(set(pm_cols) - set(labels))
        x = x.drop(cols_to_drop, axis=1)

        cols_to_drop = x.columns[x.columns.str.contains('HS|TO')]
        x = x.drop(cols_to_drop, axis=1)

    if task == 'gait_events':
        all_events = x.columns[x.columns.str.contains('HS|TO')]

        for in_event in labels:
            x[in_event + '_raw'] = x[in_event]
            x[in_event] = oversample_events(x, in_event, 'binary', window)

        cols_to_drop = list(set(all_events) - set(labels))
        x = x.drop(cols_to_drop, axis=1)

        cols_to_drop = x.columns[x.columns.str.contains('stance|swing')]
        x = x.drop(cols_to_drop, axis=1)

    return x

def augment(S: Tuple[np.ndarray, np.ndarray, np.ndarray], augmentations: List[str], channels: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if augmentations is None:
        return S

    X, Y, T = S
    X_aug = np.zeros(X.shape)

    xyz_channels = [channels['acc_x'],
                    channels['acc_y'],
                    channels['acc_z']]

    for aug in augmentations:
        if aug == 'rotate':
            xyz = X[..., xyz_channels]
            vrandom_rotate = np.vectorize(lambda w: random_rotate(w, around='x'), signature='(n,c)->(m,k)')
            X_aug[..., xyz_channels] = vrandom_rotate(xyz)

    S = X_aug, Y, T

    return S


def separate(S: Tuple[np.ndarray, np.ndarray, np.ndarray], by: Optional[str] = None) -> Dict:
    _, _, info = S
    indices = {}

    if by is None:
        indices[''] = np.arange(info.shape[0])

    elif by == 'subject':
        subjects = np.unique(info[:, 1]).astype(int)
        for subject in subjects:
            indices[subject] = np.argwhere(info[:, 1] == subject).flatten()

    elif by == 'activity':
        activities = np.unique(info[:, 2]).astype(int)
        for activity in activities:
            indices[activity] = np.argwhere(info[:, 2] == activity).flatten()

    return indices



