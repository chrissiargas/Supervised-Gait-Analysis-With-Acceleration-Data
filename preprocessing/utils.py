
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocessing.features import (add_norm_xy, add_norm_xz, add_norm_xyz, add_norm_yz,
                                    add_jerk, add_angle_y_xz, add_angle_z_xy, add_angle_y_x,
                                    add_angle_x_yz)
from preprocessing.filters import (butter_lowpass_filter, lowpass_smoothing, butter_highpass_filter)
from preprocessing.gait_parameters import oversample_events
from preprocessing.rotate import rotate_by_gravity, rotate_by_pca
import matplotlib.pyplot as plt
import os
from preprocessing.irregularities import *

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
           how: Optional[str] = None) -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()

    features = x.columns[x.columns.str.contains('acc')]
    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    period_id = groups.apply(lambda g: get_period_id(g, period=2000))
    x['rotation_period'] = np.concatenate(period_id.values)

    groups = x.groupby(['dataset', 'subject_id', 'activity_id', 'rotation_period'])

    if 'gravity' in how:
        rotated = groups.apply(lambda gr: rotate_by_gravity(gr, fs))
        x[features] = np.concatenate(rotated.values)

    if 'pca' in how:
        rotated = groups.apply(lambda gr: rotate_by_pca(gr))
        x[features] = np.concatenate(rotated.values)

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

def produce(x: pd.DataFrame, features: List[str], fs: int) -> pd.DataFrame:
    if features is None:
        return x

    x = x.copy()

    if 'norm_xyz' in features:
        x = add_norm_xyz(x)

    if 'norm_xy' in features:
        x = add_norm_xy(x)

    if 'norm_xz' in features:
        x = add_norm_xz(x)

    if 'norm_yz' in features:
        x = add_norm_yz(x)

    if 'jerk' in features:
        x = add_jerk(x)

    if 'y_x_angle' in features:
        x = add_angle_y_x(x)

    if 'z_xy_angle' in features:
        x = add_angle_z_xy(x)

    if 'y_xz_angle' in features:
        x = add_angle_y_xz(x)

    if 'x_yz_angle' in features:
        x = add_angle_x_yz(x)

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



