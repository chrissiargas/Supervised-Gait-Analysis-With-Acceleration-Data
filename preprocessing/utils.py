from lib2to3.pgen2.tokenize import group
from sys import int_info

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocessing.features import (add_norm_xy, add_norm_xz, add_norm_xyz, add_norm_yz,
                                    add_jerk, add_low, add_angle_grav,
                                    add_angle_y_xz, add_angle_z_xy, add_angle_y_x, add_angle_x_yz)
from preprocessing.filters import (butter_lowpass_filter, median_smoothing, lowpass_smoothing,
                                   butter_lowpass_filter2, butter_highpass_filter, time_lowpass_filter)
from preprocessing.gait_parameters import add_parameter
from preprocessing.rotate import rotate_by_gravity, rotate_by_energy2, rotate_by_energy3, rotate_by_pca
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

def trim(x: pd.DataFrame, length: int = 0, div: int = 0) -> pd.DataFrame:
    x = x.copy()

    if div != 0:
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'], group_keys=False)
        x = groups.apply(lambda g: g.iloc[: len(g) - (len(g) % div)])
        return x

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

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    period_id = groups.apply(lambda g: get_period_id(g, period))
    x['period_id'] = np.concatenate(period_id.values)

    for ds, ds_df in x.groupby('dataset'):
        for sub, sub_df in ds_df.groupby('subject_id'):
            for act, act_df in sub_df.groupby('activity_id'):
                flags = []
                for check in checks:
                    flag = []
                    if check == 'artifacts':
                        flag = check_artifacts(act_df)
                    if check == 'stationary':
                        flag = check_stationary(act_df)
                    elif check == 'rotation':
                        flag = check_rotation(act_df)
                    flags.append(set(flag))

                all_flags = set.union(*flags)
                x.loc[(x.dataset == ds) &
                       (x.subject_id == sub) &
                       (x.activity_id == act) &
                       (x.period_id.isin(all_flags)),
                        'irregular'] = 1

                # features = x.columns[x.columns.str.contains('acc')]
                # act_df['acc_xyz'] = np.linalg.norm(act_df[features].values, axis=1)
                #
                # features = act_df.columns[act_df.columns.str.contains('acc')]
                # name = (str(ds) + '-' + str(sub) + '-' + str(act) + '-')
                # for f, flag in enumerate(all_flags):
                #     window = act_df[act_df.period_id == flag]
                #     window = window[features]
                #
                #     fig, axs = plt.subplots(1, sharex=True, figsize=(40, 15))
                #     axs.plot(window, linewidth=1, label=features)
                #     plt.legend()
                #     filepath = os.path.join(figpath, name + str(f) + ".png")
                #     plt.savefig(filepath, format="png", bbox_inches="tight")
                #     plt.close()

    return x


def orient(x: pd.DataFrame, how: Optional[str] = None, dim: str = '3d') -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()

    features = x.columns[x.columns.str.contains('acc')]
    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    period_id = groups.apply(lambda g: get_period_id(g, period=2000))
    x['rotation_period'] = np.concatenate(period_id.values)

    groups = x.groupby(['dataset', 'subject_id', 'activity_id', 'rotation_period'])

    if 'gravity' in how:
        rotated = groups.apply(lambda gr: rotate_by_gravity(gr.name, gr))
        x[features] = np.concatenate(rotated.values)

    if 'pca' in how:
        rotated = groups.apply(lambda gr: rotate_by_pca(gr))
        x[features] = np.concatenate(rotated.values)

    if 'energy' in how:
        if dim == '2d':
            rotated = groups.apply(lambda gr: rotate_by_energy2(gr))
        if dim == '3d':
            rotated = groups.apply(lambda gr: rotate_by_energy3(gr))

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
        elif how == 'highpass':
            x[acc_feat] = groups[acc_feat].transform(lambda g: butter_highpass_filter(g, g_cutoff, fs / 2))

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

    if 'low_x' in features:
        x = add_low(x, fs, 'x')

    if 'low_y' in features:
        x = add_low(x, fs, 'y')

    if 'low_z' in features:
        x = add_low(x, fs, 'z')

    if 'low_xyz' in features:
        x = add_low(x, fs, direction='xyz')

    if 'low_yz' in features:
        x = add_low(x, fs, direction='yz')

    if 'y_x_angle' in features:
        x = add_angle_y_x(x)

    if 'z_xy_angle' in features:
        x = add_angle_z_xy(x)

    if 'y_xz_angle' in features:
        x = add_angle_y_xz(x)

    if 'x_yz_angle' in features:
        x = add_angle_x_yz(x)

    if 'g_angle' in features:
        x = add_angle_grav(x, fs, g_cutoff=0.5)

    return x

def smooth(x, filter_type, w=0, fs=0, cutoff=0):
    if filter_type is None:
        return x

    x = x.copy()

    if filter_type == 'median':
        x = median_smoothing(x, w)

    if filter_type == 'lowpass':
        x = lowpass_smoothing(x, fs, cutoff)

    return x

def rescale(x: pd.DataFrame, how: str = 'standard') -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()

    features = x.columns[x.columns.str.contains("acc|norm|jerk|low|angle|gyr")]

    if how == 'min-max':
        rescaler = MinMaxScaler()
    elif how == 'standard':
        rescaler = StandardScaler()

    x[features] = rescaler.fit_transform(x[features].values)

    return x

def get_parameters(x: pd.DataFrame, parameters: List[str], calculate: bool = True) -> pd.DataFrame:
    x = x.copy()

    if parameters is None:
        return x

    if calculate:
        pm_cols = x.columns[x.columns.str.contains('stance')]
        x = x.drop(pm_cols, axis=1)

        if 'LF_stance' in parameters:
            x = add_parameter(x, 'LF_HS', 'LF_TO', 'LF_stance_time')

        if 'RF_stance' in parameters:
            x = add_parameter(x, 'RF_HS', 'RF_TO', 'RF_stance_time')

        if 'LF_swing' in parameters:
            x = add_parameter(x, 'LF_TO', 'LF_HS', 'LF_swing_time')

        if 'RF_swing' in parameters:
            x = add_parameter(x, 'RF_TO', 'RF_HS', 'RF_swing_time')

        if 'step' in parameters:
            x = add_parameter(x, 'RF_HS', 'RF_HS', 'step_time')

        if 'LF_double_support' in parameters:
            x = add_parameter(x, 'RF_HS', 'LF_TO', 'LF_double_support_time')

        if 'RF_double_support' in parameters:
            x = add_parameter(x, 'LF_HS', 'RF_TO', 'RF_double_support_time')

    else:
        pm_cols = x.columns[x.columns.str.contains('stance')]
        cols_to_drop = list(set(pm_cols) - set(parameters))
        x = x.drop(cols_to_drop, axis=1)

    return x

def to_categorical(x: pd.DataFrame, drop: bool = False) -> pd.DataFrame:
    x = x.copy()

    x['subject_id'] = x['subject'].astype(int)
    x['activity_id'] = x['activity'].astype(int)

    if drop:
        x = x.drop(['subject', 'activity'], axis=1)

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



