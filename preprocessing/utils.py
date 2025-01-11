import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math

from preprocessing.features import (add_norm_xy, add_norm_xz, add_norm_xyz, add_norm_yz,
                                    add_jerk, add_low, add_azimuth, add_elevation)
from preprocessing.filters import (butter_lowpass_filter, median_smoothing, lowpass_smoothing,
                                   butter_lowpass_filter2, butter_highpass_filter)
from preprocessing.info import info
from preprocessing.gait_parameters import add_parameter


def fill_nan(x: pd.DataFrame, features: List[str], how: str) -> pd.DataFrame:
    x = x.copy()

    x['is_NaN'] = x[features].isnull().any(axis='columns')
    groups = x.groupby(['subject', 'activity'])

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

    return x

def impute(x: pd.DataFrame, how: str) -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()
    features = x.columns[x.columns.str.contains("acc")]
    x = fill_nan(x, features, how)

    return x

def remove_g(x: pd.DataFrame, fs: int, include_g:bool) -> pd.DataFrame:
    if include_g:
        return x

    x = x.copy()
    features = x.columns[x.columns.str.contains("acc")]
    cutoff = 0.5

    for acc_feat in features:
        groups = x.groupby(['subject', 'activity'])
        grav = groups[acc_feat].transform(lambda g: butter_lowpass_filter(g, cutoff, fs / 2))
        x[acc_feat] = x[acc_feat] - grav

    return x

def produce(x: pd.DataFrame, features: List[str], fs: int) -> pd.DataFrame:
    x = x.copy()

    if features is None:
        return x

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

    if 'azimuth' in features:
        x = add_azimuth(x)

    if 'elevation' in features:
        x = add_elevation(x)

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

    features = x.columns[x.columns.str.contains("acc|norm|jerk|low|angle")]

    if how == 'min-max':
        rescaler = MinMaxScaler()
    elif how == 'standard':
        rescaler = StandardScaler()

    x[features] = rescaler.fit_transform(x[features].values)

    return x

def get_parameters(x: pd.DataFrame, parameters: List[str]) -> pd.DataFrame:
    x = x.copy()

    if parameters is None:
        return x

    if 'left_stance' in parameters:
        x = add_parameter(x, 'LF_HS', 'LF_TO', 'left_stance_time')

    if 'right_stance' in parameters:
        x = add_parameter(x, 'RF_HS', 'RF_TO', 'right_stance_time')

    if 'left_swing' in parameters:
        x = add_parameter(x, 'LF_TO', 'LF_HS', 'left_swing_time')

    if 'right_swing' in parameters:
        x = add_parameter(x, 'RF_TO', 'RF_HS', 'right_swing_time')

    if 'step' in parameters:
        x = add_parameter(x, 'RF_HS', 'RF_HS', 'step_time')

    if 'left_double_support' in parameters:
        x = add_parameter(x, 'RF_HS', 'LF_TO', 'left_double_support_time')

    if 'right_double_support' in parameters:
        x = add_parameter(x, 'LF_HS', 'RF_TO', 'right_double_support_time')

    return x