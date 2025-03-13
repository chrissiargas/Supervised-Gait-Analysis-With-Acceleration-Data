import numpy as np
import pandas as pd
from scipy.linalg import norm
import math
from preprocessing.filters import butter_lowpass_filter
from scipy.constants import g

def get_jerk(x: pd.DataFrame, by: str = '') -> float:
    x = x.values
    jerk = np.diff(x, axis=0)
    jerk = np.linalg.norm(jerk, axis=1)

    if by == 'min':
        q = np.quantile(jerk, 0.1)
        jerk = jerk[jerk < q].mean()
    if by == 'max':
        q = np.quantile(jerk, 0.9)
        jerk = jerk[jerk > q].mean()
    if by == 'mean':
        jerk = np.mean(jerk)

    return jerk

def check_artifacts(x: pd.DataFrame):
    features = x.columns[x.columns.str.contains('acc')]
    # jerk = np.diff(x[features].values, axis=0)
    # jerk = np.linalg.norm(jerk, axis=1)

    jerks = x.groupby('period_id').apply(lambda gr: get_jerk(gr[features], by='max'))
    threshold = np.mean(jerks) + 2 * np.std(jerks)
    ids = jerks[jerks > threshold].keys()

    return ids

def get_magnitude(x: pd.DataFrame, by: str = '') -> float:
    x = x.values
    magn = np.linalg.norm(x, axis = 1)

    if by == 'mean':
        magn = np.mean(magn)
    if by == 'var':
        magn = np.var(magn)
    if by == 'min':
        magn = np.min(magn)
    if by == 'max':
        magn = np.max(magn)

    return magn

def get_variance(x: pd.DataFrame, by: str = '') -> np.ndarray:
    x = x.values
    var = np.var(x, axis=0)

    if by == 'min':
        var = np.min(var)
    if by == 'max':
        var = np.max(var)
    if by == 'mean':
        var = np.mean(var)

    return var

def check_stationary(x: pd.DataFrame) -> pd.DataFrame:
    features = x.columns[x.columns.str.contains('acc')]
    vars = x.groupby('period_id').apply(lambda gr: get_variance(gr[features], by='max'))
    ids = vars[vars < 5].keys()

    return ids

def get_gravity(x: pd.DataFrame) -> np.ndarray:
    x = x.values
    grav = np.mean(np.apply_along_axis(
        lambda ax: butter_lowpass_filter(ax, cutoff_freq=0.5, nyq_freq=25.0),
        axis=0, arr=x), axis=0)

    return grav / norm(grav)

def get_angle(x: pd.DataFrame, grav: np.ndarray) -> float:
    grav_ = get_gravity(x)
    angle = np.arccos(grav @ grav_)
    angle = np.clip(angle, a_min=-1.0, a_max=1.0)

    return np.degrees(angle)

def check_rotation(x: pd.DataFrame) -> pd.DataFrame:
    features = x.columns[x.columns.str.contains('acc')]
    grav = get_gravity(x[features])

    angles = x.groupby('period_id').apply(lambda gr: get_angle(gr[features], grav))
    threshold = np.mean(angles) + 1.5 * np.std(angles)
    ids = angles[angles > threshold].keys()

    return ids

