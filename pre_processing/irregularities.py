import numpy as np
import pandas as pd
from scipy.linalg import norm
import math
from pre_processing.filters import butter_lowpass_filter
from scipy.constants import g

def get_gravity(x: pd.DataFrame, fs: float) -> np.ndarray:
    x = x.values
    grav = np.mean(np.apply_along_axis(
        lambda ax: butter_lowpass_filter(ax, cutoff_freq=0.5, nyq_freq=fs/2),
        axis=0, arr=x), axis=0)

    return grav / norm(grav)

def get_angle(x: pd.DataFrame, grav: np.ndarray, fs: float) -> float:
    grav_ = get_gravity(x, fs)
    angle = np.arccos(grav @ grav_)
    angle = np.clip(angle, a_min=-1.0, a_max=1.0)

    return np.degrees(angle)

def check_rotation(x: pd.DataFrame, fs: float) -> pd.DataFrame:
    features = x.columns[x.columns.str.contains('acc')]
    grav = get_gravity(x[features], fs)

    angles = x.groupby('period_id').apply(lambda gr: get_angle(gr[features], grav, fs))
    threshold = np.mean(angles) + 1.5 * np.std(angles)
    ids = angles[angles > threshold].keys()

    return ids

