import pandas as pd
import numpy as np
from preprocessing.filters import butter_lowpass_filter

def add_norm_xyz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_xyz'] = np.sqrt(x['acc_x'] ** 2 + x['acc_y'] ** 2 + x['acc_z'] ** 2)

    return x


def add_norm_xy(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_xy'] = np.sqrt(x['acc_x'] ** 2 + x['acc_y'] ** 2)

    return x


def add_norm_yz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_yz'] = np.sqrt(x['acc_y'] ** 2 + x['acc_z'] ** 2)

    return x


def add_norm_xz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_xz'] = np.sqrt(x['acc_x'] ** 2 + x['acc_z'] ** 2)

    return x


def add_jerk(x: pd.DataFrame, fillna: bool = True) -> pd.DataFrame:
    x = x.copy()

    groups = x.groupby(['subject', 'activity'])
    acc_dx = (groups['acc_x'].diff() / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dy = (groups['acc_y'].diff() / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dz = (groups['acc_z'].diff() / groups['timestamp'].diff()).values[:, np.newaxis]

    acc_di = np.concatenate((acc_dx, acc_dy, acc_dz), axis=1)
    jerk = np.sqrt(np.sum(np.square(acc_di), axis=1))

    x['jerk'] = jerk
    groups = x.groupby(['subject', 'activity'])

    if fillna:
        mask = groups.cumcount() == 0
        x['jerk'] = x['jerk'].where(~mask, 0)

    return x


def add_grav(x: pd.DataFrame, fs: int, direction: str) -> pd.DataFrame:
    x = x.copy()
    x = x.interpolate()

    cutoff = 1.

    if direction == 'x':
        acc = 'acc_x'
    elif direction == 'y':
        acc = 'acc_y'
    elif direction == 'z':
        acc = 'acc_z'
    else:
        return None

    groups = x.groupby(['subject', 'activity'])
    low = groups[acc].transform(lambda g: butter_lowpass_filter(g, cutoff, fs / 2))
    x['grav_' + direction] = low

    return x
