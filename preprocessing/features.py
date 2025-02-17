import pandas as pd
import numpy as np
from preprocessing.filters import butter_lowpass_filter, time_lowpass_filter


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

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    acc_dx = (groups['acc_x'].diff() * 1000. / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dy = (groups['acc_y'].diff() * 1000. / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dz = (groups['acc_z'].diff() * 1000. / groups['timestamp'].diff()).values[:, np.newaxis]

    acc_di = np.concatenate((acc_dx, acc_dy, acc_dz), axis=1)
    jerk = np.sqrt(np.sum(np.square(acc_di), axis=1))

    x['jerk'] = jerk
    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    if fillna:
        mask = groups.cumcount() == 0
        x['jerk'] = x['jerk'].where(~mask, 0)

    return x


def add_low(x: pd.DataFrame, fs: int, direction: str) -> pd.DataFrame:
    x = x.copy()
    x = x.interpolate()

    cutoff = 0.1

    if direction == 'x':
        acc = 'acc_x'
    elif direction == 'y':
        acc = 'acc_y'
    elif direction == 'z':
        acc = 'acc_z'
    elif direction == 'xyz':
        x['low_' + direction] = np.sqrt(x['low_x'] ** 2 + x['low_y'] ** 2 + x['low_z'] ** 2)
        return x
    elif direction == 'yz':
        acc = 'norm_yz'
    else:
        return None

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    low = groups[acc].transform(lambda g: time_lowpass_filter(g, cutoff, fs / 2))
    x['low_' + direction] = low

    return x

def add_angle_y_x(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['y_x_angle'] = np.arctan2(x['acc_y'], x['acc_x'])

    return x

def add_angle_z_xy(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['z_xy_angle'] = np.arctan2(x['acc_z'], np.sqrt(x['acc_x'] ** 2 + x['acc_y'] ** 2))

    return x

def add_angle_y_xz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['y_xz_angle'] = np.arctan2(x['acc_y'], np.sqrt(x['acc_x'] ** 2 + x['acc_z'] ** 2))

    return x

def add_angle_x_yz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['x_yz_angle'] = np.arctan2(x['acc_x'], np.sqrt(x['acc_y'] ** 2 + x['acc_z'] ** 2))

    return x

def add_angle_grav(x: pd.DataFrame, fs: int, g_cutoff: float):
    x = x.copy()

    features = x.columns[x.columns.str.contains("acc")]
    X = x[features].values
    G = None

    for acc_feat in features:
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        g = groups[acc_feat].transform(lambda g: butter_lowpass_filter(g, g_cutoff, fs / 2)).to_numpy()

        if G is None:
            G = g[:, np.newaxis]
        else:
            G = np.concatenate((G, g[:, np.newaxis]), axis=1)

    unit_X = (X-G) / np.linalg.norm(X-G, axis=1)[:, np.newaxis]
    unit_G = G / np.linalg.norm(G, axis=1)[:, np.newaxis]
    x['g_angle'] = np.arccos(np.sum(unit_X * unit_G, axis=1))

    return x

