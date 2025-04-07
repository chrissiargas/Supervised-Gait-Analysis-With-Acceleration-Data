import pandas as pd
import numpy as np
from pre_processing.filters import butter_lowpass_filter


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
