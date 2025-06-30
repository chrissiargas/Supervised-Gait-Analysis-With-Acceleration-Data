import pandas as pd
import numpy as np
from typing import Optional


def add_norm_xyz(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_y = 'acc_y_' + position
        acc_z = 'acc_z_' + position
        norm_xyz = 'norm_xyz_' + position
    else:
        acc_x = 'acc_x'
        acc_y = 'acc_y'
        acc_z = 'acc_z'
        norm_xyz = 'norm_xyz'

    x[norm_xyz] = np.sqrt(x[acc_x] ** 2 + x[acc_y] ** 2 + x[acc_z] ** 2)

    return x


def add_norm_xy(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_y = 'acc_y_' + position
        norm_xy = 'norm_xy_' + position
    else:
        acc_x = 'acc_x'
        acc_y = 'acc_y'
        norm_xy = 'norm_xy'

    x[norm_xy] = np.sqrt(x[acc_x] ** 2 + x[acc_y] ** 2)

    return x


def add_norm_yz(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_y = 'acc_y_' + position
        acc_z = 'acc_z_' + position
        norm_yz = 'norm_yz_' + position
    else:
        acc_y = 'acc_y'
        acc_z = 'acc_z'
        norm_yz = 'norm_yz'

    x[norm_yz] = np.sqrt(x[acc_y] ** 2 + x[acc_z] ** 2)

    return x


def add_norm_xz(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_z = 'acc_z_' + position
        norm_xz = 'norm_xz_' + position
    else:
        acc_x = 'acc_x'
        acc_z = 'acc_z'
        norm_xz = 'norm_xz'

    x[norm_xz] = np.sqrt(x[acc_x] ** 2 + x[acc_z] ** 2)

    return x


def add_jerk(x: pd.DataFrame, position: Optional[str] = None, fillna: bool = True) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_y = 'acc_y_' + position
        acc_z = 'acc_z_' + position
        jerk = 'jerk_' + position
    else:
        acc_x = 'acc_x'
        acc_y = 'acc_y'
        acc_z = 'acc_z'
        jerk = 'jerk'

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    acc_dx = (groups[acc_x].diff() * 1000. / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dy = (groups[acc_y].diff() * 1000. / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dz = (groups[acc_z].diff() * 1000. / groups['timestamp'].diff()).values[:, np.newaxis]

    acc_di = np.concatenate((acc_dx, acc_dy, acc_dz), axis=1)
    d_acc = np.sqrt(np.sum(np.square(acc_di), axis=1))

    x[jerk] = d_acc
    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    if fillna:
        mask = groups.cumcount() == 0
        x[jerk] = x[jerk].where(~mask, 0)

    return x

def add_angle_y_x(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_y = 'acc_y_' + position
        y_x_angle = 'y_x_angle_' + position
    else:
        acc_x = 'acc_x'
        acc_y = 'acc_y'
        y_x_angle = 'y_x_angle'

    x[y_x_angle] = np.arctan2(x[acc_y], x[acc_x])

    return x

def add_angle_z_xy(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_y = 'acc_y_' + position
        acc_z = 'acc_z_' + position
        z_xy_angle = 'z_xy_angle_' + position
    else:
        acc_x = 'acc_x'
        acc_y = 'acc_y'
        acc_z = 'acc_z'
        z_xy_angle = 'z_xy_angle'

    x[z_xy_angle] = np.arctan2(x[acc_z], np.sqrt(x[acc_x] ** 2 + x[acc_y] ** 2))

    return x

def add_angle_y_xz(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_y = 'acc_y_' + position
        acc_z = 'acc_z_' + position
        y_xz_angle = 'y_xz_angle' + position
    else:
        acc_x = 'acc_x'
        acc_y = 'acc_y'
        acc_z = 'acc_z'
        y_xz_angle = 'y_xz_angle'

    x[y_xz_angle] = np.arctan2(x[acc_y], np.sqrt(x[acc_x] ** 2 + x[acc_z] ** 2))

    return x

def add_angle_x_yz(x: pd.DataFrame, position: Optional[str] = None) -> pd.DataFrame:
    x = x.copy()

    if position is not None:
        acc_x = 'acc_x_' + position
        acc_y = 'acc_y_' + position
        acc_z = 'acc_z_' + position
        x_yz_angle = 'x_yz_angle_' + position
    else:
        acc_x = 'acc_x'
        acc_y = 'acc_y'
        acc_z = 'acc_z'
        x_yz_angle = 'x_yz_angle'

    x[x_yz_angle] = np.arctan2(x[acc_x], np.sqrt(x[acc_y] ** 2 + x[acc_z] ** 2))

    return x
