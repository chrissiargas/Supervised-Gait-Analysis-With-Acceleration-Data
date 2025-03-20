import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

def rotation_by_axis(angle, axis='z', degrees: bool = True):
    if degrees:
        angle_rad = np.deg2rad(angle)
    else:
        angle_rad = angle

    if axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return R

def inv_calibrate(x: pd.DataFrame, type: str) -> pd.DataFrame:
    acc_features = x.columns[x.columns.str.contains("acc")]

    a_free = x[acc_features].values
    g_world = np.array([0, 0, 9.81])

    if type == 'quat':
        quat_features = ['q1', 'qi', 'qj', 'qk']
        q = x[quat_features].values
        rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])

    elif type == 'euler':
        rot_features = ['course', 'pitch', 'roll']
        rot = x[rot_features].values
        rotation = Rotation.from_euler('ZYX', rot, degrees=True)

    a_raw = rotation.inv().apply(a_free + g_world)
    x[acc_features] = a_raw

    return x