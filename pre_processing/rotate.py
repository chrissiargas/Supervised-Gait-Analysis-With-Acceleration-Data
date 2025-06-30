import scipy.optimize as opt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, norm
import numpy as np
import pandas as pd
from pre_processing.filters import butter_lowpass_filter
from pre_processing.irregularities import get_gravity
from rotation_utils import rotation_by_axis

def get_grav_comps(x: pd.DataFrame, fs: float) -> np.ndarray:
    x = x.values
    g = np.apply_along_axis(
        lambda ax: butter_lowpass_filter(ax, cutoff_freq=0.1, nyq_freq=fs / 2.),
        axis=0, arr=x)
    return g

def rotate_by_gravity(x: pd.DataFrame, features: pd.Index, fs: float, grav_axis: str = 'x') -> np.ndarray:
    g_comps = get_grav_comps(x[features], fs)
    g_comps = g_comps[~x.irregular.astype(bool)]

    g = np.mean(g_comps, axis=0)
    g = g / norm(g)

    if grav_axis == 'x':
        to = [1, 0, 0]
    if grav_axis == 'y':
        to = [0, 1, 0]
    if grav_axis == 'z':
        to = [0, 0, 1]

    target = np.array(to)

    k = np.cross(g, target)
    k /= norm(k)
    angle = np.arccos(g @ target)

    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R_opt = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    acc = x[features].values
    acc = acc @ R_opt.T

    return acc

def rotate_by_pca(x: pd.DataFrame, features: pd.Index, around: str = 'x') -> np.ndarray:
    acc = x.loc[~x.irregular.astype(bool), features].values

    if around == 'x':
        acc_to_rotate = acc[:, [1,2]]
    if around == 'y':
        acc_to_rotate = acc[:, [0,2]]
    if around == 'z':
        acc_to_rotate = acc[:, [0,1]]

    cov = np.cov(acc_to_rotate.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, 1]

    theta_pca = np.arctan2(pc1[1], pc1[0])
    R = rotation_by_axis(theta_pca, axis=around, degrees=False)

    acc = x[features].values
    acc = acc @ R.T

    return acc
