import scipy.optimize as opt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, norm
import numpy as np
import pandas as pd
from preprocessing.filters import butter_lowpass_filter
from preprocessing.irregularities import get_gravity


def to_rotation_matrix(angle: float) -> np.ndarray:
    rot = np.eye(3)
    rot[0,0] = np.cos(angle)
    rot[0,2] = np.sin(angle)
    rot[2,0] = -np.sin(angle)
    rot[2,2] = np.cos(angle)

    return rot

def get_grav_comps(x: pd.DataFrame, fs: float) -> np.ndarray:
    x = x.values
    g = np.apply_along_axis(
        lambda ax: butter_lowpass_filter(ax, cutoff_freq=0.1, nyq_freq=fs / 2.),
        axis=0, arr=x)
    return g

def rotate_by_gravity(x: pd.DataFrame, fs: float) -> np.ndarray:
    features = x.columns[x.columns.str.contains('acc')]
    g_comps = get_grav_comps(x[features], fs)
    reg_g_comps = g_comps[~x.irregular.astype(bool)]

    g = np.mean(reg_g_comps, axis=0)
    g = g / norm(g)
    target = np.array([0, 1, 0])

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

def rotate_by_pca(x: pd.DataFrame) -> np.ndarray:
    features = x.columns[x.columns.str.contains('acc')]
    reg_acc = x.loc[~x.irregular.astype(bool), features].values

    xz = reg_acc[: , [0,2]]
    cov = np.cov(xz.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, 1]

    theta_pca = np.arctan2(pc1[1], pc1[0])
    R = to_rotation_matrix(theta_pca)

    acc = x[features].values
    acc = acc @ R.T

    return acc
