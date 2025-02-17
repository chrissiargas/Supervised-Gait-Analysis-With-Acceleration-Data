import scipy.optimize as opt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, norm
import numpy as np
import pandas as pd
from preprocessing.filters import butter_lowpass_filter
from preprocessing.irregularities import get_gravity

def get_grav_comps(x: pd.DataFrame) -> np.ndarray:
    x = x.values
    g = np.apply_along_axis(
        lambda ax: butter_lowpass_filter(ax, cutoff_freq=0.1, nyq_freq=25.0),
        axis=0, arr=x)
    return g

def rotate_by_gravity(name, x: pd.DataFrame) -> np.ndarray:
    features = x.columns[x.columns.str.contains('acc')]
    g_comps = get_grav_comps(x[features])
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
    pc1 = eigvecs[:, 1]  # Max variance

    theta_pca = np.arctan2(pc1[1], pc1[0])
    R = to_rotation_matrix2(theta_pca)

    acc = x[features].values
    acc = acc @ R.T

    return acc

def to_rotation_matrix2(angle: float) -> np.ndarray:
    rot = np.eye(3)
    rot[0,0] = np.cos(angle)
    rot[0,2] = np.sin(angle)
    rot[2,0] = -np.sin(angle)
    rot[2,2] = np.cos(angle)

    return rot

def objective2(angle: float, x: np.ndarray) -> np.ndarray:
    rot = to_rotation_matrix2(angle)
    x_ = x @ rot.T

    energy_x = np.sum(x_[:, 0] ** 2)
    energy_z = np.sum(x_[:, 2] ** 2)
    loss = energy_z - energy_x

    return loss

def rotate_by_energy2(x: pd.DataFrame) -> np.ndarray:
    features = x.columns[x.columns.str.contains('acc')]
    acc = x[features].values
    reg_acc = x.loc[~x.irregular, features].values

    initial_theta = 0
    theta = opt.minimize(
        objective2,
        initial_theta,
        args = (reg_acc,),
        method = 'BFGS',
        options = {'disp': False}
    )
    R_opt = to_rotation_matrix2(theta.x)

    acc = acc @ R_opt.T

    return acc

def to_rotation_matrix3(vector: np.ndarray) -> np.ndarray:
    rot = R.from_rotvec(vector)
    return rot.as_matrix()

def objective3(vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    rot = to_rotation_matrix3(vector)
    x_ = x @ rot.T

    energy_y = np.sum(x_[:, 1] ** 2)
    energy_z = np.sum(x_[:, 2] ** 2)

    loss = energy_z - energy_y

    return loss

def rotate_by_energy3(x: pd.DataFrame) -> np.ndarray:
    features = x.columns[x.columns.str.contains('acc')]
    acc = x[features].values
    reg_acc = x.loc[~x.irregular, features].values

    initial_vector = np.zeros(3)
    vector = opt.minimize(
        objective3,
        initial_vector,
        args = (reg_acc,),
        method = 'BFGS',
        options = {'disp': False}
    )
    R_opt = to_rotation_matrix3(vector.x)
    acc = acc @ R_opt.T

    return acc