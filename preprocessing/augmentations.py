import itertools
import random

import numpy as np
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat


def add_noise(X, sigma=0.1):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise


def random_scale(X, sigma=0.1):
    X = X[np.newaxis, ...]
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    X = X * scaling_factor
    return X[0]


def random_rotate(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))


def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]
    y = axes[:, 1]
    z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    m = np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])
    matrix_transposed = np.transpose(m, axes=(2, 0, 1))
    return matrix_transposed


def time_mask(X, max_num=2, max_length=20):
    X = X[np.newaxis, ...]

    samples = X.shape[0]
    length = X.shape[1]

    for i in range(max_num):
        t = np.random.randint(0, max_length, size=samples)
        t0 = np.random.randint(0, length - t, size=samples)

        index_array = np.zeros((samples, length)) + np.arange(length)

        mask_start = (index_array >= t0[:, None])
        mask_stop = (index_array < (t0 + t)[:, None])
        mask = mask_start & mask_stop
        mask = (1 - mask)[..., None]

        X = X * mask

    return X[0]


def generate_random_curves(len, sigma=0.2, num_knots=4):
    xx = (np.arange(0, len, (len - 1) / (num_knots + 1))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(num_knots + 2))
    time_stamps = np.arange(len)
    cs_x = CubicSpline(xx[:], yy[:])
    return np.array([cs_x(time_stamps)]).transpose()


def distort_timesteps(len, sigma=0.2):
    tt = generate_random_curves(len, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    t_scale = [(len - 1) / tt_cum[-1]]
    tt_cum[:] = tt_cum[:] * t_scale
    return tt_cum


def DA_time_warp(len, sigma=0.2):
    tt_new = distort_timesteps(len, sigma)
    tt_new = np.squeeze(tt_new)
    x_range = np.arange(len)
    return tt_new, x_range


def time_warp(X):
    X = X[np.newaxis, ...]
    warped_X = np.zeros_like(X)

    for i, pair in enumerate(X[100:]):
        anchor, target = pair

        # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        # axs[0].plot(anchor[:, 0])
        # axs[1].plot(target[:, 0])

        tt_new, x_range = DA_time_warp(anchor.shape[0], 0.2)
        warped_pair = np.array([np.array(
            [np.interp(x_range, tt_new, sample[:, channel]) for channel in
             range(anchor.shape[1])]).transpose() for sample in pair])

        # warped_anchor, warped_target = warped_pair
        # axs[0].plot(warped_anchor[:, 0], '--')
        # axs[1].plot(warped_target[:, 0], '--')
        # plt.show()

        warped_X[i, ...] = warped_pair

    return warped_X[0]
