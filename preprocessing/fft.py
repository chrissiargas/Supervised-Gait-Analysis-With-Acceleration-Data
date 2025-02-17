import numpy as np
from keras.src.layers import concatenate
from scipy.signal import spectrogram
from scipy.interpolate import interp2d
import pandas as pd
from typing import Dict, Tuple
import math

def resize(sg: np.ndarray, freq: np.ndarray, time: np.ndarray, height: int = 48, f_interp: str = 'log'):
    width = time.shape[0]

    out_sg = np.zeros((height, width), dtype=np.float32)

    if f_interp == 'log':
        log_f = np.log(freq + freq[1])
        log_f_normalized = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
        f = height * log_f_normalized

    else:
        f_normalized = (freq - freq[0]) / (freq[-1] - freq[0])
        f = height * f_normalized

    f_i = np.arange(height)
    t_i = np.arange(width)

    spectrogram_fn = interp2d(time, f, sg, copy=False)
    out_sg[...] = spectrogram_fn(t_i, f_i)

    return out_sg

def to_spectrogram(signal: np.ndarray, fs: int, nperseg: int, noverlap: int,
                   resizing: bool = False, log_power: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, sg = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if resizing:
        sg = resize(sg, f, t)
    if log_power:
        np.log(sg + 1e-10, dtype=np.float32, out=sg)

    return sg[np.newaxis, 1:, :], f[1:], t

def get_spectrograms(x: pd.DataFrame, fs: int, nperseg: int, noverlap: int,
                     resize: bool = False, log_power: bool = False) -> Tuple[Dict, np.ndarray, np.ndarray]:
    x = x.copy()
    features = x.columns[x.columns.str.contains("acc")]

    spectrograms = {}
    subs = x['subject'].unique().tolist()
    for sub in subs:
        sub_x = x[x['subject'] == sub]
        spectrograms[sub] = {}
        for act in sub_x['activity_id'].unique():
            sub_act_x = sub_x[sub_x['activity_id'] == act]
            for f, feature in enumerate(features):
                signal = sub_act_x[feature]
                sg, _, _ = to_spectrogram(signal, fs, nperseg, noverlap, resize, log_power)
                if f == 0:
                    spectrograms[sub][act] = sg
                else:
                    spectrograms[sub][act] = np.concatenate((spectrograms[sub][act], sg), axis=0)

    _, f, t = to_spectrogram(signal, fs, nperseg, noverlap, resize, log_power)
    return spectrograms, f, t

def slice(sgs: Dict, nperseg: int, nstride: int, length: int = 0) -> Dict:
    sgs = sgs.copy()

    if length == 0:
        return sgs

    sg_length = int(max((length - nperseg // 2) // nstride, 0))
    sg_start = sg_length
    sg_end = None if sg_length == 0 else -sg_length
    
    for sub in sgs.keys():
        sub_sgs = sgs[sub]
        for act in sub_sgs.keys():
            act_sgs = sub_sgs[act]
            sgs[sub][act] = act_sgs[..., sg_start:sg_end]

    return sgs

def time_split(sgs: np.ndarray, split_type: str, hold_out: float) -> Tuple[np.ndarray, np.ndarray]:
    test_size = int(sgs.shape[2] * hold_out)
    train_size = sgs.shape[2] - test_size

    if 'start' in split_type:
        test, train = sgs[..., :test_size], sgs[..., test_size:]
    elif 'end' in split_type:
        train, test = sgs[..., :train_size], sgs[..., train_size:]
    else:
        train, test = np.ndarray([]), np.ndarray([])

    return train, test

def to_windows(X: np.ndarray, length: int, step: int) -> np.ndarray:
    n_windows = math.ceil((X.shape[2] - length + 1) / step)
    n_windows = max(0, n_windows)

    X = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, X.shape[0], X.shape[1], length),
        strides=(step * X.strides[2], X.strides[0], X.strides[1], X.strides[2]))

    return X

def segment(sgs: Dict, sizes: Dict, length: int, step: int, nstride: int) -> np.ndarray:
    sgs = sgs.copy()
    out_sgs = None
    length =int(length // nstride)
    step = int(step // nstride)

    for sub in sgs.keys():
        sub_sgs = sgs[sub]
        sub_sizes = sizes[sub]
        for act in sub_sgs.keys():
            act_sgs = sub_sgs[act]
            size = int(sub_sizes[act])
            segmented_sgs = to_windows(act_sgs, length, step)
            if out_sgs is None:
                out_sgs = segmented_sgs[:size]
            else:
                out_sgs = np.concatenate((out_sgs, segmented_sgs[:size]), axis=0)

    return out_sgs


