from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd

def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = filtfilt(b, a, data)
    return y

def median_filter(signal: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(signal, np.ones(w), 'same') / w

def median_smoothing(x: pd.DataFrame, w: int) -> pd.DataFrame:
    x = x.copy()

    features =  x.columns[x.columns.str.contains("acc|jerk|grav|norm")]

    groups = x.groupby(['subject', 'activity'])

    for feature in features:
        smoothed_ft = groups[feature].transform(lambda g: median_filter(g, w))
        x[feature] = smoothed_ft

    return x