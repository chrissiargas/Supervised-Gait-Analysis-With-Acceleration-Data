from scipy.signal import butter, filtfilt, buttord
import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter

def butter_design(nyq_freq):
    n, wn = buttord(0.7 / nyq_freq, 0.5 / nyq_freq, 3,  10)
    return n, wn

def butter_lowpass2(nyq_freq):
    n, wn = butter_design(nyq_freq)
    b, a = butter(n, wn, btype='lowpass')
    return b, a

def butter_lowpass_filter2(data, nyq_freq):
    b, a = butter_lowpass2(nyq_freq)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = butter(order, normal_cutoff, btype='highpass')
    return b, a

def butter_highpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_highpass(cutoff_freq, nyq_freq, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq: float = 0.1, nyq_freq: float = 25., order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = filtfilt(b, a, data)
    return y

def time_lowpass_filter(data, cutoff_freq: float = 0.5, nyq_freq: float = 25.):
    normal_cuttoff = float(cutoff_freq) / nyq_freq
    response = firwin(100, normal_cuttoff, pass_zero='lowpass')
    y = lfilter(response, 1, data)
    return y

def median_filter(signal: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(signal, np.ones(w), 'same') / w

def median_smoothing(x: pd.DataFrame, w: int) -> pd.DataFrame:
    x = x.copy()

    features =  x.columns[x.columns.str.contains("acc")]

    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])

    for feature in features:
        smoothed_ft = groups[feature].transform(lambda g: median_filter(g, w))
        x[feature] = smoothed_ft

    return x

def lowpass_smoothing(x: pd.DataFrame, fs: int, cutoff: float) -> pd.DataFrame:
    x = x.copy()

    features = x.columns[x.columns.str.contains("acc|norm")]
    for acc_feat in features:
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        low = groups[acc_feat].transform(lambda g: butter_lowpass_filter(g, cutoff, fs / 2))
        x[acc_feat] = low

    return x