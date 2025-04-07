from scipy.signal import butter, filtfilt, buttord
import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter

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

def lowpass_smoothing(x: pd.DataFrame, fs: int, cutoff: float) -> pd.DataFrame:
    x = x.copy()

    features = x.columns[x.columns.str.contains("acc|norm")]
    for acc_feat in features:
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        low = groups[acc_feat].transform(lambda g: butter_lowpass_filter(g, cutoff, fs / 2))
        x[acc_feat] = low

    return x

