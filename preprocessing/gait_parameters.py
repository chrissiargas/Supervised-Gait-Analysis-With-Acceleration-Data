import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
from scipy import stats
from typing import Optional

def calc_parameter(timeseries: pd.DataFrame, event1: str, event2: str) -> List[float]:
    event1_indices = np.where(timeseries[event1] == 1)[0]
    event2_indices = np.where(timeseries[event2] == 1)[0]
    parameter_vals = [None] * len(timeseries)

    for event1_index in event1_indices:
        next_indices = event2_indices[event2_indices > event1_index]
        if len(next_indices):
            event2_index = next_indices[0]
            duration = event2_index - event1_index
            parameter_vals[event1_index+1:event2_index+1] = [duration] * (event2_index - event1_index)

    return parameter_vals

def add_parameter(x: pd.DataFrame, event1: str, event2: str, parameter: str) -> pd.DataFrame:
    groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
    param_vals = groups.apply(lambda g: pd.Series(calc_parameter(g, event1, event2), index=g.index))
    x[parameter] = param_vals.reset_index(level=['dataset', 'subject_id', 'activity_id'], drop=True)

    return x


def to_binary_window(df: pd.DataFrame, event: str = 'LF_HS', window_size: int = 5) -> np.ndarray:
    kernel = np.ones(2 * window_size + 1)
    smoothed = np.clip(
        np.convolve(df[event], kernel, mode='same'), 0, 1
    ).astype(int)

    return smoothed


def to_gaussian_window(df, event_col='LF_HS', std=5):
    indices = df.index.values
    event_times = indices[df[event_col] == 1]
    smoothed = np.zeros(df.shape[0], dtype=float)

    for t in event_times:
        weights = stats.norm.pdf(indices, loc=t, scale=std)
        smoothed += weights

    return smoothed


# Example usage
def smooth_events(x: pd.DataFrame, event: str, how: Optional[str] = None) -> pd.DataFrame:
    if how == 'binary':
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        y = groups.apply(lambda g: pd.Series(to_binary_window(g, event, window_size=2), index=g.index))
        y = y.reset_index(level=['dataset', 'subject_id', 'activity_id'], drop=True)
    elif how == 'prob':
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        y = groups.apply(lambda g: pd.Series(to_gaussian_window(g, event, std=2), index=g.index))
        y = y.reset_index(level=['dataset', 'subject_id', 'activity_id'], drop=True)
    else:
        y = x[event]

    return y



