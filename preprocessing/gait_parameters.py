import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
from scipy import stats
from typing import Optional

def to_binary_window(df: pd.DataFrame, event: str = 'LF_HS', window_size: int = 5) -> np.ndarray:
    kernel = np.ones(2 * window_size + 1)
    smoothed = np.clip(
        np.convolve(df[event], kernel, mode='same'), 0, 1
    ).astype(int)

    return smoothed

def smooth_events(x: pd.DataFrame, event: str, how: Optional[str] = None) -> pd.DataFrame:
    if how == 'binary':
        groups = x.groupby(['dataset', 'subject_id', 'activity_id'])
        y = groups.apply(lambda g: pd.Series(to_binary_window(g, event, window_size=2), index=g.index))
        y = y.reset_index(level=['dataset', 'subject_id', 'activity_id'], drop=True)
    else:
        y = x[event]

    return y



