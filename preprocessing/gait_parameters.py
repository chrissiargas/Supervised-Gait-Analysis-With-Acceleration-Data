import pandas as pd
import numpy as np
from typing import List

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


