import pandas as pd
import numpy as np
from typing import Dict


def get_parameters(x: pd.DataFrame, tolerance: int = 5) -> pd.DataFrame:
    matched_events = {
        'LF': match_foot_events(x, 'LF', tolerance),
        'RF': match_foot_events(x, 'RF', tolerance),
    }

    params = None
    for foot in matched_events.keys():
        foot_params = calculate_foot_parameters(matched_events[foot])
        foot_params['foot'] = foot

        if params is None:
            params = foot_params
        else:
            params = pd.concat([params, foot_params], axis=0)

    return params

def get_phase_times(matches: Dict, event1: str, event2: str) -> pd.DataFrame:
    phase_times = []

    event1_indices = matches[event1]
    event2_indices = matches[event2]

    for event1_index in event1_indices:
        next_indices = event2_indices[event2_indices[:, 0] > event1_index[0]]
        if len(next_indices):
            event2_index = next_indices[0]

            true_phase_time = event2_index[0] - event1_index[0]
            pred_phase_time = event2_index[1] - event1_index[1]

            phase_times.append({
                'true_value': true_phase_time,
                'pred_value': pred_phase_time,
                'error': true_phase_time - pred_phase_time
            })

    return pd.DataFrame.from_dict(phase_times)

def calculate_foot_parameters(matched_events: Dict):
    stance_times = get_phase_times(matched_events, 'HS', 'TO')
    stance_times['parameter'] = 'stance_time'
    swing_times = get_phase_times(matched_events, 'TO', 'HS')
    swing_times['parameter'] = 'swing_time'
    stride_times = get_phase_times(matched_events, 'HS', 'HS')
    stride_times['parameter'] = 'stride_time'

    times = pd.concat([stance_times, swing_times, stride_times], axis=0)
    return times

def match_single_events(x: pd.DataFrame, pred_label: str, true_label: str, tolerance: int = 5):
    true_times = x[x[true_label] == 1].index
    pred_times = x[x[pred_label] == 1].index

    matches = np.zeros(shape=(len(true_times), 2))
    for i, t in enumerate(true_times):
        closest = pred_times[np.abs(pred_times - t).argmin()]
        if np.abs(closest - t) <= tolerance:
            matches[i] = [t, closest]
        else:
            matches[i] = [t, np.nan]

    return matches


def match_foot_events(x: pd.DataFrame, foot: str, tolerance: int = 5):
    matched = {
        'HS': match_single_events(x, f"{foot}_HS_pred", f"{foot}_HS_raw", tolerance),
        'TO': match_single_events(x, f"{foot}_TO_pred", f"{foot}_TO_raw", tolerance)
    }
    return matched


def get_conf(x: pd.DataFrame, pred_label, true_label, n=1):
    pred_mask = x[pred_label] == 1
    true_mask = x[true_label] == 1
    nb_pred_mask = pred_mask.rolling(2 * n + 1, center=True, min_periods=1).max()
    nb_true_mask = true_mask.rolling(2 * n + 1, center=True, min_periods=1).max()

    true_positives = pred_mask & nb_true_mask
    n_tp = true_positives.sum()

    false_positives = pred_mask & nb_true_mask.replace({0: 1, 1: 0})
    n_fp = false_positives.sum()

    false_negatives = true_mask & nb_pred_mask.replace({0: 1, 1: 0})
    n_fn = false_negatives.sum()

    return n_tp, n_fp, n_fn

def get_scores(tp, fp, fn):
    if tp != 0 or fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp != 0 or fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return precision, recall, f1_score

def get_min_error(w, n):
    w = w.to_numpy()
    e = (n - np.argwhere(w).reshape((-1))).min() if len(np.argwhere(w)) else -1
    return e

def get_time_error(x: pd.DataFrame, pred_label, true_label, n=1, how='min'):
    pred_mask = x[pred_label] == 1
    true_mask = x[true_label] == 1
    nb_true_mask = true_mask.rolling(2 * n + 1, center=True, min_periods=1).max()
    true_positives = (pred_mask & nb_true_mask).astype(bool)

    dists = true_mask.rolling(2 * n + 1, center=True, min_periods=1).apply(lambda w: get_min_error(w, n))
    tp_dists = dists[true_positives]

    return tp_dists





