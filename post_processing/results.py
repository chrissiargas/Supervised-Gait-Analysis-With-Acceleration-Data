import numpy as np

from config.config_utils import config_edit
from pre_processing.building import builder
from model_utils.model import alligaitor
import pandas as pd
from post_processing.postprocess import find_peak_positions
from typing import Optional, List, Tuple
from rotation_utils import rotation_by_axis

ft_cols = [
    'acc_x',
    'acc_y',
    'acc_z'
]

id_cols = [
    'subject_id',
    'activity_id',
    'timestamp'
]

def split_with_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    res_df = df.copy()

    res_df['id'] = res_df[id_cols].astype(str).agg('-'.join, axis=1)
    data_df = res_df[['id', *ft_cols, *id_cols]]
    res_df = res_df.drop(columns=[*ft_cols, *id_cols])

    return res_df, data_df

def modify_args(task, targets, labels, head):
    config_edit('main_args', 'task', task)
    config_edit('main_args', 'targets', targets)
    config_edit('main_args', 'labels', labels)
    config_edit('main_args', 'head', head)

def get_preds(x: pd.DataFrame, label: str) -> np.ndarray:
    x = x.copy()

    event_indices = find_peak_positions(x[label + '_prob'].values)
    predictions = np.zeros(x.shape[0])
    predictions[event_indices] = 1.0

    return predictions

def filter_events(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()

    groups = df.groupby(['subject_id', 'activity_id'])
    preds = groups.apply(lambda gr: get_preds(gr, label))
    df[label + '_pred'] = np.concatenate(preds.values)

    return df

def reconstruct_y(task: str, targets: str, arch: str,
                  subjects: List[int], activities: Optional[List[int]] = None,
                  events: Optional[List] = None, filter: bool = True,
                  rotation: Optional[np.ndarray] = None) -> pd.DataFrame:

    data = builder(subjects=[2001])
    data(selected=[2001])

    if events:
        all_labels = [[event] for event in events]

    else:
        if task == 'gait_events':
            all_labels = [['LF_HS'], ['RF_HS'], ['LF_TO'], ['RF_TO']]
        elif task == 'gait_phases':
            all_labels = [['LF_stance'], ['RF_stance']]
        else:
            all_labels = []

    if targets == 'one':
        head = 'single'
    elif targets == 'all':
        head = 'temporal_single'
    else:
        head = ''

    results = None
    for labels in all_labels:
        result_cols = [*labels, *ft_cols, *id_cols]
        modify_args(task, targets, labels, head)

        model_args = f'{task}-{targets}-{str(labels)}'
        model_dir = f'archive/model_weights/{model_args}'
        model_file = '%s.weights.h5' % arch
        model_file = f'{model_dir}/{model_file}'

        model = alligaitor(data)
        model.compile()
        model.build_model(data.input_shape)
        model.load_weights(model_file)

        yy_ = data.get_yy_(model, subjects, activities, rotation=rotation)

        y_cols = yy_.columns[yy_.columns.str.contains('|'.join(result_cols))]
        yy_ = yy_[y_cols].copy()

        yy_, same = split_with_id(yy_)

        if results is None:
            results = yy_
        else:
            results = pd.merge(results, yy_, on='id', how='left')

        del model
        del yy_

    results = pd.merge(results, same, on='id', how='left')

    results.drop(columns=['id'])

    if filter:
        for labels in all_labels:
            for label in labels:
                results = filter_events(results, label)

    return results
