import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Dict

import preprocessing.fft as fft

def split(x: pd.DataFrame, split_type: str, hold_out: Union[List, int, float, str],
          seed: Optional[int] = None, sgs: Optional[Dict] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    x = x.copy()
    train_sgs, test_sgs = (sgs.copy(), sgs.copy()) if sgs is not None else (None, None)

    if split_type == 'dataset':
        if hold_out == 'marea':
            train = x[x['dataset'] == 'nonan']
            test = x[x['dataset'] == 'marea']

        elif hold_out == 'nonan':
            train = x[x['dataset'] == 'marea']
            test = x[x['dataset'] == 'nonan']

    elif split_type == 'loso_marea':
        subs = x[x['dataset'] == 'marea']['subject_id'].unique().tolist()
        rng = np.random.default_rng(seed=seed)

        if isinstance(hold_out, float):
            r = int(len(subs) * hold_out)
            test_subs = rng.choice(subs, r, replace=False)
        elif isinstance(hold_out, int):
            r = hold_out
            test_subs = rng.choice(subs, r, replace=False)
        elif isinstance(hold_out, list):
            test_subs = hold_out

        train_subs = list(set(subs) - set(test_subs))
        test = x[(x['dataset'] == 'marea') & (x['subject_id'].isin(test_subs))]
        train = x[(x['dataset'] == 'nonan') | (x['subject_id'].isin(train_subs))]

    elif split_type == 'loso':
        subs = x['subject_id'].unique().tolist()
        rng = np.random.default_rng(seed=seed)

        if isinstance(hold_out, float):
            r = int(len(subs) * hold_out)
            test_subs = rng.choice(subs, r, replace=False)
        elif isinstance(hold_out, int):
            r = hold_out
            test_subs = rng.choice(subs, r, replace=False)
        elif isinstance(hold_out, list):
            test_subs = hold_out

        train_subs = list(set(subs) - set(test_subs))

        train = x[x['subject_id'].isin(train_subs)]
        test = x[x['subject_id'].isin(test_subs)]

        if sgs is not None:
            train_sgs = {k: v for k, v in sgs.items() if k in train_subs}
            test_sgs = {k: v for k, v in sgs.items() if k in test_subs}

    elif 'start' in split_type or 'end' in split_type:
        subs = x['subject_id'].unique().tolist()
        train, test = pd.DataFrame(), pd.DataFrame()

        for sub in subs:
            sub_x = x[x['subject_id'] == sub]

            if 'sub' in split_type:
                sub_train, sub_test = time_split(sub_x, split_type, hold_out)

                train = pd.concat([train, sub_train], axis=0)
                test = pd.concat([test, sub_test], axis=0)

            elif 'act' in split_type:
                for act in sub_x['activity_id'].unique():
                    sub_act_x = sub_x[sub_x['activity_id'] == act]

                    sub_act_train, sub_act_test = time_split(sub_act_x, split_type, hold_out)

                    train = pd.concat([train, sub_act_train], axis=0)
                    test = pd.concat([test, sub_act_test], axis=0)

                    if sgs is not None:
                        act_sgs = sgs[sub][act]
                        act_sgs_train, act_sgs_test = fft.time_split(act_sgs, split_type, hold_out)
                        train_sgs[sub][act] = act_sgs_train
                        test_sgs[sub][act] = act_sgs_test

    else:
        train = x
        test = pd.DataFrame()

    return train, test, train_sgs, test_sgs

def time_split(x: pd.DataFrame, split_type: str, hold_out: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_size = int(len(x) * hold_out)
    train_size = len(x) - test_size

    if 'start' in split_type:
        test, train = x.iloc[:test_size], x.iloc[test_size:]
    elif 'end' in split_type:
        train, test = x.iloc[:train_size], x.iloc[train_size:]
    else:
        train, test = pd.DataFrame(), pd.DataFrame()

    return train, test


