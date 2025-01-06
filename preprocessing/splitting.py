import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union


def split(x: pd.DataFrame, split_type: str, hold_out: Union[List, int, float, str], seed: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = x.copy()
    subs = x['subject'].unique().tolist()

    if split_type == 'loso':
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

        train = x[x['subject'].isin(train_subs)]
        test = x[x['subject'].isin(test_subs)]

    elif 'start' in split_type or 'end' in split_type:
        train, test = pd.DataFrame(), pd.DataFrame()
        for sub in subs:
            sub_x = x[x['subject'] == sub]

            if 'sub' in split_type:
                sub_train, sub_test = time_split(sub_x, split_type, hold_out)

                train = pd.concat([train, sub_train], axis=0)
                test = pd.concat([test, sub_test], axis=0)

            elif 'act' in split_type:
                for act in sub_x['activity'].unique():
                    sub_act_x = sub_x[sub_x['activity'] == act]

                    sub_act_train, sub_act_test = time_split(sub_act_x, split_type, hold_out)

                    train = pd.concat([train, sub_act_train], axis=0)
                    test = pd.concat([test, sub_act_test], axis=0)

    else:
        train = x
        test = pd.DataFrame()

    return train, test

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


