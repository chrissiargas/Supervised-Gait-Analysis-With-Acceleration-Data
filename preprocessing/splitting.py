import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Dict


def split_all(x: pd.DataFrame,
              validation: bool,
              split_type: str,
              test_hold_out: Union[List, int, float, str],
              val_hold_out: Union[List, int, float, str],
              seed: Optional[int] = None):

    train, test = split(x, split_type, test_hold_out, seed)

    if validation:
        train, val = split(train, split_type, val_hold_out, seed)
    else:
        val = None

    return train, test, val

def split(x: pd.DataFrame, split_type: str,
          hold_out: Union[List, int, float, str],
          seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    x = x.copy()

    if split_type == 'dataset':
        test = x[x['dataset'] == hold_out]
        train = x[x['dataset'] != hold_out]

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

    else:
        train = x
        test = pd.DataFrame()

    return train, test


