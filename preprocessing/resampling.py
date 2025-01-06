import resampy
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

METHOD = 'decimate'

def resample(ds: pd.DataFrame, old_fs: int, new_fs: int, thres: float = 1.) -> pd.DataFrame:
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    event_cols = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

    resampled_ds = pd.DataFrame()
    step = 1000. / new_fs
    fs_scale = new_fs / old_fs

    for sub_id, sub_df in ds.groupby('subject'):
        for act_id, act_df in sub_df.groupby('activity'):
            for pos_id, pos_df in act_df.groupby('position'):
                old_t = pos_df['timestamp'].values


                old_acc = pos_df[acc_cols].values
                new_acc = resampy.resample(old_acc, old_fs, new_fs, axis=0)
                resampled_df = pd.DataFrame(new_acc, columns=acc_cols)
                new_t =  np.arange(start=old_t[0], stop=old_t[0] + new_acc.shape[0] * step, step=step)


                n_samples = int(pos_df.shape[0] * fs_scale)
                new_events = pd.DataFrame(columns=event_cols)
                for event_col in event_cols:
                    old_event_indexes = np.where(pos_df[event_col])[0]
                    new_event_indexes = (old_event_indexes * fs_scale).astype(int)
                    new_event = np.zeros(n_samples)
                    new_event[new_event_indexes] = 1.
                    new_events[event_col] = new_event
                resampled_df = pd.concat((resampled_df, new_events), axis=1)

                NaNs = pos_df.isna().any(axis=1).values.astype(int)
                f = interp1d(old_t, NaNs, kind='previous', axis=0, fill_value='extrapolate')
                prev_NaNs = f(new_t)
                f = interp1d(old_t, NaNs, kind='next', axis=0, fill_value='extrapolate')
                next_NaNs = f(new_t)
                resampled_df.loc[(prev_NaNs == 1) | (next_NaNs == 1)] = np.nan

                f = interp1d(old_t, old_t, kind='nearest', axis=0, fill_value='extrapolate')
                nearest_t = f(new_t)
                resampled_df.loc[abs(new_t - nearest_t) > thres * 1000. / old_fs] = np.nan

                resampled_df['timestamp'] = new_t
                resampled_df['position'] = pos_id
                resampled_df['activity'] = act_id
                resampled_df['subject'] = sub_id

                resampled_ds = pd.concat((resampled_ds, resampled_df),
                                         axis=0, ignore_index=True)

    return resampled_ds
