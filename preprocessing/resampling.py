import resampy
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional
from scipy import signal as ssn
import matplotlib.pyplot as plt
import os
from datetime import datetime
figpath = os.path.join('archive', 'figures')

def resample(df: pd.DataFrame, old_fs: int, new_fs: int, how: str = 'decimate') -> pd.DataFrame:
    imu = df.columns[df.columns.str.contains('acc|gyr')]
    phases = df.columns[df.columns.str.contains('stance')]
    events = df.columns[df.columns.str.contains('HS|TO')]

    resampled_ds = pd.DataFrame()
    step = 1000. / new_fs
    fs_scale = new_fs / old_fs
    T_scale = old_fs / new_fs

    for sub_id, sub_df in df.groupby('subject'):
        for act_id, act_df in sub_df.groupby('activity'):
            for pos_id, pos_df in act_df.groupby('position'):
                x = pos_df.interpolate(method='linear')
                old_imu = x[imu].values

                if how == 'resampy':
                    old_t = x['time'].values
                    new_imu = resampy.resample(old_imu, old_fs, new_fs, axis=0)
                elif how == 'decimate':
                    old_t = x['time'].values * 1000.
                    new_imu = ssn.decimate(x=old_imu, q=int(T_scale), ftype='fir',axis=0)

                resampled_df = pd.DataFrame(new_imu, columns=imu)
                new_t =  np.arange(start=old_t[0], stop=old_t[0] + new_imu.shape[0] * step, step=step)

                if len(phases) > 0:
                    old_phases = x[phases].values
                    new_phases = interp1d(old_t, old_phases, kind='nearest', axis=0, fill_value='extrapolate')(new_t)
                    new_phases = pd.DataFrame(new_phases, columns=phases)
                    resampled_df = pd.concat((resampled_df, new_phases), axis=1)

                n_samples = new_imu.shape[0]
                new_events = pd.DataFrame(columns=events)
                for event in events:
                    old_event_indexes = np.where(x[event])[0]
                    new_event_indexes = np.clip((old_event_indexes * fs_scale).astype(int), 0, n_samples-1)
                    new_event = np.zeros(n_samples)
                    new_event[new_event_indexes] = 1
                    new_events[event] = new_event
                resampled_df = pd.concat((resampled_df, new_events), axis=1)

                NaNs = pos_df.isna().any(axis=1).values.astype(int)
                f = interp1d(old_t, NaNs, kind='previous', axis=0, fill_value='extrapolate')
                prev_NaNs = f(new_t)
                f = interp1d(old_t, NaNs, kind='next', axis=0, fill_value='extrapolate')
                next_NaNs = f(new_t)

                resampled_df['is_NaN'] = False
                resampled_df.loc[(prev_NaNs == 1) | (next_NaNs == 1), 'is_NaN'] = True

                resampled_df['time'] = new_t
                resampled_df['position'] = pos_id
                resampled_df['activity'] = act_id
                resampled_df['subject'] = sub_id

                resampled_ds = pd.concat((resampled_ds, resampled_df), axis=0, ignore_index=True)

                del x

    return resampled_ds
