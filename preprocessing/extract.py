import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import legend
from pkg_resources import to_filename
from tqdm import tqdm
from preprocessing.resampling import resample
import numpy as np
from config_parser import Parser
from preprocessing.info import info
from scipy.spatial.transform import Rotation as R
from typing import Optional
from preprocessing.filters import butter_lowpass_filter
from scipy.constants import g
from typing import List

def inv_calibrate(x: pd.DataFrame, type: str) -> pd.DataFrame:
    acc_features = x.columns[x.columns.str.contains("acc")]

    a_free = x[acc_features].values
    g_world = np.array([0, 0, 9.81])

    if type == 'quat':
        quat_features = ['q1', 'qi', 'qj', 'qk']
        q = x[quat_features].values
        rotation = R.from_quat(q[:, [1, 2, 3, 0]])

    elif type == 'euler':
        rot_features = ['course', 'pitch', 'roll']
        rot = x[rot_features].values
        rotation = R.from_euler('ZYX', rot, degrees=True)

    a_raw = rotation.inv().apply(a_free + g_world)
    x[acc_features] = a_raw

    return x

def add_phase(timeseries: pd.DataFrame, event1: str, event2: str) -> np.ndarray:
    event1_indices = np.where(timeseries[event1] == 1)[0]
    event2_indices = np.where(timeseries[event2] == 1)[0]
    phase_ts = np.zeros(timeseries.shape[0])

    for event1_index in event1_indices:
        next_indices = event2_indices[event2_indices > event1_index]
        if len(next_indices):
            event2_index = next_indices[0]
            phase_ts[event1_index+1:event2_index+1] = 1

    return phase_ts.astype(int)

def get_phases(x: pd.DataFrame) -> pd.DataFrame:
    x['LF_stance'] = add_phase(x, 'LF_HS', 'LF_TO')
    x['RF_stance'] = add_phase(x, 'RF_HS', 'RF_TO')

    return x

def convert_activity(x: pd.DataFrame, initial_acts: List[str]) -> pd.DataFrame:
    def row_split(row, place):
        if row[place + '_walknrun'] == 1 and row[place + '_walk'] == 0:
            return 1
        return 0

    places = ['treadmill', 'indoor', 'outdoor']
    for place in places:
        if place + '_walknrun' in x.columns:
            x[place + '_run'] = x.apply(lambda row: row_split(row, place), axis=1)

    walknrun_cols = [col for col in x if col.endswith('walknrun')]
    x = x.drop(walknrun_cols, axis=1)

    activities = x[x.columns.intersection(initial_acts)]
    x['activity'] = activities.idxmax(axis=1)
    x.loc[~activities.any(axis='columns'), 'activity'] = 'undefined'

    x = x.drop(x.columns.intersection(initial_acts), axis=1)

    return x

class extractor:
    def __init__(self, dataset: str):
        self.sub_offset = 0

        config = Parser()
        config.get_args()
        self.conf = config

        self.dataset = dataset

        self.path = os.path.join(
            os.path.expanduser('~'),
            self.conf.path,
            'wrist_gait'
        )

        try:
            for f in os.listdir(self.path):
                os.unlink(os.path.join(self.path,f))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    def __call__(self):
        if 'marea' in self.dataset:
            self.info = info('marea')
            self.marea_extract()
            self.sub_offset += 1000
        if 'nonan' in self.dataset:
            self.info = info('nonan')
            self.nonan_extract(population='older')
            self.sub_offset += 1000
        if 'MMgait' in self.dataset:
            self.info = info('MMgait')
            self.mmgait_extract()
            self.sub_offset += 1000

    def mmgait_extract(self):
        print(f'Loading MM-Gait data...')

        subject_dir = sorted(os.listdir(self.info.path))
        for sub_file in tqdm(subject_dir):
            if 'id' not in sub_file:
                continue

            sub_id = int(sub_file[2:-4]) + self.sub_offset
            sub_path = os.path.join(self.info.path, sub_file)
            sub_df = self.mmgait_load_subject(sub_path)
            sub_df = sub_df.sort_values(by=['activity', 'time'])
            sub_df.insert(0, 'dataset', 'mmgait')

            to_file = os.path.join(self.path, f'id{sub_id}.csv')
            sub_df.to_csv(to_file)

            del sub_df

    def mmgait_load_subject(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        position = self.conf.position

        if position == 'Wrist':
            position = 'LH'

        pos_imu = df.columns[df.columns.str.contains(position)]
        columns = df.columns.intersection([*self.info.indicators, *pos_imu, *self.info.events, *self.info.phases])
        df = df[columns]

        df = df.reset_index()
        df['position'] = position
        df.columns = df.columns.str.replace('_' + position, '')

        df = df.rename(columns=self.info.imu_features)

        df['position'] = df['position'].map(self.info.pos_pairs)
        df = inv_calibrate(df, type='quat')

        for sensor in ['acc', 'quat']:
            features = df.columns[df.columns.str.contains(sensor)]
            df.loc[(df[features] == 0).all(axis='columns'), features] = np.nan

        if self.conf.fs is not None:
            df = resample(df, self.info.initial_fs, self.conf.fs, how='resampy')

        df = df[self.info.columns.keys()]
        df = df.astype(self.info.columns)

        df['subject'] += self.sub_offset

        return df

    def nonan_extract(self, population: Optional[str] = None):
        print(f'Loading NONAN data...')

        if population is None:
            self.nonan_load_population(self.info.path, 'young')
            self.nonan_load_population(self.info.path, 'older')
        elif population == 'young' or population == 'older':
            self.nonan_load_population(self.info.path, population)

    def nonan_load_population(self, path, population: str):
        population_path = os.path.join(path, population)
        subject_dir = sorted(os.listdir(population_path))

        print(f'Loading {population} data...')
        for sub_file in tqdm(subject_dir):
            if 'S' != sub_file[0]:
                continue

            sub_id = int(sub_file[1:-4]) + self.sub_offset
            sub_path = os.path.join(population_path, sub_file)
            sub_df = self.nonan_load_subject(sub_path)
            sub_df = sub_df.sort_values(by=['activity', 'time'])

            sub_df.insert(0, 'population', population)
            sub_df.insert(0, 'dataset', 'nonan')

            to_file = os.path.join(self.path, f'id{sub_id}.csv')
            sub_df.to_csv(to_file)

            del sub_df


    def nonan_load_subject(self, path) -> pd.DataFrame:
        df = pd.read_csv(path)
        position = self.conf.position

        if position == 'Wrist':
            position = 'LH'

        pos_imu = df.columns[df.columns.str.contains(position)]
        columns = df.columns.intersection([*self.info.indicators, *pos_imu, *self.info.events, *self.info.phases])
        df = df[columns]

        df = df.reset_index()
        df['position'] = self.conf.position
        df.columns = df.columns.str.replace('_' + self.conf.position, '')

        df = df.rename(columns=self.info.imu_features)

        df['position'] = df['position'].map(self.info.pos_pairs)

        features = df.columns[df.columns.str.contains('acc')]
        df.loc[(df[features] == 0).all(axis='columns'), features] = np.nan

        if self.conf.fs is not None:
            df = resample(df, self.info.initial_fs, self.conf.fs, how='resampy', r=1000.)

        df = df[self.info.columns.keys()]
        df = df.astype(self.info.columns)

        df['subject'] += self.sub_offset

        return df

    def marea_extract(self):
        print(f'Loading MAREA data...')

        subject_dir = sorted(os.listdir(self.info.path))
        for sub_file in tqdm(subject_dir):
            if 'Sub' != sub_file[:3]:
                continue

            sub_id = int(sub_file[4:-4]) + self.sub_offset
            sub_path = os.path.join(self.info.path, sub_file)
            sub_df = self.marea_load_subject(sub_path, sub_id)
            sub_df = sub_df.sort_values(by=['activity', 'time'])
            sub_df.insert(0, 'dataset', 'marea')

            to_file = os.path.join(self.path, f'id{sub_id}.csv')
            sub_df.to_csv(to_file)

            del sub_df

    def marea_load_subject(self, path, sub_id: int) -> pd.DataFrame:
        df = pd.read_csv(path)

        initial_activities = [
            "treadmill_walk", "treadmill_walknrun", "treadmill_slope_walk",
            "indoor_walk", "indoor_walknrun", "outdoor_walk", "outdoor_walknrun"
        ]
        position = self.conf.position

        if position == 'LH':
            position = 'Wrist'

        acc_x = 'accX_' + position
        acc_y = 'accY_' + position
        acc_z = 'accZ_' + position

        columns = [acc_x, acc_y, acc_z, *initial_activities, *self.info.events]
        df = df[df.columns.intersection(columns)].copy()
        df['subject'] = sub_id

        df = convert_activity(df, self.info.activities)
        df = get_phases(df)

        df = df.reset_index()
        df['time'] = df.index * (1000. / 128.)
        df = df.drop(['index'], axis=1)

        df.columns = df.columns.str.replace('_' + position, '')
        df['position'] = position

        df = df.rename(columns=self.info.imu_features)

        activities = self.conf.activities
        fs = self.conf.fs

        if activities is not None:
            df = df[df['activity'].str.contains('|'.join(activities))]

        df['activity'] = df['activity'].map(self.info.act_pairs)
        df['position'] = df['position'].map(self.info.pos_pairs)

        if fs is not None:
            df = resample(df, self.info.initial_fs, fs, how='resampy')

        df = df[self.info.columns.keys()]
        df = df.astype(self.info.columns)

        return df