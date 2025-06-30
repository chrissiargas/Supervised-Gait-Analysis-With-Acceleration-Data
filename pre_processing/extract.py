import os

import pandas as pd
from tqdm import tqdm
from pre_processing.resampling import resample
import numpy as np
from config.config_parser import Parser
from pre_processing.info import info
from typing import Optional, Tuple
from typing import List
from scipy.spatial.transform import Rotation
from rotation_utils import inv_calibrate

id_cols = [
    'subject',
    'activity',
    'time'
]

label_cols = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO', 'LF_stance', 'RF_stance', 'mode']

def split_with_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df['id'] = df[id_cols].astype(str).agg('-'.join, axis=1)
    res_cols = df.columns[df.columns.str.contains('acc|NaN')]
    data_cols = df.columns[~df.columns.str.contains('acc|NaN')]
    data_df = df[data_cols]
    res_df = df[['id', *res_cols]]

    return res_df, data_df


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
    def __init__(self, dataset: str, experiment: str):
        self.sub_offset = 0

        config = Parser()
        config.get_args()
        self.conf = config

        self.dataset = dataset
        self.experiment = experiment

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
            self.nonan_extract()
            self.sub_offset += 1000
        if 'MMgait' in self.dataset:
            self.info = info('MMgait')
            self.mmgait_extract()
            self.sub_offset += 1000

    def get_ds(self, sub_path: str, dataset: str, position: str, sub_id: Optional[int] = None) -> pd.DataFrame:
        if dataset == 'marea':
            df = self.marea_load_subject(sub_path, sub_id, position)
        elif dataset == 'nonan':
            df = self.nonan_load_subject(sub_path, position)
        elif dataset == 'mmgait':
            df = self.mmgait_load_subject(sub_path, position)

        return df

    def get_subject(self, sub_path: str, dataset: str, sub_id: Optional[int] = None) -> pd.DataFrame:
        if self.experiment == 'supervised':
            sub_df = self.get_ds(sub_path, dataset, self.conf.position, sub_id)

        elif self.experiment == 'self_supervised':
            sub_df = None
            for position in self.conf.position:
                pos_df = self.get_ds(sub_path, dataset, position, sub_id)
                pos_df = pos_df.rename(columns={ft: ft + '_' + position for ft in self.info.imu_features.values()})
                pos_df = pos_df.rename(columns={'is_NaN': 'is_NaN' + '_' + position})
                pos_df = pos_df.drop(columns=['position'])

                pos_df, same = split_with_id(pos_df)

                if sub_df is None:
                    sub_df = pos_df
                else:
                    sub_df = pd.merge(sub_df, pos_df, on='id', how='left')

            sub_df = pd.merge(sub_df, same, on='id', how='left')
            sub_df = sub_df.drop(columns=['id'])

        return sub_df

    def mmgait_extract(self):
        print(f'Loading MM-Gait data...')

        subject_dir = sorted(os.listdir(self.info.path))
        for sub_file in tqdm(subject_dir):
            if 'id' not in sub_file:
                continue

            sub_id = int(sub_file[2:-4]) + self.sub_offset
            sub_path = os.path.join(self.info.path, sub_file)
            sub_df = self.get_subject(sub_path, 'mmgait')
            sub_df = sub_df.sort_values(by=['activity', 'time'])
            sub_df.insert(0, 'dataset', 'mmgait')

            to_file = os.path.join(self.path, f'id{sub_id}.csv')
            sub_df.to_csv(to_file)

            del sub_df

    def mmgait_load_subject(self, path: str, position: str) -> pd.DataFrame:
        df = pd.read_csv(path)

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

        acc_features = list(self.info.imu_features.values())
        a = df[acc_features].values
        R = self.info.rotation[position]
        a_rot = Rotation.from_matrix(R).apply(a)
        df[acc_features] = a_rot

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
            sub_df = self.get_subject(sub_path, 'nonan')
            sub_df = sub_df.sort_values(by=['activity', 'time'])

            sub_df.insert(0, 'population', population)
            sub_df.insert(0, 'dataset', 'nonan')

            to_file = os.path.join(self.path, f'id{sub_id}.csv')
            sub_df.to_csv(to_file)

            del sub_df


    def nonan_load_subject(self, path, position: str) -> pd.DataFrame:
        df = pd.read_csv(path)

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
            sub_df = self.get_subject(sub_path, 'marea', sub_id)
            sub_df = sub_df.sort_values(by=['activity', 'time'])
            sub_df.insert(0, 'dataset', 'marea')

            to_file = os.path.join(self.path, f'id{sub_id}.csv')
            sub_df.to_csv(to_file)

            del sub_df

    def marea_load_subject(self, path, sub_id: int, position: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        initial_activities = [
            "treadmill_walk", "treadmill_walknrun", "treadmill_slope_walk",
            "indoor_walk", "indoor_walknrun", "outdoor_walk", "outdoor_walknrun"
        ]

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

        acc_features = list(self.info.imu_features.values())
        a = df[acc_features].values

        if df['acc_y'].mean() > 0:
            R = self.info.y_pos_rotation[position]
        elif df['acc_y'].mean() < 0:
            R = self.info.y_neg_rotation[position]

        a_rot = Rotation.from_matrix(R).apply(a)
        df[acc_features] = a_rot

        return df

import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PTXAS_OPTIONS'] = '-w'

if __name__ == '__main__':
    extract_data = extractor('MMgait', 'self_supervised')
    extract_data()