import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict

def get_imu_features(from_sensor: str, to_sensor: str, side: str) -> Dict:


    imu = {f'{from_sensor} Accel Sensor X {side} (mG)': f'accX_{to_sensor}',
           f'{from_sensor} Accel Sensor Y {side} (mG)': f'accY_{to_sensor}',
           f'{from_sensor} Accel Sensor Z {side} (mG)': f'accZ_{to_sensor}',
           f'{from_sensor} course {side} (deg)': f'course_{to_sensor}',
           f'{from_sensor} pitch {side} (deg)': f'pitch_{to_sensor}',
           f'{from_sensor} roll {side} (deg)': f'roll_{to_sensor}'}

    return imu


class nonan:
    def __init__(self, delete: bool = False):
        self.source_path = os.path.join(
            '/media/crizo/X9 Pro/',
            'datasets',
            'NONAN'
        )

        self.target_path = os.path.join(os.path.expanduser('~'),
                                      'datasets','NONAN_new')

        if delete:
            try:
                for f in os.listdir(self.target_path):
                    shutil.rmtree(os.path.join(self.target_path, f))
                    os.mkdir(os.path.join(self.target_path, f))
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))


        self.phases = {'Contact LT': 'LF_stance', 'Contact RT': 'RF_stance'}
        self.timestamps = {'time': 'time'}

        self.imu_LH = get_imu_features('Forearm', 'LH', 'LT')
        self.imu_RH = get_imu_features('Forearm', 'RH', 'RT')
        self.imu_LF = get_imu_features('Shank', 'LF', 'LT')
        self.imu_RF = get_imu_features('Shank', 'RF', 'RT')

        self.foot_events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

        self.types = {
            'time': 'float64',
            'subject': 'int16',
            'activity': 'int8',
            'LF_HS': 'boolean',
            'RF_HS': 'boolean',
            'LF_TO': 'boolean',
            'RF_TO': 'boolean',
            'LF_stance': 'boolean',
            'RF_stance': 'boolean'
        }

        self.acc_factor = 9.80665 / 1000.

    def __call__(self, *args, **kwargs):
        population_dir = sorted(os.listdir(self.source_path))

        for pop_folder in tqdm(population_dir):
            pop_path = os.path.join(self.source_path, pop_folder)
            subject_dir = sorted(os.listdir(pop_path))

            for sub_file in tqdm(subject_dir):
                sub_df = None

                if 'S' != sub_file[0] or 'Spatiotemporal' in sub_file:
                    continue

                to_file = os.path.join(self.target_path, pop_folder, sub_file + '.csv')

                if os.path.exists(to_file):
                    continue

                sub_id = int(sub_file[-3:])
                sub_path = os.path.join(pop_path, sub_file)
                session_dir = sorted(os.listdir(sub_path))

                for session, session_name in enumerate(session_dir):
                    session_file = os.path.join(sub_path, session_name)
                    df = pd.read_csv(session_file)

                    LH_df = df[self.imu_LH.keys()]
                    RH_df = df[self.imu_RH.keys()]
                    LF_df = df[self.imu_LF.keys()]
                    RF_df = df[self.imu_RF.keys()]
                    phases_df = df[self.phases.keys()].astype(bool).astype(int)
                    timestamps_df = df[self.timestamps.keys()]

                    LH_df = LH_df.rename(index=str, columns=self.imu_LH)
                    RH_df = RH_df.rename(index=str, columns=self.imu_RH)
                    LF_df = LF_df.rename(index=str, columns=self.imu_LF)
                    RF_df = RF_df.rename(index=str, columns=self.imu_RF)
                    phases_df = phases_df.rename(index=str, columns=self.phases)
                    timestamps_df = timestamps_df.rename(index=str, columns=self.timestamps)
                    events_df = self.get_events(phases_df)

                    session_df = pd.concat([timestamps_df,
                                              LH_df, RH_df, LF_df, RF_df,
                                              events_df, phases_df], axis=1, sort=False)

                    acc_features = session_df.columns[session_df.columns.str.contains("acc")]
                    session_df.loc[:, acc_features] *= self.acc_factor

                    session_df.insert(0, 'activity', session + 1)
                    session_df.insert(0, 'subject', sub_id)

                    session_df = session_df.astype(self.types)

                    if sub_df is None:
                        sub_df = session_df
                    else:
                        sub_df = pd.concat([sub_df, session_df], axis=0, ignore_index=True)

                sub_df.to_csv(to_file)
                del sub_df


    def get_events(self, phases_data: pd.DataFrame) -> pd.DataFrame:
        event_names = ['HS', 'TO']
        event_indicators = [1, -1]
        events_data = pd.DataFrame(0, columns=self.foot_events, index=phases_data.index)

        for phase in phases_data.columns:
            foot = phase[:2]
            contacts = phases_data[phase].to_numpy()
            transitions = np.diff(contacts, prepend=contacts[0])

            for event, event_indicator in zip(event_names, event_indicators):
                foot_event = foot + '_' + event
                event_indices = np.argwhere(transitions == event_indicator).squeeze()
                events = np.zeros(shape=transitions.shape)
                events[event_indices] = 1
                events_data[foot_event] = events

        return events_data

if __name__ == '__main__':
    constructor = nonan(delete=True)
    constructor()
