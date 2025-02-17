import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

class nonan:
    def __init__(self):
        self.source_path = os.path.join(
            os.path.expanduser('~'),
            'datasets',
            'NONAN'
        )

        self.target_path = os.path.join(os.path.expanduser('~'),
                                      'datasets','NONAN_new')

        z = os.path.join(self.source_path, 'extracted')

        try:
            shutil.rmtree(z)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        self.phases = {'Contact LT': 'LF_stance', 'Contact RT': 'RF_stance'}
        self.timestamps = {'time': 'time'}

        self.imu_LH = {'Forearm Accel Sensor X LT (mG)': 'accX_LH',
                       'Forearm Accel Sensor Y LT (mG)': 'accY_LH',
                       'Forearm Accel Sensor Z LT (mG)': 'accZ_LH',
                       'Noraxon MyoMotion-Segments-Forearm LT-Gyroscope-x (deg/s)': 'gyrX_LH',
                       'Noraxon MyoMotion-Segments-Forearm LT-Gyroscope-y (deg/s)': 'gyrY_LH',
                       'Noraxon MyoMotion-Segments-Forearm LT-Gyroscope-z (deg/s)': 'gyrZ_LH'}

        self.imu_RH = {'Forearm Accel Sensor X RT (mG)': 'accX_RH',
                       'Forearm Accel Sensor Y RT (mG)': 'accY_RH',
                       'Forearm Accel Sensor Z RT (mG)': 'accZ_RH',
                       'Noraxon MyoMotion-Segments-Forearm RT-Gyroscope-x (deg/s)': 'gyrX_RH',
                       'Noraxon MyoMotion-Segments-Forearm RT-Gyroscope-y (deg/s)': 'gyrY_RH',
                       'Noraxon MyoMotion-Segments-Forearm RT-Gyroscope-z (deg/s)': 'gyrZ_RH'}

        self.imu_LF = {'Foot Accel Sensor X LT (mG)': 'accX_LF',
                       'Foot Accel Sensor Y LT (mG)': 'accY_LF',
                       'Foot Accel Sensor Z LT (mG)': 'accZ_LF',
                       'Noraxon MyoMotion-Segments-Foot LT-Gyroscope-x (deg/s)': 'gyrX_LF',
                       'Noraxon MyoMotion-Segments-Foot LT-Gyroscope-y (deg/s)': 'gyrY_LF',
                       'Noraxon MyoMotion-Segments-Foot LT-Gyroscope-z (deg/s)': 'gyrZ_LF'}

        self.imu_RF = {'Foot Accel Sensor X RT (mG)': 'accX_RF',
                       'Foot Accel Sensor Y RT (mG)': 'accY_RF',
                       'Foot Accel Sensor Z RT (mG)': 'accZ_RF',
                       'Noraxon MyoMotion-Segments-Foot RT-Gyroscope-x (deg/s)': 'gyrX_RF',
                       'Noraxon MyoMotion-Segments-Foot RT-Gyroscope-y (deg/s)': 'gyrY_RF',
                       'Noraxon MyoMotion-Segments-Foot RT-Gyroscope-z (deg/s)': 'gyrZ_RF'}

        self.foot_events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

        self.acc_factor = 9.80665 / 1000.

    def __call__(self, *args, **kwargs):
        subject_dir = sorted(os.listdir(self.source_path))
        params_path = os.path.join(self.source_path, 'Spatiotemporal_Variables')

        data = None
        for sub_file in tqdm(subject_dir):
            if "S0" not in sub_file:
                continue

            to_file = os.path.join(self.target_path, sub_file + '.csv')
            if os.path.exists(to_file):
                sub_data = pd.read_csv(to_file)
                print(sub_data.isna().sum())
                data = pd.concat([data, sub_data], axis=0, ignore_index=True)
                continue

            sub_id = int(sub_file[-3:])
            sub_path = os.path.join(self.source_path, sub_file)
            session_dir = sorted(os.listdir(sub_path))

            sub_data = None
            for session, session_name in tqdm(enumerate(session_dir)):
                parameters_file = os.path.join(params_path, session_name)
                parameters = pd.read_csv(parameters_file)
                if len(parameters) == 1:
                    print('wrong contact data: ' + session_name)
                    continue

                session_file = os.path.join(sub_path, session_name)
                df = pd.read_csv(session_file)

                LH = df[self.imu_LH.keys()]
                RH = df[self.imu_RH.keys()]
                LF = df[self.imu_LF.keys()]
                RF = df[self.imu_RF.keys()]
                phases = df[self.phases.keys()].astype(bool).astype(int)
                timestamps = df[self.timestamps.keys()]

                LH = LH.rename(index=str, columns=self.imu_LH)
                RH = RH.rename(index=str, columns=self.imu_RH)
                LF = LF.rename(index=str, columns=self.imu_LF)
                RF = RF.rename(index=str, columns=self.imu_RF)
                phases = phases.rename(index=str, columns=self.phases)
                timestamps = timestamps.rename(index=str, columns=self.timestamps)
                events = self.get_events(phases)

                session_data = pd.concat([timestamps, LH, RH, LF, RF, phases, events], axis=1, sort=False)
                acc_features = session_data.columns[session_data.columns.str.contains("acc")]
                session_data.loc[:, acc_features] *= self.acc_factor

                session_data.insert(0, 'activity', session+1)
                session_data.insert(0, 'subject', sub_id)

                session_data = session_data.astype({'time': float, 'subject': 'int8', 'activity': 'int8',
                                    'LF_stance': 'int8', 'RF_stance': 'int8',
                                    'LF_HS': 'int8', 'RF_HS': 'int8', 'LF_TO': 'int8', 'RF_TO': 'int8'})

                if sub_data is None:
                    sub_data = session_data
                else:
                    sub_data = pd.concat([sub_data, session_data], axis=0, ignore_index=True)

            if data is None:
                data = sub_data
            else:
                data = pd.concat([data, sub_data], axis=0, ignore_index=True)

            sub_data.to_csv(to_file)

        data.to_csv(os.path.join(self.target_path, "nonan" + ".csv"))
        return data

    def get_events(self, phases_data: pd.DataFrame) -> pd.DataFrame:
        event_names = ['HS', 'TO']
        event_indicators = [1, -1]
        events_data = pd.DataFrame(0, columns=self.foot_events, index=phases_data.index)
        print()

        for phase in phases_data.columns:
            foot = phase[:2]
            contacts = phases_data[phase].to_numpy()
            transitions = np.diff(contacts, prepend=contacts[0])

            for event, event_indicator in zip(event_names, event_indicators):
                foot_event = foot + '_' + event
                event_indices = np.argwhere(transitions == event_indicator).squeeze()
                print(foot_event, event_indices.shape[0])
                events = np.zeros(shape=transitions.shape)
                events[event_indices] = 1
                events_data[foot_event] = events

        return events_data

if __name__ == '__main__':
    constructor = nonan()
    constructor()
