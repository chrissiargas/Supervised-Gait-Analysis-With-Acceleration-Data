import os
import pandas as pd
import numpy as np
from typing import Dict
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



char_to_int = {'A': 1, 'B': 2, 'C': 3}
def get_imu_features(from_sensor: str, to_sensor: str) -> Dict:


    imu = {f'sensorFreeAcceleration_{from_sensor}_x': f'accX_{to_sensor}',
           f'sensorFreeAcceleration_{from_sensor}_y': f'accY_{to_sensor}',
           f'sensorFreeAcceleration_{from_sensor}_z': f'accZ_{to_sensor}',
           f'sensorOrientation_{from_sensor}_q1': f'q1_{to_sensor}',
           f'sensorOrientation_{from_sensor}_qi': f'qi_{to_sensor}',
           f'sensorOrientation_{from_sensor}_qj': f'qj_{to_sensor}',
           f'sensorOrientation_{from_sensor}_qk': f'qk_{to_sensor}'}

    return imu

class MMgait:
    def __init__(self):
        self.source_path = os.path.join(
            os.path.expanduser('~'),
            'datasets',
            'MMgait',
            'data_set'
        )

        self.target_path = os.path.join(os.path.expanduser('~'),
                                      'datasets','MMgait_new')

        self.foot_events = {'insoles_LeftFoot_is_step': 'LF_HS',
                            'insoles_RightFoot_is_step': 'RF_HS',
                            'insoles_LeftFoot_is_lifted': 'LF_TO',
                            'insoles_RightFoot_is_lifted': 'RF_TO'}
        self.foot_phases = {'insoles_LeftFoot_on_ground': 'LF_stance',
                            'insoles_RightFoot_on_ground': 'RF_stance'}

        self.imu_LH = get_imu_features('LeftForeArm', 'LH')
        self.imu_RH = get_imu_features('RightForeArm', 'RH')
        self.imu_LF = get_imu_features('LeftLowerLeg', 'LF')
        self.imu_RF = get_imu_features('RightLowerLeg', 'RF')

        self.timestamps = {'participant_id': 'subject',
                           'task': 'activity',
                           'walk_mode': 'mode',
                           'time': 'time'}

        self.types = {
            'time': float,
            'subject': 'int8',
            'activity': 'int8',
            'mode': str,
            'LF_HS': 'int8',
            'RF_HS': 'int8',
            'LF_TO': 'int8',
            'RF_TO': 'int8',
            'LF_stance': 'int8',
            'RF_stance': 'int8'
        }

    def __call__(self, *args, **kwargs):
        courses_dir = sorted(os.listdir(self.source_path))
        course_path = os.path.join(self.source_path, courses_dir[0])
        subject_dir = sorted(os.listdir(course_path))

        data = None
        for sub_file in tqdm(subject_dir):
            if "id" not in sub_file:
                continue

            sub_data = None

            to_file = os.path.join(self.target_path, sub_file + '.csv')
            for course_path in courses_dir:
                if 'course' not in course_path:
                    continue

                sub_path = os.path.join(self.source_path, course_path, sub_file)
                imu_file = os.path.join(sub_path, 'xsens.csv')
                labels_file = os.path.join(sub_path, 'labels.csv')

                imu_df = pd.read_csv(imu_file)
                labels_df = pd.read_csv(labels_file)

                LH_df = imu_df[self.imu_LH.keys()]
                RH_df = imu_df[self.imu_RH.keys()]
                LF_df = imu_df[self.imu_LF.keys()]
                RF_df = imu_df[self.imu_RF.keys()]

                phases_df = labels_df[self.foot_phases.keys()].astype(int)
                events_df = labels_df[self.foot_events.keys()].astype(int)

                timestamps_df = labels_df[self.timestamps.keys()]

                LH_df = LH_df.rename(index=str, columns=self.imu_LH)
                RH_df = RH_df.rename(index=str, columns=self.imu_RH)
                LF_df = LF_df.rename(index=str, columns=self.imu_LF)
                RF_df = RF_df.rename(index=str, columns=self.imu_RF)

                phases_df = phases_df.rename(index=str, columns=self.foot_phases)
                events_df = events_df.rename(index=str, columns=self.foot_events)

                timestamps_df = timestamps_df.rename(index=str, columns=self.timestamps)
                timestamps_df = timestamps_df.replace({'activity': char_to_int})

                course_data = pd.concat([timestamps_df,
                                              LH_df, RH_df, LF_df, RF_df,
                                              events_df, phases_df], axis=1, sort=False)

                course_data = course_data.astype(self.types)

                if sub_data is None:
                    sub_data = course_data
                else:
                    sub_data = pd.concat([sub_data, course_data], axis=0, ignore_index=True)

            if data is None:
                data = sub_data
            else:
                data = pd.concat([data, sub_data], axis=0, ignore_index=True)

            sub_data.to_csv(to_file)

        return data

if __name__ == '__main__':
    constructor = MMgait()
    constructor()














