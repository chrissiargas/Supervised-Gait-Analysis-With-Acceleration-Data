import os
from config.config_parser import Parser

class info:
    def __init__(self, dataset: str):
        config = Parser()
        config.get_args()
        self.conf = config

        if dataset == 'marea':
            self.path = os.path.join(
                os.path.expanduser('~'),
                config.path,
                'MAREA_new'
            )

            self.initial_fs = 128

            self.activities = [
                "treadmill_walk", "treadmill_run", "treadmill_slope_walk",
                "indoor_walk", "indoor_run", "outdoor_walk", "outdoor_run"
            ]

            self.events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']
            self.positions = ['LF', 'RF', 'Waist', 'Wrist']

            self.act_pairs = {
                "undefined": 'no pair',
                "treadmill_walk": 1,
                "treadmill_run": 5,
                "treadmill_slope_walk": 2,
                "indoor_walk": 3,
                "indoor_run": 6,
                "outdoor_walk": 4,
                "outdoor_run": 7
            }

            self.pos_pairs = {
                'LF': 'left_lower_leg',
                'RF': 'right_lower_leg',
                'Waist': 'waist',
                'Wrist': 'left_lower_arm'
            }

            self.columns = {'position': str, 'subject': 'int16',
                            'activity': 'int8', 'time': 'float64', 'is_NaN': 'boolean',
                            'acc_x': 'float64', 'acc_y': 'float64', 'acc_z': 'float64',
                            'LF_HS': 'int8', 'RF_HS': 'int8', 'LF_TO': 'int8', 'RF_TO': 'int8',
                            'LF_stance': 'int8', 'RF_stance': 'int8'}

            self.imu_features = {"accX": "acc_x",
                                "accY": "acc_y",
                                "accZ": "acc_z"}

            self.y_pos_rotation = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
            self.y_neg_rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

        if dataset == 'nonan':
            self.path = os.path.join(
                os.path.expanduser('~'),
                config.path,
                'NONAN_new'
            )

            self.initial_fs = 200
            self.events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']
            self.phases = ['LF_stance', 'RF_stance']
            self.positions = ['LF', 'RF', 'LH', 'RH']

            self.pos_pairs = {
                'LH': 'left_lower_arm',
                'RH': 'right_lower_arm',
                'LF': 'left_lower_leg',
                'RF': 'right_lower_leg'
            }

            self.indicators = ['time', 'subject', 'activity']

            self.columns = {'position': str, 'subject': 'int16',
                            'activity': 'int8', 'time': 'float64', 'is_NaN': 'boolean',
                            'acc_x': 'float64', 'acc_y': 'float64', 'acc_z': 'float64',
                            'LF_HS': 'int8', 'RF_HS': 'int8', 'LF_TO': 'int8', 'RF_TO': 'int8',
                            'LF_stance': 'int8', 'RF_stance': 'int8'}

            self.imu_features = {"accX": "acc_x",
                                "accY": "acc_y",
                                "accZ": "acc_z"}

            self.rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        if dataset == 'MMgait':
            self.path = os.path.join(
                os.path.expanduser('~'),
                config.path,
                'MMgait_new'
            )

            self.initial_fs = 60
            self.events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']
            self.phases = ['LF_stance', 'RF_stance']
            self.positions = ['LF', 'RF', 'LH', 'RH']
            self.indicators = ['subject', 'activity', 'mode', 'time']

            self.pos_pairs = {
                'LH': 'left_lower_arm',
                'RH': 'right_lower_arm',
                'LF': 'left_lower_leg',
                'RF': 'right_lower_leg'
            }

            self.columns = {'position': str, 'subject': 'int16',
                            'activity': 'int8', 'mode': str, 'time': 'float64', 'is_NaN': 'boolean',
                            'acc_x': 'float64', 'acc_y': 'float64', 'acc_z': 'float64',
                            'LF_HS': 'int8', 'RF_HS': 'int8', 'LF_TO': 'int8', 'RF_TO': 'int8',
                            'LF_stance': 'int8', 'RF_stance': 'int8'}

            self.imu_features = {"accX": "acc_x",
                                "accY": "acc_y",
                                "accZ": "acc_z"}

            self.rotation = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]

        if dataset == 'synthetic':
            self.pos_pairs = {
                'Waist': 'waist',
                'Wrist': 'left_lower_arm',
                'LH': 'left_lower_arm',
                'RH': 'right_lower_arm',
                'LF': 'left_lower_leg',
                'RF': 'right_lower_leg'
            }




