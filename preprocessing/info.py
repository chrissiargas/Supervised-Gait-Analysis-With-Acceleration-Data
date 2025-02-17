import os
from config_parser import Parser

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
                "treadmill_walk": 0,
                "treadmill_run": 4,
                "treadmill_slope_walk": 1,
                "indoor_walk": 2,
                "indoor_run": 5,
                "outdoor_walk": 3,
                "outdoor_run": 6
            }

            self.pos_pairs = {
                'LF': 'left_lower_leg',
                'RF': 'right_lower_leg',
                'Waist': 'waist',
                'Wrist': 'left_lower_arm'
            }

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

        if dataset == 'synthetic':
            self.pos_pairs = {
                'Waist': 'waist',
                'Wrist': 'left_lower_arm',
                'LH': 'left_lower_arm',
                'RH': 'right_lower_arm',
                'LF': 'left_lower_leg',
                'RF': 'right_lower_leg'
            }