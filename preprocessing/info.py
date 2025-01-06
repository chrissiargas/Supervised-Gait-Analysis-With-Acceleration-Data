import os
from config_parser import Parser

class info:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

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
            "treadmill_walk": 'treadmill_walking',
            "treadmill_run": 'treadmill_running',
            "treadmill_slope_walk": 'treadmill_slope_walking',
            "indoor_walk": 'indoor_walking',
            "indoor_run": 'indoor_running',
            "outdoor_walk": 'outdoor_walking',
            "outdoor_run": 'outdoor_running'
        }

        self.pos_pairs = {
            'LF': 'left_lower_leg',
            'RF': 'right_lower_leg',
            'Waist': 'waist',
            'Wrist': 'left_lower_arm'
        }