import matplotlib.pyplot as plt

from config_parser import Parser
import os
from model_utils.supervised import alligaitor
from preprocessing.building import builder
from plots import plot_signal
from typing import Optional
import numpy as np
from datetime import datetime
from itertools import permutations
from scipy.spatial.transform import Rotation
from rotation_utils import rotation_by_axis

def visualize(path: Optional[str] = None,
              set: str = 'test',
              subject: Optional[int] = None,
              activity: int = 3):
    data = builder()
    data()

    config = Parser()
    config.get_args()

    model_dir = f'archive/model_weights/{config.architecture}'
    model_file = '%s.weights.h5' % config.architecture
    model_file = f'{model_dir}/{model_file}'

    model = alligaitor(data)
    model.compile()
    model.build_model(data.input_shape)
    model.load_weights(model_file)

    if subject is None:
        subject = config.test_hold_out[0]

    df, _, _ = data.compare_yy_(model, which=set, subject=subject, activity=activity, oversample=True)

    show_events = True if config.task == 'gait_events' else False
    show_phases = True if config.task == 'gait_phases' else False

    for i, start in enumerate(range(0, df.shape[0], 1000)):
        plot_signal(df, 'left_lower_arm', subject=subject,
                    activity=activity, start=start, length=1000,
                    show_events=show_events, features='acc', turn=i,
                    show_phases=show_phases, raw=True, figpath=path)


def visualize_rot(path: Optional[str] = None,
                  set: str = 'test',
                  subject: Optional[int] = None,
                  activity: int = 3):

    data = builder()
    data()

    config = Parser()
    config.get_args()

    model_dir = f'archive/model_weights/{config.architecture}'
    model_file = '%s.weights.h5' % config.architecture
    model_file = f'{model_dir}/{model_file}'

    model = alligaitor(data)
    model.compile()
    model.build_model(data.input_shape)
    model.load_weights(model_file)

    if subject is None:
        subject = config.test_hold_out[0]

    start = 4000
    window = 1000

    angles = np.arange(-50, 50, 10)  # Rotate 0-360° in 10° steps

    for subject in config.test_hold_out:
        for angle in angles:
            R = rotation_by_axis(angle, axis='x')

            df, _, _ = data.compare_yy_(model, which=set, subject=subject, activity=activity,
                                            start=start, end=start+window, rotation=R, oversample=True)

            show_events = True if config.task == 'gait_events' else False
            show_phases = True if config.task == 'gait_phases' else False

            for i, start_ in enumerate(range(0, df.shape[0], window)):
                plot_signal(df, 'left_lower_arm', subject=subject,
                            activity=activity, start=start_, length=window,
                            show_events=show_events, features='acc', turn=f'-{angle}-',
                            show_phases=show_phases, raw=True, figpath=path, R=R)

if __name__ == '__main__':
    visualize(set='test', subject=1001, activity=1)

