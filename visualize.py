import matplotlib.pyplot as plt

from config_parser import Parser
import os
from models.supervised import predictor
from preprocessing.building import builder
from plots import plot_signal
from typing import Optional
import numpy as np
from datetime import datetime

def visualize(path: Optional[str] = None):
    data = builder()
    data()

    config = Parser()
    config.get_args()

    model_dir = f'archive/models/{config.architecture}'
    model_file = '%s.weights.h5' % config.architecture
    model_file = f'{model_dir}/{model_file}'

    model = predictor(data)
    model.compile()
    model.build(data.input_shape)
    model.load_weights(model_file)

    set = 'test'
    subject = 3
    activity = 3

    df, _, _ = data.compare_yy_(model, which=set, subject=subject, activity=activity)

    show_events = True if config.task == 'gait_events' else False
    show_phases = True if config.task == 'gait_phases' else False

    for i, start in enumerate(range(0, df.shape[0], 1000)):
        plot_signal(df, 'left_lower_arm', subject=subject,
                    activity=activity, start=start, length=1000,
                    show_events=show_events, features='acc', turn=i,
                    show_phases=show_phases, raw=True, figpath=path)

def rotation_by_axis(angle_deg, axis='z'):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return R

def visualize_rot(path: Optional[str] = None):
    data = builder()
    data()

    config = Parser()
    config.get_args()

    model_dir = f'archive/models/{config.architecture}'
    model_file = '%s.weights.h5' % config.architecture
    model_file = f'{model_dir}/{model_file}'

    model = predictor(data)
    model.compile()
    model.build_model(data.input_shape)
    model.load_weights(model_file)

    set = 'test'
    subject = 3
    activity = 3

    start = 4000
    window = 1000

    angles = np.arange(0, 360, 10)  # Rotate 0-360° in 10° steps
    for axis in ['y']:
        for angle in angles:
            R = rotation_by_axis(angle, axis=axis)

            df, X, X_rot = data.compare_yy_(model, which=set, subject=subject, activity=activity,
                                            start=start, end=start+window, rotation=R, rotated=True)

            show_events = True if config.task == 'gait_events' else False
            show_phases = True if config.task == 'gait_phases' else False

            for i, start_ in enumerate(range(0, df.shape[0], window)):
                plot_signal(df, 'left_lower_arm', subject=subject,
                            activity=activity, start=start_, length=window,
                            show_events=show_events, features='acc', turn=f'{axis}-{angle}-',
                            show_phases=show_phases, raw=True, figpath=path)

            fig, axs = plt.subplots(2, sharex=True, figsize=(40, 15))
            axs[0].plot(X[0], linewidth=1)
            axs[1].plot(X_rot[0], linewidth=1)

            plt.legend()
            filepath = os.path.join('archive', 'figures', str(subject) + '-' + f'rotate-{angle}' + '-' + datetime.now().strftime(
                "%Y%m%d-%H%M%S-%f") + ".png")
            plt.savefig(filepath, format="png", bbox_inches="tight")
            plt.close()

if __name__ == '__main__':
    visualize_rot()

