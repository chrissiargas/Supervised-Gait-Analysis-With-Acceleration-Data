import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import *
import shutil

second_plot = True
print(f"Current working directory: {os.getcwd()}")

def plot_signal(x: pd.DataFrame, pos: str, dataset: Optional[str] = None,
                subject: Optional[int] = None, activity: Optional[int] = None,
                mode: Optional[str] = None, population: Optional[str] = None,
                start: Optional[int] = None, length: Optional[int] = None,
                show_events: bool = False, show_phases: bool = False,
                features: Optional[str] = None, sign: bool = False,
                turn: Optional = None, raw: bool = False,
                figpath: Optional[str] = None, R: Optional[np.ndarray] = None):

    if figpath is None:
        figpath = os.path.join('archive', 'figures')

    if subject is not None:
        x = x[x['subject_id'] == subject]
    if activity is not None:
        x = x[x['activity_id'] == activity]
    if dataset is not None:
        x = x[x['dataset'] == dataset]
    if mode is not None:
        x = x[x['mode'] == mode]
    if population is not None:
        x = x[x['population'] == population]

    if features is None:
        features = 'acc|norm|jerk|angle'

    if turn is not None:
        turn = str(turn)
    else:
        turn = ''

    positions = pd.unique(x['position'])
    features = x.columns[x.columns.str.contains(features)]
    events_cols = x.columns[x.columns.str.contains('HS|TO')]
    phases_cols = x.columns[x.columns.str.contains('stance')]

    if pos in positions:
        x = x[x['position'] == pos]

        t = x['timestamp'].values
        sig = x[features].values
        evs = x[events_cols].values
        phases = x[phases_cols].values

        if start is not None and length is not None:
            t = t[start: start + length]
            t = pd.to_datetime(t, unit='ms')
            sig = sig[start: start + length]
            evs = evs[start: start + length]
            phases = phases[start: start + length] * 10

            if sign:
                sig = np.sign(np.mean(sig, axis=0)) * sig

        fig, axs = plt.subplots(1, sharex=True, figsize=(40, 15))
        axs.plot( t, sig, linewidth=1, label=features)

        if show_events:
            for ev, name in zip(evs.transpose(), events_cols):
                if 'prob' in name:
                    axs.plot(t, ev * 10., linewidth=2, linestyle='solid', label=name)
                    continue

                elif 'pred' in name:
                    continue

                elif 'raw' in name and raw:
                    ev_ixs = np.where(ev == 1)
                    axs.vlines(t[ev_ixs], 0, 1, transform=axs.get_xaxis_transform(),
                               linewidth=1, linestyle='solid', label=name)

                elif 'real' in name:
                    axs.plot(t, ev * 10., linewidth=2, linestyle='dashed', label=name)
                    continue

                else:
                    ev_ixs = np.where(ev == 1)
                    axs.vlines(t[ev_ixs], 0, 1, transform=axs.get_xaxis_transform(),
                               linewidth=1, linestyle='solid', label=name)


        elif show_phases:
            for phase, name in zip(phases.transpose(), phases_cols):
                if 'prob' in name:
                    axs.plot(t, phase, linewidth=2, linestyle='solid', label=name)

                elif 'pred' in name:
                    axs.plot(t, phase, linewidth=2, linestyle='solid', label=name)

                else:
                    axs.plot(t, phase, linewidth=2, linestyle='dashed', label=name)

        if R is not None:
            table = axs.table(
                cellText=np.around(R, 2),
                loc=2,
                cellLoc='center',
                colLabels=['X', 'Y', 'Z'],
                rowLabels=['X’', 'Y’', 'Z’'],
                bbox=[0, 0.9, 0.06, 0.1],
                edges='open'
            )
            table.scale(1, 2)
            axs.axis('off')

        plt.legend()
        filepath = os.path.join(figpath, str(subject) + '-' + turn + '-' + datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".png")
        plt.savefig(filepath, format="png", bbox_inches="tight")
        plt.close()

def plot_parameters(x: pd.DataFrame, pos: str, subject: int = 5, activity: int = 0,
                    start: Optional[int] = None, length: Optional[int] = None,
                    show_events: bool = False, features: Optional[str] = None):

    x = x[x['subject_id'] == subject]

    positions = pd.unique(x['position'])

    if features is None:
        features = 'stance|swing|step'

    parameters = x.columns[x.columns.str.contains(features)]
    all_events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

    if pos in positions:
        x = x[x['position'] == pos]

        t = x['timestamp'].values
        params = x[parameters].to_numpy()
        params = np.nan_to_num(params, nan=0)
        params = params.astype(bool).astype(int)
        evs = x[all_events].values

        if start is not None and length is not None:
            t = t[start: start + length]
            params = params[start: start + length]
            evs = evs[start: start + length]

        fig, axs = plt.subplots(1, sharex=True, figsize=(20, 15))
        axs.plot(params, linewidth=1, label = parameters)

        if show_events:
            colors = ['b', 'k', 'r', 'g']
            for color, ev, name in zip(colors, evs.transpose(), all_events):
                ev_ixs = np.where(ev == 1)
                axs.vlines(ev_ixs, 0, 1, transform=axs.get_xaxis_transform(), colors=color,
                           linewidth=1, linestyles='dashed', label=name)

        plt.legend()
        filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".png")
        plt.savefig(filepath, format="png", bbox_inches="tight")
        plt.close()

def plot_window(S: Tuple[np.ndarray, np.ndarray, np.ndarray], act_dict: Dict,
                subject: int = 5, activity: int = 0,
                search: Optional[float] = None, plot_y: bool = False):
    if plot_y:
        X, _, Y, T = S
    else:
        X, Y, T = S

    idx = np.argwhere(T[:, 0] == subject).squeeze()
    X = X[idx]
    Y = Y[idx]
    T = T[idx]

    idx = np.argwhere(T[:, 1] == activity).squeeze()
    X = X[idx]
    Y = Y[idx]
    T = T[idx]

    idx = np.argwhere((T[:, 2] < search) & (search < T[:, 3])).squeeze()

    if len(idx.shape) > 0:
        idx = idx[-1]

    window = X[idx]
    y = Y[idx]

    fig, axs = plt.subplots(1, sharex=True, figsize=(20, 15))
    axs.plot(window, linewidth=0.5)

    if plot_y:
        colors = ['b', 'k', 'r', 'g']
        for color, ev, name in zip(colors, y.transpose(), ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']):
            ev_ixs = np.where(ev == 1)
            axs.vlines(ev_ixs, 0, 1, transform=axs.get_xaxis_transform(), colors=color,
                       linewidth=1, linestyles='dashed', label=name, alpha=0.5)

    plt.legend()
    filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    plt.savefig(filepath, format="png", bbox_inches="tight")
    plt.close()

def plot_window_to_spectrogram(S: Tuple[np.ndarray, np.ndarray, np.ndarray], sgs, idx):
    X, Y, T = S
    window = X[idx]
    feature = 0

    fig, axs = plt.subplots(1, sharex=True, figsize=(20, 15))
    axs.plot(window[:, feature], linewidth=0.5)

    plt.legend()
    filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    plt.savefig(filepath, format="png", bbox_inches="tight")
    plt.close()

    sg = sgs[idx, feature, ...]

    plt.imshow(sg)
    filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    plt.savefig(filepath, format="png", bbox_inches="tight")
    plt.close()

    fs = 50
    nperseg = 80
    noverlap = 75
    resize = False
    log_power = False
    sg, _, _ = fft.to_spectrogram(window[:,feature], fs, nperseg, noverlap, resize, log_power)

    plt.imshow(sg[0])
    filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    plt.savefig(filepath, format="png", bbox_inches="tight")
    plt.close()

def plot_spectrogram(spectrograms, f, t, subject, activity, start, length, colormesh = False):
    n_ticks = 10
    ylabel = "f (Hz) "
    xlabel = "t (s)"
    plt.figure(figsize=(30, 10))

    feature = 1
    sg = spectrograms[subject][activity][feature, :, start:start + length]
    t = t[start:start + length]

    if colormesh:
        plt.pcolormesh(t, f, sg, shading='gouraud')
    else:
        matrix_shape = sg.shape
        time_list = [f'{t[i]:.0f}' for i in np.round(np.linspace(0, matrix_shape[1] - 1, n_ticks)).astype(int)]
        freq_list = [f'{f[i]:.1f}' for i in np.round(np.linspace(0, matrix_shape[0] - 1, n_ticks)).astype(int)]
        plt.xticks(np.linspace(0, matrix_shape[1] - 1, n_ticks), time_list)
        plt.yticks(np.linspace(0, matrix_shape[0] - 1, n_ticks), freq_list)
        plt.imshow(sg)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    plt.savefig(filepath, format="png", bbox_inches="tight")
    plt.close()
