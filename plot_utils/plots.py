import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import *
import shutil
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.spatial.transform import Rotation

from concat import df_append

second_plot = True
print(f"Current working directory: {os.getcwd()}")

def get_color(name: str):
    if 'LF_HS' in name:
        if 'prob' in name:
            return 'red'
        elif 'raw' in name:
            return 'red'
        else:
            return 'red'
    elif 'LF_TO' in name:
        if 'prob' in name:
            return 'cyan'
        elif 'raw' in name:
            return 'cyan'
        else:
            return 'cyan'
    elif 'RF_HS' in name:
        if 'prob' in name:
            return 'deeppink'
        elif 'raw' in name:
            return 'deeppink'
        else:
            return 'deeppink'
    elif 'RF_TO' in name:
        if 'prob' in name:
            return 'darkviolet'
        elif 'raw' in name:
            return 'darkviolet'
        else:
            return 'darkviolet'

    if 'acc_x' in name:
        return 'royalblue'
    if 'acc_y' in name:
        return 'darkorange'
    if 'acc_z' in name:
        return 'green'

def plot_cummulative(dist_stats: Dict, return_plot: bool = False):
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Scores vs. Distance Error', fontsize=16)
    events = list(dist_stats.keys())

    # Plot metrics for each event
    for i, event in enumerate(events):
        ax = axs[i // 2, i % 2]  # Determine subplot position
        event_stats = dist_stats[event]

        # Extract metrics in distance order
        distances = sorted(event_stats.keys())
        f1_scores = [event_stats[d]['f1 score'] for d in distances]
        precisions = [event_stats[d]['precision'] for d in distances]
        recalls = [event_stats[d]['recall'] for d in distances]  # Note: check if your key is 'recall' or 'recall'

        # Plot lines
        ax.plot(distances, f1_scores, 'o-', label='F1-score', color='blue')
        ax.plot(distances, precisions, 's-', label='Precision', color='red')
        ax.plot(distances, recalls, 'd-', label='Recall', color='green')

        # Add milliseconds conversion on top axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(distances)
        ax2.set_xticklabels([f'{d * 20}' for d in distances])
        ax2.set_xlabel('Time Error (ms)', labelpad=10)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))

        # Customize subplot
        ax.set_title(f'{event} Event', fontsize=14)
        ax.set_xlabel('Distance Error (frames)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, 10.5)
        ax.grid(alpha=0.4, linestyle='--')
        ax.legend(loc='lower right')

        # Add conversion note
        ax.text(0.98, 0.02, '1 frame = 20ms',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()  # Adjust layout

    plt.legend()

    if return_plot:
        return fig


def plot_results(x: pd.DataFrame, start: Optional[int] = None, length: Optional[int] = None,
                 show_events: bool = False, show_phases: bool = False,
                 turn: Optional = None, raw: bool = False, real: bool = False, signal: bool = False,
                 figpath: Optional[str] = None, return_plot: bool = False):

    if figpath is None:
        figpath = os.path.join('archive', 'figures')

    if turn is not None:
        turn = str(turn)
    else:
        turn = ''

    feature_cols = x.columns[x.columns.str.contains('acc')]
    events_cols = x.columns[x.columns.str.contains('HS|TO')]
    phases_cols = x.columns[x.columns.str.contains('stance')]

    t = x['timestamp'].values
    sgn = x[feature_cols].values
    evs = x[events_cols].values
    phases = x[phases_cols].values

    if start is not None and length is not None:
        t = t[start: start + length]
        t = pd.to_datetime(t, unit='ms')
        sgn = sgn[start: start + length]
        evs = evs[start: start + length]
        phases = phases[start: start + length] * 10

    fig, axs = plt.subplots(1, sharex=True, figsize=(40, 15))

    if signal:
        axs.plot( t, sgn, linewidth=1, label=feature_cols)

    if show_events:
        for ev, name in zip(evs.transpose(), events_cols):
            color = get_color(name)

            if 'prob' in name:
                axs.plot(t, ev * 10., linewidth=2, linestyle='solid', label=name,
                         color=color, alpha=0.6)
                continue

            elif 'raw' in name and raw:
                ev_ixs = np.where(ev == 1)
                axs.vlines(t[ev_ixs], 0, 1, transform=axs.get_xaxis_transform(),
                           linewidth=1, linestyle='solid', label=name, colors=color)

            elif 'real' in name and real:
                axs.plot(t, ev * 10., linewidth=2, linestyle='dashed', label=name)
                continue

    if show_phases:
        pass

    plt.legend()

    if return_plot:
        return fig
    else:
        filepath = os.path.join(figpath, turn + '-' + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
        plt.savefig(filepath, format="png", bbox_inches="tight")

    plt.close()


def plot_cycle(x: pd.DataFrame, show_events: bool = False, show_phases: bool = False,
               turn: Optional = None, raw: bool = False, real: bool = False,
               figpath: Optional[str] = None, feature_hue: bool = False,
               events_hue: bool = False, return_plot: bool = False):
    length = 100

    if figpath is None:
        figpath = os.path.join('archive', 'figures')

    if turn is not None:
        turn = str(turn)
    else:
        turn = ''

    feature_cols = x.columns[x.columns.str.contains('acc')]
    events_cols = x.columns[x.columns.str.contains('HS|TO')]
    phases_cols = x.columns[x.columns.str.contains('stance')]

    cycles = [cycle for _, cycle in x.groupby('cycle')]

    sgn_cycles = None
    evs_cycles = None
    phases_cycles = np.zeros((len(cycles), length, len(phases_cols)))

    for c, cycle in enumerate(cycles):
        if cycle[events_cols].isnull().values.sum() > 0:
            continue

        old_t = cycle.reset_index().index
        new_t = np.linspace(start=old_t[0], stop=old_t[-1], num=length)

        old_sgn = cycle[feature_cols].values
        new_sgn = interp1d(old_t, old_sgn, kind='linear', axis=0, fill_value='extrapolate')(new_t)
        sgn_cycle = pd.DataFrame(new_sgn, columns=feature_cols)
        sgn_cycle['timestamp'] = np.arange(length)
        sgn_cycle['cycle'] = c

        sgn_cycles = df_append(sgn_cycles, sgn_cycle)

        if show_events:
            old_events = cycle[events_cols].values
            new_events = interp1d(old_t, old_events, kind='linear', axis=0, fill_value='extrapolate')(new_t)
            evs_cycle = pd.DataFrame(new_events, columns=events_cols) * 10
            evs_cycle['timestamp'] = np.arange(length)
            evs_cycle['cycle'] = c

            evs_cycles = df_append(evs_cycles, evs_cycle)

        if show_phases:
            pass

    if show_events:
        evs = np.mean(evs_cycles, axis=0)

    if show_phases:
        phases = np.mean(phases_cycles, axis=0)

    fig, axs = plt.subplots(1, sharex=True, figsize=(40, 15))

    for f, feature in enumerate(feature_cols):
        if feature_hue:
            sns.lineplot(data=sgn_cycles, x='timestamp', y=feature, hue='cycle', ax=axs, legend=False, alpha=0.15,
                         palette=sns.color_palette([get_color(feature)], len(sgn_cycles['cycle'].unique())))

        else:
            sns.lineplot(data=sgn_cycles, x='timestamp', y=feature, ax=axs, label=feature, color=get_color(feature),
                         errorbar = lambda x: (np.mean(x) - 1 * np.std(x), np.mean(x) + 1 * np.std(x)))

    if show_events:

        for event in events_cols:
            if 'prob' in event:
                if events_hue:
                    sns.lineplot(data=evs_cycles, x='timestamp', y=event, hue='cycle', ax=axs, legend=False,
                                 palette=sns.color_palette([get_color(event)], len(sgn_cycles['cycle'].unique())),
                                 alpha=0.15)

                else:
                    sns.lineplot(data=evs_cycles, x='timestamp', y=event, ax=axs, color=get_color(event), legend=False)

            elif 'raw' in event and raw:
                event_values = evs_cycles.groupby('timestamp').mean()[event].values
                peak_ix = np.argmax(event_values)
                std = np.std(event_values)

                axs.vlines(peak_ix, 0, 1, transform=axs.get_xaxis_transform(),
                           linewidth=2, linestyle='solid', label=event, colors=get_color(event))

                axs.vlines(peak_ix + std, 0, 1, transform=axs.get_xaxis_transform(),
                           linewidth=2, linestyle='dashed', colors=get_color(event))

                axs.vlines(peak_ix - std, 0, 1, transform=axs.get_xaxis_transform(),
                           linewidth=2, linestyle='dashed', colors=get_color(event))

            elif 'real' in event and real:
                sns.lineplot(data=evs_cycles, x='timestamp', y=event, ax=axs, label=event)

    if show_phases:
        pass

    plt.legend()

    if return_plot:
        return fig
    else:
        filepath = os.path.join(figpath, turn + '-' + 'cycle.png')
        plt.savefig(filepath, format="png", bbox_inches="tight")

    plt.close()


def plot_signal(x: pd.DataFrame, pos: Optional[str] = None, dataset: Optional[str] = None,
                subject: Optional[int] = None, activity: Optional[int] = None,
                mode: Optional[str] = None, population: Optional[str] = None,
                start: Optional[int] = None, length: Optional[int] = None,
                show_events: bool = False, show_phases: bool = False,
                features: Optional[str] = None, sign: bool = False,
                turn: Optional = None, raw: bool = True, real: bool = False,
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

    features = x.columns[x.columns.str.contains(features)]

    if 'position' in x.columns:
        positions = pd.unique(x['position'])
        has_pos_col = True
    else:
        acc_features = x.columns[x.columns.str.contains('acc')]
        positions = set(acc_features.str.split('_').str[2])
        has_pos_col = False

    events_cols = x.columns[x.columns.str.contains('HS|TO')]
    phases_cols = x.columns[x.columns.str.contains('stance')]

    if pos is not None and pos in positions:
        positions = [pos]

    fig, axs = plt.subplots(len(positions), sharex=True, figsize=(10, 8))

    for p, position in enumerate(positions):
        if has_pos_col:
            pos_features = features
            sig = x[x['position'] == position].values
        else:
            pos_features = features[features.str.contains(position)]
            sig = x[pos_features]

        t = x['timestamp'].values
        evs = x[events_cols].values
        phases = x[phases_cols].values

        if start is not None and length is not None:
            t = t[start: start + length]
            t = pd.to_datetime(t, unit='ms')
            sig = sig[start: start + length]
            evs = evs[start: start + length]
            phases = phases[start: start + length] * 10

            # if subject in [2001, 2002, 2010]:
            #     print('a')
            #     Rot = [[-1, 0, 0],
            #            [0, 1, 0],
            #            [0, 0, 1]]
            #     sig = Rotation.from_matrix(Rot).apply(sig)

            if sign:
                sig = np.sign(np.mean(sig, axis=0)) * sig

        axs[p].plot(t, sig, linewidth=1, label=pos_features)

        if show_events:
            for ev, name in zip(evs.transpose(), events_cols):
                if 'prob' in name:
                    axs[p].plot(t, ev * 10., linewidth=2, linestyle='solid', label=name)
                    continue

                elif 'pred' in name:
                    continue

                elif 'raw' in name and raw:
                    ev_ixs = np.where(ev == 1)
                    axs[p].vlines(t[ev_ixs], 0, 1, transform=axs[p].get_xaxis_transform(),
                               linewidth=1, linestyle='solid', colors=get_color(name), label=name)

                elif 'real' in name and real:
                    axs[p].plot(t, ev * 10., linewidth=2, linestyle='dashed', label=name)
                    continue

                else:
                    ev_ixs = np.where(ev == 1)
                    axs[p].vlines(t[ev_ixs], 0, 1, transform=axs[p].get_xaxis_transform(),
                               linewidth=1, linestyle='solid', colors=get_color(name), label=name)
                    axs[p].legend(loc="upper right")

        elif show_phases:
            for phase, name in zip(phases.transpose(), phases_cols):
                if 'prob' in name:
                    axs[p].plot(t, phase, linewidth=2, linestyle='solid', label=name)

                elif 'pred' in name:
                    axs[p].plot(t, phase, linewidth=2, linestyle='solid', label=name)

                else:
                    axs[p].plot(t, phase, linewidth=2, linestyle='dashed', label=name)

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
