from datetime import datetime

from pandas.core.config_init import val_mca

from config_parser import Parser
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Dict
import pickle
import tensorflow as tf

from preprocessing.info import info
from preprocessing.resampling import resample
from preprocessing.utils import impute, remove_g, produce, smooth, rescale, get_parameters
from preprocessing.splitting import split
from preprocessing.segments import finalize
from preprocessing.transformations import transformer, fft_transformer

seed = 45
figpath = os.path.join('archive', 'figures')
pd.set_option('display.max_rows', 1000)
second_plot = True

def set_shuffle(set):
    a, b, c = set
    assert len(a) == len(b) == len(c)
    idx = np.random.permutation(len(a))
    return a[idx], b[idx], c[idx]


def plot_signal(x: pd.DataFrame, pos: str, subject: int = 5,
                start: Optional[int] = None, length: Optional[int] = None,
                show_events: bool = False):
    x = x[x['subject'] == subject]

    positions = pd.unique(x['position'])
    features = x.columns[x.columns.str.contains("acc|norm|jerk|low|angle")]
    all_events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

    if pos in positions:
        x = x[x['position'] == pos]

        t = x['timestamp'].values
        sig = x[features].values
        evs = x[all_events].values

        if start is not None and length is not None:
            t = t[start: start + length]
            sig = sig[start: start + length]
            evs = evs[start: start + length]

        sig = np.array(sig, dtype=np.float32)

        fig, axs = plt.subplots(1, sharex=True, figsize=(20, 15))
        axs.plot(sig, linewidth=1, label=features)

        if show_events:
            colors = ['b', 'k', 'r', 'g']
            for color, ev, name in zip(colors, evs.transpose(), all_events):
                ev_ixs = np.where(ev == 1)
                axs.vlines(ev_ixs, 0, 1, transform=axs.get_xaxis_transform(), colors=color,
                           linewidth=1, linestyles='dashed', label=name)

        plt.legend()
        filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".png")
        plt.savefig(filepath, format="png", bbox_inches="tight")

def plot_parameters(x: pd.DataFrame, pos: str, subject: int = 5, activity: str = 'treadmill_walking',
                    start: Optional[int] = None, length: Optional[int] = None,
                    show_events: bool = False):

    x = x[x['subject'] == subject]
    x = x[x['activity'] == activity]

    positions = pd.unique(x['position'])
    parameters = x.columns[x.columns.str.contains("left|right|step")]
    all_events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

    if pos in positions:
        x = x[x['position'] == pos]

        t = x['timestamp'].values
        params = x[parameters].to_numpy()
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


class builder:
    def __init__(self, generate: bool = False):
        self.output_type = None
        self.output_shape = None
        self.input_type = None
        self.input_shape = None
        self.transformer = None
        self.train_dict = None
        self.randomize = True

        config = Parser()
        config.get_args()
        self.conf = config
        self.info = info()

        self.path = os.path.join(
            os.path.expanduser('~'),
            self.conf.path,
            'wrist_gait'
        )

        self.extract_path = os.path.join(
            self.path,
            'extracted.csv'
        )

        if generate:
            self.extract()
            self.data = pd.read_csv(self.extract_path)

        else:
            self.data = pd.read_csv(self.extract_path)

    def __call__(self, verbose: bool = False):
        if self.conf.load_data:
            train, test, val = self.load()

        else:
            train, test, val = self.preprocess(verbose)
            self.save(train, test, val)

        if self.randomize:
            set_shuffle(train)
            set_shuffle(test)
            set_shuffle(val)

        self.get_transformers()
        self.output_shape = train[1].shape[-1]
        self.output_type = tf.float32
        self.classes = self.conf.parameters

        self.train_size = train[0].shape[0]
        self.test_size = test[0].shape[0]
        self.val_size = val[0].shape[0]

        train = self.generate(train, training=True)
        test = self.generate(test, training=False)
        val = self.generate(val, training=False)

        return self.batch_prefetch(train, test, val)

    def generate(self, S: Tuple[np.ndarray, np.ndarray, np.ndarray], training: bool = True):
        X, Y, _ = S
        output_shape = () if self.output_shape == 1 else self.output_shape
        def gen():
            for x, y in zip(X, Y):
                x_ = self.transformer(x, training)
                y_ = tf.convert_to_tensor(y, dtype=tf.float32)
                y_ = tf.squeeze(y_)

                if self.conf.fft:
                    x_fft = self.fft_transformer(x, training)
                    yield (x_, x_fft), y_
                else:
                    yield x_, y_

        if self.conf.fft:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=((tf.float32, tf.float32), tf.float32),
                output_shapes=((self.input_shape, self.fft_input_shape), output_shape),
            )
        else:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=(tf.float32, tf.float32),
                output_shapes=(self.input_shape, output_shape)
            )

    def batch_prefetch(self, train, test, val):
        train = train.shuffle(1000).repeat().batch(batch_size=self.conf.batch_size).prefetch(tf.data.AUTOTUNE)
        test = test.batch(batch_size=self.conf.batch_size).prefetch(tf.data.AUTOTUNE)
        val = val.batch(batch_size=self.conf.batch_size).prefetch(tf.data.AUTOTUNE)

        return train, test, val

    def load(self):
        load_path = os.path.join('attic', 'data', 'data.pkl')

        pkl_file = open(load_path, 'rb')
        my_data = pickle.load(pkl_file)
        train = my_data['train_X'], my_data['train_Y'], my_data['train_T']
        test = my_data['test_X'], my_data['test_Y'], my_data['test_T']

        if self.conf.validation:
            val = my_data['val_X'], my_data['val_Y'], my_data['val_T']
        else:
            val = None

        pkl_file.close()

        return train, test, val

    def save(self, train: Tuple[np.ndarray, np.ndarray, np.ndarray],
             test: Tuple[np.ndarray, np.ndarray, np.ndarray],
             val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
        save_path = os.path.join('attic', 'data', 'data.pkl')

        my_data = {'train_X': train[0],
                   'train_Y': train[1],
                   'train_T': train[2],
                   'val_X': val[0],
                   'val_Y': val[1],
                   'val_T': val[2],
                   'test_X': test[0],
                   'test_Y': test[1],
                   'test_T': test[2]}

        output = open(save_path, 'wb')
        pickle.dump(my_data, output)

        output.close()

    def get_transformers(self):
        self.transformer = transformer()
        self.fft_transformer = fft_transformer()
        self.input_shape = self.transformer.get_shape()
        self.fft_input_shape = self.fft_transformer.get_shape()

    def preprocess(self, verbose: bool = False):
        data = self.data.copy()
        data = data.drop(data.columns[0], axis=1)

        if verbose:
            position = self.info.pos_pairs[self.conf.position]
            subject = 15
            start = 5000

        if verbose:
            plot_signal(data, position, subject, start, 400, show_events=second_plot)

        data = impute(data, self.conf.cleaner)

        if verbose:
            plot_signal(data, position, subject, start, 400, show_events=second_plot)

        data = remove_g(data, self.conf.fs, self.conf.include_g)

        if verbose:
            plot_signal(data, position, subject, start, 400, show_events=second_plot)

        data = smooth(data, self.conf.filter, self.conf.filter_window, self.conf.fs, self.conf.filter_cutoff)

        if verbose:
            plot_signal(data, position, subject, start, 400, show_events=second_plot)

        data = produce(data, self.conf.new_features, self.conf.fs)

        if verbose:
            plot_signal(data, position, subject, start, 400, show_events=second_plot)

        data = rescale(data, self.conf.rescaler)

        if verbose:
            plot_signal(data, position, subject, start, 400, show_events=second_plot)

        data = get_parameters(data, self.conf.parameters)

        if verbose:
            plot_parameters(data, position, subject, 'treadmill_walking', start, 400,
                            show_events=True)

        train, test = split(data, self.conf.split_type, self.conf.test_hold_out, seed)

        if self.conf.validation:
            train, val = split(train, self.conf.split_type, self.conf.val_hold_out, seed)
        else:
            val = None

        if verbose:
            plot_signal(train, position, subject, start, 400, show_events=second_plot)

        train, test, val = self.to_windows(train, test, val)

        if verbose:
            plot_window(train, self.train_dict, subject, self.train_dict['treadmill_walking'], search=100, plot_y=False)

        return train, test, val

    def to_windows(self, train, test, val):
        test_step = 'min'
        val_step = 'same'

        train, self.train_dict = finalize(train, self.conf.length, self.conf.step, self.conf.task, get_events=False)

        which_set = val if self.conf.validation else test
        which_step = self.conf.step if val_step == 'same' else self.conf.step // 5 if val_step == 'low' else 1
        val, _ = finalize(which_set, self.conf.length, which_step, self.conf.task, get_events=False)

        which_step = self.conf.step if test_step == 'same' else self.conf.step // 5 if test_step == 'low' else 1
        test, _ = finalize(test, self.conf.length, which_step, self.conf.task, get_events=False)

        print(train[0].shape, val[0].shape, test[0].shape)

        return train, test, val

    def extract(self):
        subject_dir = os.listdir(self.info.path)
        ds = pd.DataFrame()

        for sub_file in subject_dir:
            if 'marea_full' in sub_file:
                continue

            sub_id = int(sub_file[4:-4])
            sub_path = os.path.join(self.info.path, sub_file)

            sub_df = self.marea_load_subject(sub_path)

            sub_df['subject'] = sub_id
            ds = ds._append(sub_df)

        ds = ds.rename(columns={"accX": "acc_x", "accY": "acc_y", "accZ": "acc_z"})
        ds = ds.astype({'timestamp': float, 'subject': int,
                        'activity': str, 'position': str,
                        'acc_x': float, 'acc_y': float, 'acc_z': float,
                        'LF_HS': int, 'RF_HS': int, 'LF_TO': int, 'RF_TO': int})

        activities = self.conf.activities
        fs = self.conf.fs

        ds['activity'] = ds['activity'].map(self.info.act_pairs)
        ds['position'] = ds['position'].map(self.info.pos_pairs)

        if activities is not None:
            ds = ds[ds['activity'].str.contains('|'.join(activities))]

        if fs is not None:
            ds = resample(ds, self.info.initial_fs, fs)

        ds = ds.sort_values(by=['subject', 'timestamp'])
        ds.to_csv(self.extract_path)

    def marea_load_subject(self, path):
        df = pd.read_csv(path)

        initial_activities = [
            "treadmill_walk", "treadmill_walknrun", "treadmill_slope_walk",
            "indoor_walk", "indoor_walknrun", "outdoor_walk", "outdoor_walknrun"
        ]
        events = self.info.events
        position = self.conf.position

        acc_x = 'accX_' + position
        acc_y = 'accY_' + position
        acc_z = 'accZ_' + position
        columns = [acc_x, acc_y, acc_z, *initial_activities, *events]

        pos_df = df[df.columns.intersection(columns)].copy()
        pos_df = self.convert_activity(pos_df)

        pos_df = pos_df.reset_index()
        pos_df['timestamp'] = df.index * (1000. / 128.)
        pos_df = pos_df.drop(['index'], axis=1)
        pos_df.columns = pos_df.columns.str.replace('_' + position, '')
        pos_df['position'] = position

        return pos_df

    def convert_activity(self, x):

        def row_split(row, place):
            if row[place + '_walknrun'] == 1 and row[place + '_walk'] == 0:
                return 1
            return 0

        places = ['treadmill', 'indoor', 'outdoor']
        for place in places:
            if place + '_walknrun' in x.columns:
                x[place + '_run'] = x.apply(lambda row: row_split(row, place), axis=1)

        walknrun_cols = [col for col in x if col.endswith('walknrun')]
        x = x.drop(walknrun_cols, axis=1)

        activities = x[x.columns.intersection(self.info.activities)]
        x['activity'] = activities.idxmax(axis=1)
        x.loc[~activities.any(axis='columns'), 'activity'] = 'undefined'

        x = x.drop(x.columns.intersection(self.info.activities), axis=1)

        return x



