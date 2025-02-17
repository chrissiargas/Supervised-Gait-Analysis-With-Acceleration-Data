from datetime import datetime
from config_parser import Parser
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import pickle
import tensorflow as tf

from preprocessing.info import info
from preprocessing.resampling import resample
from preprocessing.utils import (impute, remove_g, produce, smooth, rescale,
                                 get_parameters, to_categorical, trim,separate, orient, is_irregular)
from preprocessing.splitting import split
from preprocessing.segments import finalize
from preprocessing.transformations import transformer, specter, fourier
import preprocessing.fft as fft
import random
from tqdm import tqdm
from plots import *

seed = 45
position = 'left_lower_arm'
dataset = None
subject = 1
activity = 2
start = 400
length = 400

def plot_all(data):
    for sub in sorted(pd.unique(data['subject_id'])):
        if sub > 100:
            act = 100 + activity
        else:
            act = activity

        plot_signal(data, position, dataset, sub, act, start, length,
                    show_events=second_plot, features='acc')

def plot_one(data):
    plot_signal(data, position, dataset, subject, activity, start, length,
                show_events=second_plot, features='acc')

def set_shuffle(set, idx):
    a, b, c = set
    assert len(a) == len(b) == len(c)

    shuffled_idx = []
    for group in idx.keys():
        group_idx = idx[group].tolist()
        group_indices = random.sample(group_idx, len(group_idx))
        shuffled_idx.extend(group_indices)


    return a[shuffled_idx], b[shuffled_idx], c[shuffled_idx]

class builder:
    def __init__(self, generate: bool = False):
        self.channels = None
        self.info = None
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
            if self.conf.dataset == 'marea':
                self.marea_extract()
            elif self.conf.dataset == 'nonan':
                self.nonan_extract()
            elif self.conf.dataset == 'synthetic':
                self.synthetic_extract()

        self.info = info(self.conf.dataset)
        self.data = pd.read_csv(self.extract_path)

    def __call__(self, verbose: bool = False):
        if self.conf.load_data:
            train, test, val = self.load()

        else:
            train, test, val, train_sgs, test_sgs, val_sgs = self.preprocess(verbose)
            self.save(train, test, val)

        if self.randomize:
            train_idx = separate(train)
            train = set_shuffle(train, train_idx)

        self.get_transformers()
        self.output_shape = train[1].shape[-1]
        self.output_type = tf.float32
        self.classes = self.conf.parameters

        self.train_size = train[0].shape[0]
        self.test_size = test[0].shape[0]
        self.val_size = val[0].shape[0]

        train = self.generate(train, train_sgs, training=True)
        test = self.generate(test, test_sgs, training=False)
        val = self.generate(val, val_sgs, training=False)

        return self.batch_prefetch(train, test, val)

    def generate(self, S: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 sgs: Optional[np.ndarray] = None, training: bool = True):
        X, Y, _ = S
        output_shape = () if self.output_shape == 1 else self.output_shape
        def gen():
            for i, (x_, y_) in enumerate(zip(X, Y)):
                x = self.transformer(x_, training)
                y = tf.convert_to_tensor(y_, dtype=tf.float32)
                y = tf.squeeze(y)

                if self.conf.fft:
                    f = self.fft_transformer(x, training)
                    yield (x, f), y
                if self.conf.spectrogram:
                    f_ = sgs[i]
                    f = self.fft_transformer(f_, training)
                    yield (x, f), y

                yield x, y

        if self.conf.fft or self.conf.spectrogram:
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
        train = train.shuffle(10000).repeat().batch(batch_size=self.conf.batch_size).prefetch(tf.data.AUTOTUNE)
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
        self.transformer = transformer(self.channels)
        self.input_shape = self.transformer.get_shape()

        if self.conf.fft:
            self.fft_transformer = fourier()
            self.fft_input_shape = self.fft_transformer.get_shape()
        elif self.conf.spectrogram:
            self.fft_transformer = specter()
            self.fft_input_shape = self.fft_transformer.get_shape()

    def preprocess(self, verbose: bool = False):
        verbose = True
        data = self.data.copy()
        data = data.drop(data.columns[0], axis=1)
        data = data.rename(columns={'time': 'timestamp', 'subject': 'subject_id', 'activity': 'activity_id'})

        drop_indices = data[(data['dataset'] == 'nonan')].index
        data = data.drop(drop_indices)

        data = trim(data, length = self.conf.trim_length)
        if verbose:
            plot_all(data)

        data = is_irregular(data, period=self.conf.length, checks=self.conf.checks)
        # if verbose:
        #     plot_all(data)

        data = orient(data, 'gravity', dim='2d')
        if verbose:
            plot_all(data)

        data = orient(data, 'pca', dim='2d')
        if verbose:
            plot_all(data)

        return

        data = remove_g(data, self.conf.fs, self.conf.include_g, self.conf.g_cutoff, how='lowpass')
        if verbose:
            plot_one(data)

        data = smooth(data, self.conf.filter, self.conf.filter_window, self.conf.fs, self.conf.filter_cutoff)
        if verbose:
            plot_one(data)

        data = produce(data, self.conf.new_features, self.conf.fs)
        if verbose:
            plot_one(data)

        data = rescale(data, self.conf.rescaler)
        if verbose:
            plot_one(data)

        data = get_parameters(data, self.conf.parameters, self.conf.calc_params)
        if verbose:
            plot_one(data)

        spectrograms = None
        train, test, train_sgs, test_sgs = split(data, self.conf.split_type, self.conf.test_hold_out, seed, spectrograms)
        if self.conf.validation:
            train, val, train_sgs, val_sgs = split(train, self.conf.split_type, self.conf.val_hold_out, seed, train_sgs)
        else:
            val, val_sgs = None, None

        if verbose:
            plot_one(train)

        train, test, val, train_sgs, test_sgs, val_sgs = self.to_windows(train, test, val, train_sgs, test_sgs, val_sgs)

        return train, test, val, train_sgs, test_sgs, val_sgs

    def to_windows(self, train, test, val, train_sgs = None, test_sgs = None, val_sgs = None):
        test_step = 'same'
        val_step = 'same'
        lowest_step = 1 if train_sgs is None else self.conf.nstride

        train, sizes, self.channels = finalize(train, self.conf.length, self.conf.step, self.conf.task, get_events=False)
        if train_sgs is not None:
            train_sgs = fft.segment(train_sgs, sizes, self.conf.length, self.conf.step, self.conf.nstride)

        which_set = val if self.conf.validation else test
        which_sgs = val_sgs if self.conf.validation else test_sgs
        which_step = self.conf.step if val_step == 'same' else self.conf.step // 10 if val_step == 'low' else lowest_step

        val, sizes, _ = finalize(which_set, self.conf.length, which_step, self.conf.task, get_events=False)
        if which_sgs is not None:
            val_sgs = fft.segment(which_sgs, sizes, self.conf.length, which_step, self.conf.nstride)

        which_step = self.conf.step if test_step == 'same' else self.conf.step // 10 if test_step == 'low' else lowest_step
        test, sizes, _ = finalize(test, self.conf.length, which_step, self.conf.task, get_events=False)
        if test_sgs is not None:
            test_sgs = fft.segment(test_sgs, sizes, self.conf.length, which_step, self.conf.nstride)

        print(train[0].shape, val[0].shape, test[0].shape)

        return train, test, val, train_sgs, test_sgs, val_sgs

    def nonan_extract(self, save: bool = False):
        self.info = info('nonan')
        subject_dir = sorted(os.listdir(self.info.path))
        df = pd.DataFrame()

        for sub_file in tqdm(subject_dir):
            if 'nonan' in sub_file:
                continue

            sub_path = os.path.join(self.info.path, sub_file)
            sub_df = self.nonan_load_subject(sub_path)
            sub_df = sub_df.sort_values(by=['activity', 'time'])

            to_file = os.path.join(self.path, sub_file)
            sub_df.to_csv(to_file)

            df = df._append(sub_df)
            del sub_df

        df['dataset'] = 'nonan'

        if save:
            df.to_csv(self.extract_path)

        return df

    def nonan_load_subject(self, path):
        df = pd.read_csv(path)

        position = self.conf.position
        if position == 'Wrist':
            position = 'LH'

        pos_imu = df.columns[df.columns.str.contains(position)]
        columns = df.columns.intersection([*self.info.indicators, *pos_imu, *self.info.phases, *self.info.events])
        df = df[columns]

        df = df.reset_index()
        df['position'] = self.conf.position
        df.columns = df.columns.str.replace('_' + self.conf.position, '')

        df = df.rename(columns={"accX": "acc_x", "accY": "acc_y", "accZ": "acc_z"})
        df = df.rename(columns={"gyrX": "gyr_x", "gyrY": "gyr_y", "gyrZ": "gyr_z"})
        df['position'] = df['position'].map(self.info.pos_pairs)

        for sensor in ['acc', 'gyr']:
            features = df.columns[df.columns.str.contains(sensor)]
            df.loc[(df[features] == 0).all(axis='columns'), features] = np.nan

        if self.conf.fs is not None:
            df = resample(df, self.info.initial_fs, self.conf.fs, how='decimate')

        df = df.astype({'position': str, 'subject': int,
                        'activity': int, 'time': float, 'is_NaN': bool,
                        'LF_HS': int, 'RF_HS': int, 'LF_TO': int, 'RF_TO': int,
                        'LF_stance': int, 'RF_stance': int,
                        'acc_x': float, 'acc_y': float, 'acc_z': float,
                        'gyr_x': float, 'gyr_y': float, 'gyr_z': float})

        return df

    def marea_extract(self, save: bool = True):
        self.info = info('marea')
        subject_dir = sorted(os.listdir(self.info.path))
        ds = pd.DataFrame()

        for sub_file in tqdm(subject_dir):
            if 'marea_full' in sub_file:
                continue

            sub_id = int(sub_file[4:-4])
            sub_path = os.path.join(self.info.path, sub_file)

            sub_df = self.marea_load_subject(sub_path)

            sub_df['subject'] = sub_id
            ds = ds._append(sub_df)

        ds = ds.rename(columns={"accX": "acc_x", "accY": "acc_y", "accZ": "acc_z"})

        activities = self.conf.activities
        fs = self.conf.fs

        if activities is not None:
            ds = ds[ds['activity'].str.contains('|'.join(activities))]

        ds['activity'] = ds['activity'].map(self.info.act_pairs)
        ds['position'] = ds['position'].map(self.info.pos_pairs)

        if fs is not None:
            ds = resample(ds, self.info.initial_fs, fs, how='resampy')

        ds = ds.astype({'position': 'str', 'subject': int,
                        'activity': int, 'time': float, 'is_NaN': bool,
                        'LF_HS': int, 'RF_HS': int, 'LF_TO': int, 'RF_TO': int,
                        'acc_x': float, 'acc_y': float, 'acc_z': float})

        ds = ds.sort_values(by=['subject', 'activity', 'time'])
        ds['dataset'] = 'marea'

        if save:
            ds.to_csv(self.extract_path)

        return ds

    def marea_load_subject(self, path):
        df = pd.read_csv(path)

        initial_activities = [
            "treadmill_walk", "treadmill_walknrun", "treadmill_slope_walk",
            "indoor_walk", "indoor_walknrun", "outdoor_walk", "outdoor_walknrun"
        ]
        events = self.info.events
        position = self.conf.position
        if position == 'LH':
            position = 'Wrist'

        acc_x = 'accX_' + position
        acc_y = 'accY_' + position
        acc_z = 'accZ_' + position
        columns = [acc_x, acc_y, acc_z, *initial_activities, *events]

        pos_df = df[df.columns.intersection(columns)].copy()
        pos_df = self.convert_activity(pos_df)

        pos_df = pos_df.reset_index()
        pos_df['time'] = df.index * (1000. / 128.)
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

    def synthetic_extract(self):
        nonan = self.nonan_extract(save=False)
        marea = self.marea_extract(save=False)

        nonan = nonan.rename(columns={'acc_x': 'acc_y', 'acc_y': 'acc_x'})
        nonan['acc_z'] *= -1.

        nonan['subject'] = nonan['subject'] + 100
        nonan['activity'] = nonan['activity'] + 100
        synthetic = pd.concat([marea, nonan], axis=0, ignore_index=True)
        ordered_cols = ['dataset', 'position', 'subject', 'activity', 'time', 'is_NaN',
                        'LF_HS', 'RF_HS', 'LF_TO', 'RF_TO', 'LF_stance', 'RF_stance',
                        'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        synthetic = synthetic[ordered_cols]
        synthetic.to_csv(self.extract_path)



