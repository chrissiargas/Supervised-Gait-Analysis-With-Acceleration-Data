from datetime import datetime
import keras
from config_parser import Parser
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import pickle
import tensorflow as tf
from preprocessing.info import info
from preprocessing.resampling import resample
from preprocessing.utils import (impute, remove_g, produce, smooth, get_parameters,
                                 trim,separate, orient, is_irregular)
from preprocessing.splitting import split, split_all
from preprocessing.segments import finalize
from preprocessing.transformations import transformer
import random
from tqdm import tqdm
from plots import *
from preprocessing.extract import extractor
from scipy.spatial.transform import Rotation

seed = 45
position = 'left_lower_arm'
dataset = None
subject = 1
activity = 1
start = 11500
length = 1000
features = 'acc'
mode = None
population = None
second_plot = True

figpath = os.path.join('archive', 'figures')

def plot_all(data):
    for sub in sorted(pd.unique(data['subject_id'])):
        plot_signal(data, position, dataset, sub, activity, mode, population,
                    start, length, show_events=second_plot, features=features)

def plot_one(data):
    for i, start in enumerate(range(0, 24000, 1000)):
        plot_signal(data, position, dataset, subject, activity, mode, population,
                    start, length,
                    show_events=second_plot, features=features, turn=i)

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
        self.classes = None
        self.channels = None
        self.info = None
        self.output_type = None
        self.output_shape = None
        self.input_type = None
        self.input_shape = None
        self.transformer = None
        self.train_dict = None
        self.randomize = True
        self.data = None
        self.class_weights = {}

        config = Parser()
        config.get_args()
        self.conf = config

        try:
            for f in os.listdir(figpath):
                os.unlink(os.path.join(figpath,f))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        self.path = os.path.join(
            os.path.expanduser('~'),
            self.conf.path,
            'wrist_gait'
        )

        if generate:
            extract_data = extractor(dataset=self.conf.dataset)
            extract_data()

        self.load_data()

    def load_data(self):
        subject_dir = sorted(os.listdir(self.path))
        for sub_file in tqdm(subject_dir):
            sub_path = os.path.join(self.path, sub_file)
            sub_df = pd.read_csv(sub_path)

            if self.data is None:
                self.data = sub_df
            else:
                self.data = pd.concat([self.data, sub_df], axis=0, ignore_index=True)

        self.data = self.data.sort_values(by=['dataset', 'subject'])

    def __call__(self):
        train, test, val = self.preprocess()

        if self.randomize:
            train_idx = separate(train)
            train = set_shuffle(train, train_idx)

        self.get_transformers()
        self.get_shapes(train[1])
        self.get_class_weights(train[1])

        self.train_size = train[0].shape[0]
        self.test_size = test[0].shape[0]
        self.val_size = val[0].shape[0]

        train = self.generate(train, training=True)
        test = self.generate(test, training=False)
        val = self.generate(val, training=False)

        return self.batch_prefetch(train, test, val)

    def generate(self, S: Tuple[np.ndarray, np.ndarray, np.ndarray], training: bool = True):
        X, Y, _ = S

        def gen():
            for i, (x, y) in enumerate(zip(X, Y)):
                x = self.transformer(x)
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                y = tf.squeeze(y)

                yield x, y

        return tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.input_shape, self.output_shape)
        )

    def batch_prefetch(self, train, test, val):
        train = (train.
                 shuffle(10000).
                 repeat().
                 batch(batch_size=self.conf.batch_size).
                 prefetch(tf.data.AUTOTUNE))

        test = (test.
                batch(batch_size=self.conf.batch_size).
                prefetch(tf.data.AUTOTUNE))

        val = (val.
               batch(batch_size=self.conf.batch_size).
               prefetch(tf.data.AUTOTUNE))

        return train, test, val

    def get_transformers(self):
        self.transformer = transformer(self.channels)

    def get_shapes(self, Y):
        self.input_shape = self.transformer.get_shape()

        if self.conf.targets == 'one':
            if len(self.conf.labels) == 1:
                self.output_shape = ()
            if len(self.conf.labels) > 1:
                self.output_shape = Y[1].shape[-1]
        elif self.conf.targets == 'all':
            if len(self.conf.labels) == 1:
                self.output_shape = Y[1].shape[1]
            elif len(self.conf.labels) > 1:
                self.output_shape = Y[1].shape[1:]

        self.output_type = tf.float32
        self.classes = self.conf.labels

    def get_class_weights(self, Y):
        if len(self.classes) > 1:
            for i, label in enumerate(self.classes):
                if self.conf.targets == 'one':
                    pos_perc = np.sum(Y[..., i]) / Y.shape[0]
                elif self.conf.targets == 'all':
                    pos_perc = np.sum(Y[..., i]) / (Y.shape[0] * Y.shape[1])

                pos_weight = 1 / pos_perc
                neg_weight = 1 / (1 - pos_perc)

                self.class_weights[label] = {0: neg_weight, 1: pos_weight}

        elif len(self.classes) == 1:
            if self.conf.targets == 'one':
                pos_perc = np.sum(Y) / Y.shape[0]
            if self.conf.targets == 'all':
                pos_perc = np.sum(Y) / (Y.shape[0] * Y.shape[1])

            pos_weight = 1 / pos_perc
            neg_weight = 1 / (1 - pos_perc)

            self.class_weights = {0: neg_weight, 1: pos_weight}

    def initialize(self):
        data = self.data.copy()
        data = data.drop(data.columns[0], axis=1)
        data = data.rename(columns={'time': 'timestamp', 'subject': 'subject_id', 'activity': 'activity_id'})

        return data

    def preprocess(self, segmenting: bool = True):
        verbose = False

        data = self.initialize()

        data = data[data['subject_id'].isin([107, 1001, 1002, 1003, 1004, 1005, 1007, 1008, 1010, 1012])]

        data = trim(data, length = self.conf.trim_length)
        if verbose:
            plot_one(data)

        data = is_irregular(data, period=self.conf.length, checks=self.conf.checks)

        data = orient(data, self.conf.fs, self.conf.orient_method)
        if verbose:
            plot_one(data)

        data = remove_g(data, self.conf.fs, self.conf.include_g, self.conf.g_cutoff, how='lowpass')
        if verbose:
            plot_one(data)

        data = smooth(data, self.conf.filter, self.conf.fs, self.conf.filter_cutoff)
        if verbose:
            plot_one(data)

        data = produce(data, self.conf.new_features, self.conf.fs)
        if verbose:
            plot_one(data)

        data = get_parameters(data, self.conf.labels, self.conf.task)
        if verbose:
            plot_one(data)

        train, test, val = split_all(data, self.conf.validation, self.conf.split_type, self.conf.test_hold_out, self.conf.val_hold_out, seed)
        if verbose:
            plot_one(train)

        if segmenting:
            train, test, val = self.to_windows(train, test, val)

        return train, test, val

    def to_windows(self, train, test, val):
        test_step = 'same'
        val_step = 'same'
        lowest_step = 1

        train, _, self.channels = finalize(train, self.conf.length,
                                           self.conf.step, self.conf.task,
                                           self.conf.targets, self.conf.target_position)

        which_set = val if self.conf.validation else test
        which_step = (self.conf.step if val_step == 'same' else
                      self.conf.step // 10 if val_step == 'low'
                      else lowest_step)

        val, _, _ = finalize(which_set, self.conf.length, which_step,
                             self.conf.task, self.conf.targets,
                             self.conf.target_position)

        which_step = (self.conf.step if test_step == 'same'
                      else self.conf.step // 10 if test_step == 'low'
                      else lowest_step)

        test, _, _ = finalize(test, self.conf.length,
                              which_step, self.conf.task,
                              self.conf.targets, self.conf.target_position)

        return train, test, val

    def to_y(self, y):
        if self.conf.targets == 'one':
            y = tf.squeeze(y)

        if self.conf.targets == 'all':
            new_shape = (y.shape[0] * y.shape[1]) if len(self.conf.labels) == 1 else \
                (y.shape[0] * y.shape[1], y.shape[2])

            y = tf.squeeze(y).reshape(new_shape)

        return y

    def get_predictions(self, model: keras.Model, data, rotated: bool = False):
        X, Y, T = data
        Y_ = np.zeros(Y.shape)
        if rotated:
            X_rot = np.zeros(X.shape)
        else:
            X_rot = None

        batch = None
        offset = 0

        for i, x in enumerate(X):
            if i % 300 == 0 and i > 0:
                if rotated:
                    X_rot[offset: i] = model.layers[0](batch)

                Y_[offset: i] = tf.squeeze(model.predict(batch, verbose=0))
                offset = i
                batch = None

            x = self.transformer(x)[np.newaxis, ...]

            if batch is None:
                batch = x
            else:
                batch = np.concatenate([batch, x], axis=0)

        if batch is not None:
            if rotated:
                X_rot[offset:] = model.layers[0](batch)

            Y_[offset:] = tf.squeeze(model.predict(batch, verbose=0))

        if self.conf.targets == 'all':
            if len(self.conf.labels) == 1:
                Y_ = Y_.reshape((Y_.shape[0] * Y_.shape[1]))
                Y_ = Y_[:, np.newaxis]
            if len(self.conf.labels) > 1:
                Y_ = Y_.reshape((Y_.shape[0] * Y_.shape[1], Y_.shape[2]))

            t = T[:, 3:]
            t = t.reshape((t.shape[0] * t.shape[1]))

        elif self.conf.targets == 'one':
            if len(self.conf.labels) == 1:
                Y_ = Y_[:, np.newaxis]

            t = T[:, -1]

        prob_labels = [label + '_prob' for label in self.conf.labels]
        pred_labels = [label + '_pred' for label in self.conf.labels]
        y_df = np.concatenate([Y_, t[:, np.newaxis]], axis=1)
        y_df = pd.DataFrame(y_df, columns=[*prob_labels , 'timestamp'])
        y_df[pred_labels] = np.round(y_df[prob_labels].values.astype(np.float32))

        return y_df, X, X_rot

    def compare_yy_(self, model: keras.Model, which: str, subject: int, activity: int,
                    start: int = 0, end: int = -1, rotation: Optional[np.ndarray] = None,
                    rotated: bool = False):

        if which == 'test':
            _, df, _ = self.preprocess(segmenting=False)
        elif which == 'train':
            df, _, _ = self.preprocess(segmenting=False)

        df = df.copy()

        df = df[df['subject_id'] == subject]
        df = df[df['activity_id'] == activity]
        df = df.iloc[start:end]

        if rotation is not None:
            acc_features = df.columns[df.columns.str.contains("acc")]

            a = df[acc_features].values
            a_rot = Rotation.from_matrix(rotation).apply(a)
            df[acc_features] = a_rot

        if self.conf.targets == 'one':
            segs, _, _ = finalize(df, self.conf.length, 1, self.conf.task,
                                  self.conf.targets, self.conf.target_position)

        elif self.conf.targets == 'all':
            segs, _, _ = finalize(df, self.conf.length, self.conf.length, self.conf.task,
                                  self.conf.targets, self.conf.target_position)

        y_df, X, X_rot = self.get_predictions(model, segs, rotated)

        df = pd.merge(df, y_df, on='timestamp', how='left')
        real_labels = {label: label + '_real' for label in self.conf.labels}
        df = df.rename(columns=real_labels)

        return df, X, X_rot






