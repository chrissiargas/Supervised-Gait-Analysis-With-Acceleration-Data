import keras
import pandas as pd
from numpy.ma.core import get_data

from config.config_parser import Parser
import tensorflow as tf
from pre_processing.utils import (remove_g, produce, smooth, get_parameters,
                                  trim, separate, orient, is_irregular)
from pre_processing.splitting import split_all
from pre_processing.segments import finalize
from pre_processing.transformations import transformer
import random
from tqdm import tqdm
from plot_utils.plots import *
from pre_processing.extract import extractor
from scipy.spatial.transform import Rotation
from pre_processing.utils import augment
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

SUBJECTS = [*range(1000, 2000, 2), *range(2000, 2006)]
class builder:
    def __init__(self, generate: bool = False, subjects: Optional[List[int]] = None):
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

        subjects = subjects if subjects else SUBJECTS
        self.load_data(subjects)

    def load_data(self, subjects: Optional[List[int]] = None):
        self.data = None

        subject_dir = sorted(os.listdir(self.path))
        for sub_file in tqdm(subject_dir):
            if subjects is not None:
                if int(sub_file[2:-4]) not in subjects:
                    continue

            sub_path = os.path.join(self.path, sub_file)
            sub_df = pd.read_csv(sub_path)

            if self.data is None:
                self.data = sub_df
            else:
                self.data = pd.concat([self.data, sub_df], axis=0, ignore_index=True)

        self.data = self.data.sort_values(by=['dataset', 'subject'])

    def __call__(self, selected: Optional[List[int]] = None):
        train, test, val = self.preprocess(True, selected)

        if not train:
            self.get_transformers()
            self.get_shapes(test[1])
            self.get_class_weights(test[1])

            return

        if self.randomize:
            train_idx = separate(train)
            train = set_shuffle(train, train_idx)

        self.get_transformers()
        self.get_shapes(train[1])
        self.get_class_weights(train[1])

        self.train_size = train[0].shape[0]
        self.test_size = test[0].shape[0]
        self.val_size = val[0].shape[0]

        train = self.generate(train, training=False)
        test = self.generate(test, training=False)
        val = self.generate(val, training=False)

        return self.batch_prefetch(train, test, val)

    def generate(self, S: Tuple[np.ndarray, np.ndarray, np.ndarray], training: bool = True):
        X, Y, _ = S

        def gen():
            for i, (x, y) in enumerate(zip(X, Y)):
                x = self.transformer(x, training)
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
                self.output_shape = Y.shape[-1]
        elif self.conf.targets == 'all':
            if len(self.conf.labels) == 1:
                self.output_shape = Y.shape[1]
            elif len(self.conf.labels) > 1:
                self.output_shape = Y.shape[1:]

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

    def initialize(self, selected: Optional[List[int]] = None):
        data = self.data.copy()
        data = data.drop(data.columns[0], axis=1)
        data = data.rename(columns={'time': 'timestamp', 'subject': 'subject_id', 'activity': 'activity_id'})

        if selected is not None:
            data = data[data['subject_id'].isin(selected)]

        return data
    
    def prepare(self, selected: Optional[List[int]] = None, verbose: bool = False) -> pd.DataFrame:
        data = self.initialize(selected)

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

        data = get_parameters(data, self.conf.labels, self.conf.task, self.conf.target_oversampling)
        if verbose:
            plot_one(data)

        return data


    def preprocess(self, segmenting: bool = True, selected: Optional[List[int]] = None):
        verbose = False
        augment_before = True

        data = self.prepare(selected, verbose)

        train, test, val = split_all(data, self.conf.validation, self.conf.split_type, self.conf.test_hold_out, self.conf.val_hold_out, seed)
        if verbose:
            plot_one(train)

        if segmenting:
            train, test, val = self.to_windows(train, test, val)

        if augment_before:
            train = augment(train, self.conf.augmentations, self.channels)
            test = augment(test, self.conf.augmentations, self.channels)
            val = augment(val, self.conf.augmentations, self.channels)

        return train, test, val

    def to_windows(self, train, test, val):
        test_step = 'same'
        val_step = 'same'
        lowest_step = 1

        train, _, _ = finalize(train, self.conf.length,
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

        test, _, self.channels = finalize(test, self.conf.length,
                              which_step, self.conf.task,
                              self.conf.targets, self.conf.target_position)

        return train, test, val

    def get_yy_(self, model: keras.Model,
                subjects: List[int],
                activities: Optional[List[int]] = None,
                start: int = 0, end: Optional[int] = None,
                rotation: Optional[np.ndarray] = None,
                oversample: bool = True) -> pd.DataFrame:

        df, windows = self.get_data(subjects, activities, rotation, oversample)
        yy_ = None

        for subject, sub_windows in windows.items():
            sub_yy_ = None

            for activity, act_windows in sub_windows.items():
                act_df = df[subject][activity]

                y = self.get_predictions(act_windows, model, oversample, start, end)
                act_yy_ = pd.merge(act_df, y, on='timestamp', how='left')
                real_labels = {label: label + '_real' for label in self.conf.labels}
                act_yy_ = act_yy_.rename(columns=real_labels)

                if sub_yy_ is None:
                    sub_yy_ = act_yy_
                else:
                    sub_yy_ = pd.concat([sub_yy_, act_yy_], axis=0)

            if yy_ is None:
                yy_ = sub_yy_
            else:
                yy_ = pd.concat([yy_, sub_yy_], axis=0)

        return yy_

    def get_data(self, subjects_: List[int],
                 activities_: Optional[List[int]] = None,
                 rotation: Optional[np.ndarray] = None,
                 oversample: bool = False) -> Tuple[Dict, Dict]:
        self.load_data(subjects_)
        data = self.prepare(subjects_)
        pd.set_option('display.max_columns', None)

        df = {}
        windows = {}

        subjects = data['subject_id'].unique()
        for subject in subjects:
            df[subject] = {}
            windows[subject] = {}

            sub_data = data[data['subject_id'] == subject]
            activities = activities_ if activities_ else sub_data['activity_id'].unique()
            for activity in activities:
                act_data = sub_data[sub_data['activity_id'] == activity]

                if rotation is not None:
                    acc_features = act_data.columns[act_data.columns.str.contains("acc")]
                    a = act_data[acc_features].values
                    a_rot = Rotation.from_matrix(rotation).apply(a)
                    act_data[acc_features] = a_rot

                if self.conf.targets == 'one':
                    x, _, _ = finalize(act_data, self.conf.length, 1, self.conf.task,
                                             self.conf.targets, self.conf.target_position)

                elif self.conf.targets == 'all':
                    step = 1 if oversample else self.conf.step
                    x, _, _ = finalize(act_data, self.conf.length, step, self.conf.task,
                                             self.conf.targets, self.conf.target_position)

                windows[subject][activity] = x
                df[subject][activity] = act_data

        return df, windows

    def get_predictions(self, data, model: keras.Model, oversample: bool = False, start: int = 0, end: int = -1,):
        X, Y, T = data

        # X = X[start:end].copy()
        # Y = Y[start:end].copy()
        # T = T[start:end].copy()

        Y_ = np.zeros(Y.shape)

        batch = None
        offset = 0

        for i, x in enumerate(X):
            if i % 300 == 0 and i > 0:
                # if rotated:
                #     rotated_X[offset: i] = model.layers[0](batch)

                Y_[offset: i] = tf.squeeze(model.predict(batch, verbose=0))
                offset = i
                batch = None

            x = self.transformer(x)[np.newaxis, ...]

            if batch is None:
                batch = x
            else:
                batch = np.concatenate([batch, x], axis=0)

        if batch is not None:
            # if rotated:
            #     rotated_X[offset:] = model.layers[0](batch)

            Y_[offset:] = tf.squeeze(model.predict(batch, verbose=0))

        if self.conf.targets == 'all':
            if oversample:
                if len(self.conf.labels) == 1:
                    Y_ = Y_[:, self.conf.length // 2]
                    Y_ = Y_[:, np.newaxis]
                elif len(self.conf.labels) > 1:
                    Y_ = Y_[:, self.conf.length // 2, :]

            else:
                if len(self.conf.labels) == 1:
                    Y_ = Y_.reshape((Y_.shape[0] * Y_.shape[1]))
                    Y_ = Y_[:, np.newaxis]
                if len(self.conf.labels) > 1:
                    Y_ = Y_.reshape((Y_.shape[0] * Y_.shape[1], Y_.shape[2]))

            t = T[:, 3:]

            if oversample:
                t = t[:, self.conf.length // 2]
            else:
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

        return y_df







