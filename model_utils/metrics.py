import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config.config_parser import Parser
import sys
from model_utils.losses import get_yy_
from keras.metrics import Metric

epsilon = 1e-7

def get_matches(y_true, y_pred):
    return tf.reduce_sum(
        tf.cast(tf.equal(y_true, y_pred),
                tf.float32)
    )

def get_scores(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(y_true * y_pred,tf.float32))
    fp = tf.reduce_sum(tf.cast(((1-y_true) * y_pred),tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1-y_pred),tf.float32))

    return tp, fp, fn

def binary_accuracies(classes, conf: Parser):
    return [binary_accuracy_class(i, name, conf) for i, name in enumerate(classes)]

class binary_accuracy_class(Metric):
    def __init__(self, class_index, class_name, conf, **kwargs):
        super().__init__(name=f"accuracy_{class_name}", **kwargs)
        self.class_index = class_index
        self.conf = conf

        self.accuracy = self.add_weight(name="acc", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = get_yy_(y_true, y_pred, self.conf)
        y_true = tf.where(y_true > 0.5, 1, 0)
        y_pred = tf.where(y_pred > 0.5, 1, 0)

        if len(self.conf.labels) == 1:
            n_matches = get_matches(y_true, y_pred)
            self.accuracy.assign_add(n_matches)
        elif len(self.conf.labels) > 2:
            n_matches = get_matches(y_true[:, self.class_index], y_pred[:, self.class_index])
            self.accuracy.assign_add(n_matches)

        self.total.assign_add(tf.shape(y_true)[0])

    def result(self):
        return self.accuracy / self.total

    def reset_state(self):
        self.accuracy.assign(0.0)
        self.total.assign(0.0)

def f1_scores(classes, conf: Parser):
    return [F1score_class(i, name, conf) for i, name in enumerate(classes)]

class F1score_class(Metric):
    def __init__(self, class_index, class_name, conf, **kwargs):
        super().__init__(name=f"f1_{class_name}", **kwargs)
        self.class_index = class_index
        self.conf = conf

        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = get_yy_(y_true, y_pred, self.conf)
        y_true = tf.where(y_true > 0.5, 1, 0)
        y_pred = tf.where(y_pred > 0.5, 1, 0)

        if len(self.conf.labels) == 1:
            tp, fp, fn = get_scores(y_true, y_pred)

        elif len(self.conf.labels) > 2:
            tp, fp, fn = get_scores(y_true[:, self.class_index], y_pred[:, self.class_index])

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + epsilon)
        recall = self.true_positives / (self.true_positives + self.false_negatives + epsilon)
        return 2 * precision * recall / (precision + recall + epsilon)

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

class Metrics(Callback):
    def __init__(self, set, steps, file_writer,
                 on='epoch_end', scores=None, average=None, verbose=0):
        super(Metrics, self).__init__()

        self.config = Parser()
        self.config.get_args()

        self.on = on
        self.set = set
        self.batch_size = self.config.batch_size
        self.steps = steps
        self.verbose = verbose
        self.file_writer = file_writer
        self.average = average if average is not None else 'macro'
        self.in_scores = scores if scores is not None else []
        self.scores = {}

        self.label_names = self.config.labels
        self.n_labels = len(self.label_names)
        self.targets = self.config.targets
        self.multi_label = True

    def get_scores(self, true, pred):
        if self.targets == 'all':
            if self.n_labels == 1:
                true = np.reshape(true, (true.shape[0] * true.shape[1]))
                pred = np.reshape(pred, (pred.shape[0] * pred.shape[1]))
            elif self.n_labels > 2:
                true = np.reshape(true, (true.shape[0] * true.shape[1], true.shape[2]))
                pred = np.reshape(pred, (pred.shape[0] * pred.shape[1], pred.shape[2]))

        if 'accuracy' in self.in_scores:
            pred_ =  np.where(pred > 0.5, 1, 0).squeeze()
            true_ = np.where(true > 0.5, 1, 0).squeeze()

            if self.n_labels > 1:
                class_accuracies = [accuracy_score(y, y_) for y, y_ in zip(true_.T, pred_.T)]
                accuracy = np.mean(class_accuracies)
                self.scores['class_accuracy'] = ['{:.4f}'.format(x) for x in class_accuracies]
            elif self.n_labels == 1:
                accuracy = accuracy_score(true_, pred_)

            self.scores['accuracy'] = '{:.4f}'.format(accuracy)

        if 'binary_crossentropy' in self.in_scores:
            binary_crossentropy = BinaryCrossentropy()(true, pred).numpy()

            self.scores['binary_crossentropy'] = '{:.4f}'.format(binary_crossentropy)

        if 'f1_score' in self.in_scores:
            pred_ =  np.where(pred > 0.5, 1, 0).squeeze()

            if self.n_labels > 1:
                class_f1_scores = [f1_score(y, y_, average='binary') for y, y_ in zip(true.T, pred_.T)]
                self.scores['class_f1_score'] = ['{:.4f}'.format(x) for x in class_f1_scores]

            F1_score = f1_score(true, pred_, average=self.average)
            self.scores['f1_score'] = '{:.4f}'.format(F1_score)

        if 'precision' in self.in_scores:
            pred_ = np.where(pred > 0.5, 1, 0).squeeze()

            if self.n_labels > 1:
                class_precisions = [precision_score(y, y_, average='binary') for y, y_ in zip(true.T, pred_.T)]
                self.scores['class_precision'] = ['{:.4f}'.format(x) for x in class_precisions]

            precision = precision_score(true, pred_, average=self.average)
            self.scores['precision'] = '{:.4f}'.format(precision)

        if 'recall' in self.in_scores:
            pred_ = np.where(pred > 0.5, 1, 0).squeeze()

            if self.n_labels > 1:
                class_recalls = [recall_score(y, y_, average='binary') for y, y_ in zip(true.T, pred_.T)]
                self.scores['class_recall'] = ['{:.4f}'.format(x) for x in class_recalls]

            recall = recall_score(true, pred_, average=self.average)
            self.scores['recall'] = '{:.4f}'.format(recall)

    def get_metrics(self):
        total_size = self.batch_size * self.steps
        step = 0

        if self.targets == 'one':
            if self.n_labels > 1:
                pred = np.zeros((total_size, self.n_labels), dtype=np.float32)
                true = np.zeros((total_size, self.n_labels), dtype=np.float32)
            elif self.n_labels == 1:
                pred = np.zeros((total_size,), dtype=np.float32)
                true = np.zeros((total_size,), dtype=np.float32)

        if self.targets == 'all':
            if self.n_labels > 1:
                pred = np.zeros((total_size, self.config.length, self.n_labels), dtype=np.float32)
                true = np.zeros((total_size, self.config.length, self.n_labels), dtype=np.float32)
            elif self.n_labels == 1:
                pred = np.zeros((total_size, self.config.length), dtype=np.float32)
                true = np.zeros((total_size, self.config.length), dtype=np.float32)

        for batch in self.set.take(self.steps):
            X = batch[0]
            y = batch[1]

            if self.multi_label:
                y_ = np.asarray(self.model.predict(X, verbose=self.verbose))
                y_ = np.squeeze(y_)

                pred[step * self.batch_size: (step + 1) * self.batch_size] = y_
                true[step * self.batch_size: (step + 1) * self.batch_size] = y

            step += 1

        self.get_scores(true, pred)

    def on_epoch_end(self, epoch, logs={}):
        if self.on == 'epoch_end':
            self.get_metrics()
            sys.stdout.write("\n")  # Ensure the progress bar finishes cleanly
            print('Validation Metrics: ', self.scores)



    def on_test_end(self, logs={}):
        if self.on == 'test_end':
            self.get_metrics()
            sys.stdout.write("\n")  # Ensure the progress bar finishes cleanly
            print('Test Metrics: ', self.scores)
