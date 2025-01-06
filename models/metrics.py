import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.losses import BinaryCrossentropy
from keras.src.layers import average
from keras.src.ops import binary_crossentropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config_parser import Parser
from keras.metrics import binary_accuracy
import sys
import time

def binary_accuracies(data):
    return [binary_accuracy_class(i, name) for i, name in zip(range(data.output_shape), data.classes)]

def binary_accuracy_class(class_index, class_name):
    def metric(y_true, y_pred):
        return binary_accuracy(y_true[:, class_index], y_pred[:, class_index])
    metric.__name__ = f"accuracy_{class_name}"
    return metric

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

        if self.config.task == 'gait_phases':
            self.label_names = self.config.parameters
            self.n_labels = len(self.label_names)
            self.multi_label = True

    def get_scores(self, true, pred):

        if 'accuracy' in self.in_scores:
            pred_ =  np.where(pred > 0.5, 1, 0).squeeze()
            class_accuracies = [accuracy_score(y, y_) for y, y_ in zip(true.T, pred_.T)]
            accuracy = np.mean(class_accuracies)

            self.scores['class_accuracy'] = ['{:.4f}'.format(x) for x in class_accuracies]
            self.scores['accuracy'] = '{:.4f}'.format(accuracy)

        if 'binary_crossentropy' in self.in_scores:
            binary_crossentropy = BinaryCrossentropy()(true, pred).numpy()

            self.scores['binary_crossentropy'] = '{:.4f}'.format(binary_crossentropy)

        if 'f1_score' in self.in_scores:
            pred_ =  np.where(pred > 0.5, 1, 0).squeeze()
            class_f1_scores = [f1_score(y, y_, average='binary') for y, y_ in zip(true.T, pred_.T)]
            F1_score = f1_score(true, pred_, average=self.average)

            self.scores['class_f1_score'] = ['{:.4f}'.format(x) for x in class_f1_scores]
            self.scores['f1_score'] = '{:.4f}'.format(F1_score)

        if 'precision' in self.in_scores:
            pred_ = np.where(pred > 0.5, 1, 0).squeeze()
            class_precisions = [precision_score(y, y_, average='binary') for y, y_ in zip(true.T, pred_.T)]
            precision = precision_score(true, pred_, average=self.average)

            self.scores['class_precision'] = ['{:.4f}'.format(x) for x in class_precisions]
            self.scores['precision'] = '{:.4f}'.format(precision)

        if 'recall' in self.in_scores:
            pred_ = np.where(pred > 0.5, 1, 0).squeeze()
            class_recalls = [recall_score(y, y_, average='binary') for y, y_ in zip(true.T, pred_.T)]
            recall = recall_score(true, pred_, average=self.average)

            self.scores['class_recall'] = ['{:.4f}'.format(x) for x in class_recalls]
            self.scores['recall'] = '{:.4f}'.format(recall)



    def get_metrics(self):
        total_size = self.batch_size * self.steps
        step = 0

        if self.multi_label:
            pred = np.zeros((total_size, self.n_labels), dtype=np.float32)
            true = np.zeros((total_size, self.n_labels), dtype=np.float32)

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
