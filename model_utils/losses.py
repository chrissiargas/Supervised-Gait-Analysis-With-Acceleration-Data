import tensorflow as tf
from config_utils.config_parser import Parser

epsilon = 1e-7

def get_yy_(y_true, y_pred, conf):
    if conf.targets == 'all':
        if len(conf.labels) == 1:
            y_true = tf.reshape(y_true, (-1,))
            y_pred = tf.reshape(y_pred, (-1,))
        elif len(conf.labels) > 2:
            y_true = tf.reshape(y_true, (-1, y_true.shape[2]))
            y_pred = tf.reshape(y_pred, (-1, y_pred.shape[2]))

    return y_true, y_pred

def get_weighted_BCE(class_weights, conf: Parser):
    def weighted_bce(y_true, y_pred):
        y_true, y_pred = get_yy_(y_true, y_pred, conf)

        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        loss_positive = -tf.math.log(y_pred) * y_true * class_weights[1]
        loss_negative = -tf.math.log(1 - y_pred) * (1 - y_true) * class_weights[0]
        total_loss = tf.reduce_mean(loss_positive + loss_negative)

        return total_loss

    return weighted_bce

def get_BCE(conf: Parser):
    def bce(y_true, y_pred):
        y_true, y_pred = get_yy_(y_true, y_pred, conf)

        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        loss_positive = -tf.math.log(y_pred) * y_true
        loss_negative = -tf.math.log(1 - y_pred) * (1 - y_true)
        total_loss = tf.reduce_mean(loss_positive + loss_negative)

        return total_loss

    return bce


