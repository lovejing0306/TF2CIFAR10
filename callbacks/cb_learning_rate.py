# coding=utf-8

from tensorflow import keras
import tensorflow as tf


RANGE = 'range'
EXPONENT = 'exponent'


class LearningRate(object):
    def __init__(self, optimizer=None, method=None, lr_range=None, initial_lr=None):
        self.optimizer = optimizer
        self.method = method
        self.lr_range = lr_range
        self.initial_lr = initial_lr

    def __call__(self, epoch, logs=None):
        if self.optimizer is None:
            raise ValueError('optimizer is none.')
        if not hasattr(self.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')

        # Get the current learning rate from model's optimizer.
        lr = float(keras.backend.get_value(self.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        if self.method == 'range':
            scheduled_lr = self.adjust_range(epoch, lr)
        elif self.method == 'exponent':
            scheduled_lr = self.adjust_exponent(epoch)
        else:
            scheduled_lr = lr

        # Set the value back to the optimizer before this epoch starts
        keras.backend.set_value(self.optimizer.learning_rate, scheduled_lr)

    def adjust_range(self, epoch, lr):
        if self.lr_range is None:
            raise ValueError('lr_ranges is none.')
        if epoch < self.lr_range[0][0] or epoch > self.lr_range[-1][0]:
            return lr
        for i in range(len(self.lr_range)-1, -1, -1):
            if epoch >= self.lr_range[i][0]:
                return self.lr_range[i][1]
        return lr

    def adjust_exponent(self, epoch):
        if self.initial_lr is None:
            raise ValueError('initial_lr is none.')
        if epoch < 10:
            return self.initial_lr
        else:
            return self.initial_lr * tf.math.exp(0.01 * (10 - epoch))
