# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/10 14:59.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
from tensorflow.contrib import slim

from .net_cnn_v1 import epilepsy_3d_cnn, epilepsy_3d_cnn_arg_scope


class Epilepsy3dCnn_V1(object):
    def __init__(self, config):
        self._config = config
        self._input_shape = (None,) + self._config.image_shape
        self._output_shape = (None,)
        self._create_placeholder()
        with slim.arg_scope(epilepsy_3d_cnn_arg_scope(batch_norm_decay=0.99)):
            self._create_test_model()

    def _create_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
        self.labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

    def _create_test_model(self):
        test_logits, test_end_points = epilepsy_3d_cnn(self.inputs,
                                                       num_classes=self._config.num_classes,
                                                       is_training=False,
                                                       dropout_keep_prob=1,
                                                       reuse=None)

        test_predictions = test_end_points['Predictions']
        test_one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=self._config.num_classes,
                                         name='test_one_hot_labels')
        # set loss
        test_loss = tf.losses.softmax_cross_entropy(onehot_labels=test_one_hot_labels, logits=test_logits)
        # get classes
        test_classes = tf.argmax(input=test_predictions, axis=1)
        # get curr accuracy
        test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.labels, tf.int64), test_classes), tf.float32),
                                       name='test_accuracy')
        test_confusion_matrix = tf.confusion_matrix(self.labels, test_classes, num_classes=self._config.num_classes)

        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.test_classes = test_classes
        self.test_logits = test_logits
        self.test_predictions = test_predictions
        self.test_confusion_matrix = test_confusion_matrix
