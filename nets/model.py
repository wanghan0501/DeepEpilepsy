# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/10 14:59.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf

from nets.cnn import epilepsy_3d_cnn
from nets.rnn import epilepsy_3d_rnn

slim = tf.contrib.slim


class Epilepsy3dCnn(object):
    def __init__(self, config):
        self._config = config
        self._input_shape = (None,) + self._config.image_shape
        self._output_shape = (None,)

        self._create_placeholder()
        if self._config.is_training:
            self._create_train_model()
            self._create_test_model()
        else:
            self._create_test_model()

    def _create_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
        self.labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

    def _create_train_model(self):
        train_logits, train_end_points = epilepsy_3d_cnn(self.inputs,
                                                         num_classes=self._config.num_classes,
                                                         is_training=True,
                                                         dropout_keep_prob=self._config.dropout_keep_prob)
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # set loss
        train_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        # set optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
        # set train_op
        train_op = slim.learning.create_train_op(train_loss, optimizer)
        # get classes
        train_classes = tf.argmax(input=train_predictions, axis=1)
        # get curr accuracy
        train_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(self.labels, tf.int64), train_classes), tf.float32), name='train_accuracy')
        train_confusion_matrix = tf.confusion_matrix(self.labels, train_classes, num_classes=self._config.num_classes)

        self.train_loss = train_loss
        self.train_op = train_op
        self.train_accuracy = train_accuracy
        self.train_classes = train_classes
        self.train_logits = train_logits
        self.train_predictions = train_predictions
        self.train_confusion_matrix = train_confusion_matrix

    def _create_test_model(self):
        test_logits, test_end_points = epilepsy_3d_cnn(self.inputs,
                                                       num_classes=self._config.num_classes,
                                                       is_training=False,
                                                       dropout_keep_prob=1,
                                                       reuse=True)

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


class Epilepsy3dRnn(object):
    def __init__(self, config):
        self._config = config
        self._input_shape = (None,) + self._config.image_shape
        self._output_shape = (None,)

        self._create_placeholder()
        if self._config.is_training:
            self._create_train_model()
            self._create_test_model()
        else:
            self._create_test_model()

    def _create_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
        self.labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

    def _create_train_model(self):
        train_logits, train_end_points = epilepsy_3d_rnn(self.inputs,
                                                         num_steps=self._config.num_steps,
                                                         num_layers=self._config.num_layers,
                                                         hidden_size=self._config.hidden_size,
                                                         num_classes=self._config.num_classes,
                                                         is_training=True,
                                                         dropout_keep_prob=self._config.dropout_keep_prob)
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # set loss
        train_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        # set optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
        # set train_op
        train_op = slim.learning.create_train_op(train_loss, optimizer)
        # get classes
        train_classes = tf.argmax(input=train_predictions, axis=1)
        # get curr accuracy
        train_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(self.labels, tf.int64), train_classes), tf.float32), name='train_accuracy')
        train_confusion_matrix = tf.confusion_matrix(self.labels, train_classes, num_classes=self._config.num_classes)

        self.train_loss = train_loss
        self.train_op = train_op
        self.train_accuracy = train_accuracy
        self.train_classes = train_classes
        self.train_logits = train_logits
        self.train_predictions = train_predictions
        self.train_confusion_matrix = train_confusion_matrix

    def _create_test_model(self):
        test_logits, test_end_points = epilepsy_3d_rnn(self.inputs,
                                                       num_steps=self._config.num_steps,
                                                       num_layers=self._config.num_layers,
                                                       hidden_size=self._config.hidden_size,
                                                       num_classes=self._config.num_classes,
                                                       is_training=False,
                                                       dropout_keep_prob=1,
                                                       reuse=True)

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
