# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/10 14:59.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
from tensorflow.contrib import slim

from .bidirectional_lstm_with_inception import bidirectional_lstm
from .unidirectional_lstm_with_inception import unidirectional_lstm


class EpilepsyUnidirectionalLSTM(object):
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
    with tf.name_scope('placeholder'):
      self._inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
      self._coefficients = tf.placeholder(dtype=tf.float32, shape=(self._config.batch_size, 160, 160, 1),
                                          name="coefficients")
      self._labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

  def _create_train_model(self):
    with tf.name_scope('train_model'):
      train_logits, train_end_points = unidirectional_lstm(
        self._inputs,
        self._coefficients,
        batch_size=self._config.batch_size,
        num_steps=self._config.num_steps,
        num_layers=self._config.num_layers,
        hidden_size=self._config.hidden_size,
        num_classes=self._config.num_classes,
        is_training=True,
        input_keep_prob=self._config.input_keep_prob,
        output_keep_prob=self._config.output_keep_prob)
      with tf.name_scope('predictions'):
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self._labels, tf.int32),
                                          depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # get classes
        train_classes = tf.argmax(input=train_predictions, axis=1)
      with tf.name_scope('losses'):
        # set loss
        train_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        # set optimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self._config.lr,
                                                   global_step=global_step,
                                                   decay_steps=500, decay_rate=0.96)
        learning_rate = tf.maximum(learning_rate, 1e-6)
        optimizer = self._config.optimizer(learning_rate=learning_rate)
        # set train_op
        train_op = slim.learning.create_train_op(train_loss, optimizer)
      with tf.name_scope('metrics'):
        # get curr accuracy
        train_accuracy = tf.reduce_mean(
          tf.cast(tf.equal(tf.cast(self._labels, tf.int64), train_classes), tf.float32),
          name='train_accuracy')
        train_confusion_matrix = tf.confusion_matrix(self._labels, train_classes,
                                                     num_classes=self._config.num_classes)
      self._train_loss = train_loss
      self._train_op = train_op
      self._learning_rate = learning_rate
      self._train_accuracy = train_accuracy
      self._train_classes = train_classes
      self._train_logits = train_logits
      self._train_predictions = train_predictions
      self._train_confusion_matrix = train_confusion_matrix

  def _create_test_model(self):
    with tf.name_scope('test_model'):
      test_logits, test_end_points = unidirectional_lstm(
        self._inputs,
        self._coefficients,
        batch_size=self._config.batch_size,
        num_steps=self._config.num_steps,
        num_layers=self._config.num_layers,
        hidden_size=self._config.hidden_size,
        num_classes=self._config.num_classes,
        is_training=False,
        input_keep_prob=1,
        output_keep_prob=1,
        reuse=tf.AUTO_REUSE)
      with tf.name_scope('predictions'):
        test_predictions = test_end_points['Predictions']
        test_one_hot_labels = tf.one_hot(indices=tf.cast(self._labels, tf.int32),
                                         depth=self._config.num_classes,
                                         name='test_one_hot_labels')
        # get classes
        test_classes = tf.argmax(input=test_predictions, axis=1)
      with tf.name_scope('losses'):
        # set loss
        test_loss = tf.losses.softmax_cross_entropy(onehot_labels=test_one_hot_labels, logits=test_logits)
      with tf.name_scope('metrics'):
        # get curr accuracy
        test_accuracy = tf.reduce_mean(
          tf.cast(tf.equal(tf.cast(self._labels, tf.int64), test_classes), tf.float32),
          name='test_accuracy')
        test_confusion_matrix = tf.confusion_matrix(self._labels, test_classes,
                                                    num_classes=self._config.num_classes)

      self._test_loss = test_loss
      self._test_accuracy = test_accuracy
      self._test_classes = test_classes
      self._test_logits = test_logits
      self._test_predictions = test_predictions
      self._test_confusion_matrix = test_confusion_matrix

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def train_loss(self):
    return self._train_loss

  @property
  def learning_rate(self):
    return self._learning_rate

  @property
  def train_op(self):
    return self._train_op

  @property
  def train_accuracy(self):
    return self._train_accuracy

  @property
  def train_classes(self):
    return self._train_classes

  @property
  def train_logits(self):
    return self._train_logits

  @property
  def train_predictions(self):
    return self._train_predictions

  @property
  def train_confusion_matrix(self):
    return self._train_confusion_matrix

  @property
  def test_loss(self):
    return self._test_loss

  @property
  def test_accuracy(self):
    return self._test_accuracy

  @property
  def test_classes(self):
    return self._test_classes

  @property
  def test_logits(self):
    return self._test_logits

  @property
  def test_predictions(self):
    return self._test_predictions

  @property
  def test_confusion_matrix(self):
    return self._test_confusion_matrix


class EpilepsyBidirectionalLSTM(object):
  def __init__(self, config):
    self._config = config
    self._input_shape = (config.batch_size,) + self._config.image_shape
    self._output_shape = (config.batch_size,)
    self._create_placeholder()
    if self._config.is_training:
      self._create_train_model()
      self._create_test_model()
    else:
      self._create_test_model()

  def _create_placeholder(self):
    with tf.name_scope('placeholder'):
      self._inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
      self._coefficients = tf.placeholder(dtype=tf.float32, shape=(self._config.batch_size, 160, 160, 1),
                                          name="coefficients")
      self._labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

  def _create_train_model(self):
    with tf.name_scope('train_model'):
      train_logits, train_end_points = bidirectional_lstm(
        self._inputs,
        self._coefficients,
        batch_size=self._config.batch_size,
        num_steps=self._config.num_steps,
        num_layers=self._config.num_layers,
        hidden_size=self._config.hidden_size,
        num_classes=self._config.num_classes,
        is_training=True,
        input_keep_prob=self._config.input_keep_prob,
        output_keep_prob=self._config.output_keep_prob)
      with tf.name_scope('predictions'):
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self._labels, tf.int32),
                                          depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # get classes
        train_classes = tf.argmax(input=train_predictions, axis=1)
      with tf.name_scope('losses'):
        # set loss
        train_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        # set optimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self._config.lr,
                                                   global_step=global_step,
                                                   decay_steps=500, decay_rate=0.96)
        learning_rate = tf.maximum(learning_rate, 1e-6)
        optimizer = self._config.optimizer(learning_rate=learning_rate)
        # set train_op
        train_op = slim.learning.create_train_op(train_loss, optimizer)
      with tf.name_scope('metrics'):
        # get curr accuracy
        train_accuracy = tf.reduce_mean(
          tf.cast(tf.equal(tf.cast(self._labels, tf.int64), train_classes), tf.float32),
          name='train_accuracy')
        train_confusion_matrix = tf.confusion_matrix(self._labels, train_classes,
                                                     num_classes=self._config.num_classes)
      self._learning_rate = learning_rate
      self._train_loss = train_loss
      self._train_op = train_op
      self._train_accuracy = train_accuracy
      self._train_classes = train_classes
      self._train_logits = train_logits
      self._train_predictions = train_predictions
      self._train_confusion_matrix = train_confusion_matrix

  def _create_test_model(self):
    with tf.name_scope('test_model'):
      test_logits, test_end_points = bidirectional_lstm(
        self._inputs,
        self._coefficients,
        batch_size=self._config.batch_size,
        num_steps=self._config.num_steps,
        num_layers=self._config.num_layers,
        hidden_size=self._config.hidden_size,
        num_classes=self._config.num_classes,
        is_training=False,
        input_keep_prob=1,
        output_keep_prob=1,
        reuse=tf.AUTO_REUSE)
      with tf.name_scope('predictions'):
        test_predictions = test_end_points['Predictions']
        test_one_hot_labels = tf.one_hot(indices=tf.cast(self._labels, tf.int32),
                                         depth=self._config.num_classes,
                                         name='test_one_hot_labels')
        # get classes
        test_classes = tf.argmax(input=test_predictions, axis=1)
      with tf.name_scope('losses'):
        # set loss
        test_loss = tf.losses.softmax_cross_entropy(onehot_labels=test_one_hot_labels, logits=test_logits)
      with tf.name_scope('metrics'):
        # get curr accuracy
        test_accuracy = tf.reduce_mean(
          tf.cast(tf.equal(tf.cast(self._labels, tf.int64), test_classes), tf.float32),
          name='test_accuracy')
        test_confusion_matrix = tf.confusion_matrix(self._labels, test_classes,
                                                    num_classes=self._config.num_classes)

      self._test_loss = test_loss
      self._test_accuracy = test_accuracy
      self._test_classes = test_classes
      self._test_logits = test_logits
      self._test_predictions = test_predictions
      self._test_confusion_matrix = test_confusion_matrix

  @property
  def inputs(self):
    return self._inputs

  @property
  def coefficients(self):
    return self._coefficients

  @property
  def labels(self):
    return self._labels

  @property
  def learning_rate(self):
    return self._learning_rate

  @property
  def train_loss(self):
    return self._train_loss

  @property
  def train_op(self):
    return self._train_op

  @property
  def train_accuracy(self):
    return self._train_accuracy

  @property
  def train_classes(self):
    return self._train_classes

  @property
  def train_logits(self):
    return self._train_logits

  @property
  def train_predictions(self):
    return self._train_predictions

  @property
  def train_confusion_matrix(self):
    return self._train_confusion_matrix

  @property
  def test_loss(self):
    return self._test_loss

  @property
  def test_accuracy(self):
    return self._test_accuracy

  @property
  def test_classes(self):
    return self._test_classes

  @property
  def test_logits(self):
    return self._test_logits

  @property
  def test_predictions(self):
    return self._test_predictions

  @property
  def test_confusion_matrix(self):
    return self._test_confusion_matrix
