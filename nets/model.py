# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/10 14:59.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
from tensorflow.contrib import slim

from nets.inception_3d_utils import inception_3d_arg_scope
from nets.inception_3d_v2 import inception_3d_v2
from nets.inception_3d_v3 import inception_3d_v3
from nets.inception_resnet_3d_v2 import inception_resnet_3d_v2_arg_scope, inception_resnet_3d_v2
from nets.rnn import epilepsy_3d_rnn


class Epilepsy3dInceptionResnetV2(object):
  def __init__(self, config):
    self._config = config
    self._input_shape = (None,) + self._config.image_shape
    self._output_shape = (None,)
    self._create_placeholder()
    if self._config.is_training:
      with slim.arg_scope(inception_resnet_3d_v2_arg_scope(batch_norm_decay=self._config.batch_norm_decay)):
        self._create_train_model()
        self._create_test_model()
    else:
      with slim.arg_scope(inception_resnet_3d_v2_arg_scope(batch_norm_decay=self._config.batch_norm_decay)):
        self._create_test_model()

  def _create_placeholder(self):
    with tf.name_scope('placeholder'):
      self._inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
      self._labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

  def _create_train_model(self):
    with tf.name_scope('train_model'):
      train_logits, train_end_points = inception_resnet_3d_v2(self._inputs,
                                                              num_classes=self._config.num_classes,
                                                              is_training=True,
                                                              dropout_keep_prob=self._config.dropout_keep_prob)
      with tf.name_scope('predictions'):
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self._labels, tf.int32),
                                          depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # get classes
        train_classes = tf.argmax(input=train_predictions, axis=1)

      with tf.name_scope('losses'):
        # set loss
        train_final_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        train_aux_logits = train_end_points['AuxLogits']
        train_aux_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_aux_logits)
        train_loss = train_final_loss + train_aux_loss

        # set optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._config.lr)
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
      self._train_accuracy = train_accuracy
      self._train_classes = train_classes
      self._train_logits = train_logits
      self._train_predictions = train_predictions
      self._train_confusion_matrix = train_confusion_matrix

  def _create_test_model(self):
    with tf.name_scope('test_model'):
      test_logits, test_end_points = inception_resnet_3d_v2(self._inputs,
                                                            num_classes=self._config.num_classes,
                                                            is_training=False,
                                                            dropout_keep_prob=1,
                                                            reuse=True)

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


class Epilepsy3dInceptionV3(object):
  def __init__(self, config):
    self._config = config
    self._input_shape = (None,) + self._config.image_shape
    self._output_shape = (None,)
    self._create_placeholder()
    if self._config.is_training:
      with slim.arg_scope(inception_3d_arg_scope(batch_norm_decay=self._config.batch_norm_decay)):
        self._create_train_model()
        self._create_test_model()
    else:
      with slim.arg_scope(inception_3d_arg_scope(batch_norm_decay=self._config.batch_norm_decay)):
        self._create_test_model()

  def _create_placeholder(self):
    with tf.name_scope('placeholder'):
      self._inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
      self._labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

  def _create_train_model(self):
    with tf.name_scope('train_model'):
      train_logits, train_end_points = inception_3d_v3(self._inputs,
                                                       num_classes=self._config.num_classes,
                                                       is_training=True,
                                                       dropout_keep_prob=self._config.dropout_keep_prob)
      with tf.name_scope('predictions'):
        train_predictions = train_end_points['Predictions']
        train_one_hot_labels = tf.one_hot(indices=tf.cast(self._labels, tf.int32),
                                          depth=self._config.num_classes,
                                          name='train_one_hot_labels')
        # get classes
        train_classes = tf.argmax(input=train_predictions, axis=1)

      with tf.name_scope('losses'):
        # set loss
        train_final_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_logits)
        train_aux_logits = train_end_points['AuxLogits']
        train_aux_loss = tf.losses.softmax_cross_entropy(onehot_labels=train_one_hot_labels, logits=train_aux_logits)
        train_loss = train_final_loss + train_aux_loss

        # set optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._config.lr)
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
      self._train_accuracy = train_accuracy
      self._train_classes = train_classes
      self._train_logits = train_logits
      self._train_predictions = train_predictions
      self._train_confusion_matrix = train_confusion_matrix

  def _create_test_model(self):
    with tf.name_scope('test_model'):
      test_logits, test_end_points = inception_3d_v3(self._inputs,
                                                     num_classes=self._config.num_classes,
                                                     is_training=False,
                                                     dropout_keep_prob=1,
                                                     reuse=True)

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


class Epilepsy3dInceptionV2(object):
  def __init__(self, config):
    self._config = config
    self._input_shape = (None,) + self._config.image_shape
    self._output_shape = (None,)
    self._create_placeholder()
    if self._config.is_training:
      with slim.arg_scope(inception_3d_arg_scope(batch_norm_decay=self._config.batch_norm_decay)):
        self._create_train_model()
        self._create_test_model()
    else:
      with slim.arg_scope(inception_3d_arg_scope(batch_norm_decay=self._config.batch_norm_decay)):
        self._create_test_model()

  def _create_placeholder(self):
    with tf.name_scope('placeholder'):
      self._inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
      self._labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

  def _create_train_model(self):
    with tf.name_scope('train_model'):
      train_logits, train_end_points = inception_3d_v2(self._inputs,
                                                       num_classes=self._config.num_classes,
                                                       is_training=True,
                                                       dropout_keep_prob=self._config.dropout_keep_prob)
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
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._config.lr)
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
      self._train_accuracy = train_accuracy
      self._train_classes = train_classes
      self._train_logits = train_logits
      self._train_predictions = train_predictions
      self._train_confusion_matrix = train_confusion_matrix

  def _create_test_model(self):
    with tf.name_scope('test_model'):
      test_logits, test_end_points = inception_3d_v2(self._inputs,
                                                     num_classes=self._config.num_classes,
                                                     is_training=False,
                                                     dropout_keep_prob=1,
                                                     reuse=True)

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
    with tf.name_scope('placeholder'):
      self._inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
      self._labels = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="labels")

  def _create_train_model(self):
    with tf.name_scope('train_model'):
      train_logits, train_end_points = epilepsy_3d_rnn(self._inputs,
                                                       batch_size=self._config.batch_size,
                                                       num_steps=self._config.num_steps,
                                                       num_layers=self._config.num_layers,
                                                       hidden_size=self._config.hidden_size,
                                                       num_classes=self._config.num_classes,
                                                       is_training=True,
                                                       dropout_keep_prob=self._config.dropout_keep_prob)
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
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._config.lr)
        # set train_op
        train_op = slim.learning.create_train_op(train_loss, optimizer)

        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(train_loss, tvars), 10)
        # optimizer = tf.train.GradientDescentOptimizer(self._config.lr)
        # train_op = optimizer.apply_gradients(zip(grads, tvars))

      with tf.name_scope('metrics'):
        # get curr accuracy
        train_accuracy = tf.reduce_mean(
          tf.cast(tf.equal(tf.cast(self._labels, tf.int64), train_classes), tf.float32),
          name='train_accuracy')
        train_confusion_matrix = tf.confusion_matrix(self._labels, train_classes,
                                                     num_classes=self._config.num_classes)

      self._train_loss = train_loss
      self._train_op = train_op
      self._train_accuracy = train_accuracy
      self._train_classes = train_classes
      self._train_logits = train_logits
      self._train_predictions = train_predictions
      self._train_confusion_matrix = train_confusion_matrix

  def _create_test_model(self):
    with tf.name_scope('test_model'):
      test_logits, test_end_points = epilepsy_3d_rnn(self._inputs,
                                                     batch_size=self._config.batch_size,
                                                     num_steps=self._config.num_steps,
                                                     num_layers=self._config.num_layers,
                                                     hidden_size=self._config.hidden_size,
                                                     num_classes=self._config.num_classes,
                                                     is_training=False,
                                                     dropout_keep_prob=1,
                                                     reuse=True)
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
