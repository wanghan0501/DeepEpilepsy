# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/18 14:15.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .cnn import epilepsy_3d_cnn
from .inception_3d_utils import inception_3d_arg_scope


def epilepsy_3d_cnn_cell(inputs,
                         is_training=True,
                         reuse=None,
                         scope='Epilepsy_3D_CNN_Cell'):
  with slim.arg_scope(inception_3d_arg_scope(batch_norm_decay=0.99)):
    net, _ = epilepsy_3d_cnn(inputs, scope=scope,
                             num_classes=None, is_training=is_training, reuse=reuse)
    net = tf.squeeze(net, [1, 2, 3], name='SpatialSqueeze')
  return net


# epilepsy_3d_bi_directional_lstm
def epilepsy_3d_bi_directional_lstm(inputs,
                                    batch_size=4,
                                    num_steps=95,
                                    num_layers=2,
                                    hidden_size=1024,
                                    num_classes=2,
                                    is_training=True,
                                    dropout_keep_prob=0.5,
                                    prediction_fn=slim.softmax,
                                    reuse=None,
                                    scope='Epilepsy_3D_RNN'):
  end_points = {}

  def lstm_cell():
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    if is_training and dropout_keep_prob < 1:
      return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    else:
      return cell

  with tf.variable_scope(scope, 'Epilepsy_3D_RNN', [inputs], reuse=reuse):
    lstm_cells_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
    lstm_cells_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
    initial_state_fw = lstm_cells_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = lstm_cells_bw.zero_state(batch_size, tf.float32)

    with tf.variable_scope('bidirectional_lstm'):
      # Forward direction
      outputs_fw = list()
      state_fw = initial_state_fw
      with tf.variable_scope('fw'):
        for step in range(num_steps):
          if step > 0:
            tf.get_variable_scope().reuse_variables()
          (output_fw, state_fw) = lstm_cells_fw(inputs[:, step, :], state_fw)
          outputs_fw.append(output_fw)

      # backward direction
      outputs_bw = list()
      state_bw = initial_state_bw
      with tf.variable_scope('bw'):
        inputs = tf.reverse(inputs, [1])
        for step in range(num_steps):
          if step > 0:
            tf.get_variable_scope().reuse_variables()
          (output_bw, state_bw) = lstm_cells_bw(inputs[:, step, :], state_bw)
          outputs_bw.append(output_bw)

      outputs_bw = tf.reverse(outputs_bw, [0])
      # concat outputs to [num_steps, batch_size, hidden_size*2]
      outputs = tf.concat([outputs_fw, outputs_bw], 2)
      # transpose outputs from [num_steps, batch_size, hidden_size] to [batch_size, num_steps, hidden_size*2]
      # outputs = tf.transpose(outputs, perm=[1, 0, 2])
    end_points['lstm'] = outputs

    with tf.variable_scope('Logits'):
      finnal_output = outputs[-1]
      logits = slim.fully_connected(finnal_output, num_classes, activation_fn=None, scope='Logits')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


def epilepsy_3d_rnn(inputs,
                    batch_size=4,
                    num_steps=95,
                    num_layers=2,
                    hidden_size=1024,
                    num_classes=2,
                    is_training=True,
                    dropout_keep_prob=0.5,
                    prediction_fn=slim.softmax,
                    reuse=None,
                    scope='Epilepsy_3D_RNN'):
  end_points = {}

  def lstm_cell():
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    if is_training and dropout_keep_prob < 1:
      return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    else:
      return cell

  with tf.variable_scope(scope, 'Epilepsy_3D_RNN', [inputs], reuse=reuse):
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
    initial_state = lstm_cells.zero_state(batch_size, tf.float32)

    with tf.variable_scope('lstm'):
      outputs = list()
      state = initial_state
      for step in range(num_steps):
        if step > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = lstm_cells(inputs[:, step, :], state)
        outputs.append(cell_output)
    end_points['lstm'] = outputs

    # net = tf.convert_to_tensor(outputs, dtype=tf.float32)
    # # change axis from [num_steps, batch_size, hidden_size] to [batch_size, num_steps, hidden_size]
    # net = tf.transpose(net, perm=[1, 0, 2])
    # # expend axis from [batch_size, num_steps, hidden_size] to [batch_size, num_steps, hidden_size, channel]
    # net = tf.expand_dims(net, axis=3)
    # with tf.variable_scope('Logits'):
    #     net = slim.flatten(net, scope='Flatten')
    #     logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
    #     end_points['Logits'] = logits
    #     end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    #
    # return logits, end_points

    with tf.variable_scope('Logits'):
      finnal_state = outputs[-1]
      logits = slim.fully_connected(finnal_state, num_classes, activation_fn=None, scope='Logits')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


epilepsy_3d_rnn.default_image_size = (190, 160)
