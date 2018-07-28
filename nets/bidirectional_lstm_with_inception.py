# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/3/21 11:03.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.inception_utils import inception_arg_scope
from .coefficient_net import coefficient_net
from .lstm_utils import lstm_cell


def bidirectional_lstm_base(inputs,
                            batch_size=4,
                            num_steps=95,
                            num_layers=2,
                            hidden_size=256,
                            is_training=True,
                            input_keep_prob=1,
                            output_keep_prob=0.5,
                            state_is_tuple=True,
                            scope='BidirectionalLSTM'):
  end_points = {}

  with tf.variable_scope(scope, 'BidirectionalLSTM', [inputs]):
    lstm_cells_fw = tf.nn.rnn_cell.MultiRNNCell(
      [lstm_cell(is_training, hidden_size, input_keep_prob, output_keep_prob) for _ in range(num_layers)],
      state_is_tuple=state_is_tuple)
    lstm_cells_bw = tf.nn.rnn_cell.MultiRNNCell(
      [lstm_cell(is_training, hidden_size, input_keep_prob, output_keep_prob) for _ in range(num_layers)],
      state_is_tuple=state_is_tuple)
    initial_state_fw = lstm_cells_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = lstm_cells_bw.zero_state(batch_size, tf.float32)

    with tf.variable_scope('Layers'):
      # Forward direction
      outputs_fw = list()
      state_fw = initial_state_fw
      with tf.variable_scope('Forward'):
        for step in range(num_steps):
          if step > 0:
            tf.get_variable_scope().reuse_variables()
          (output_fw, state_fw) = lstm_cells_fw(inputs[:, step, :], state_fw)
          outputs_fw.append(output_fw)
      end_points['forward'] = outputs_fw
      # backward direction
      outputs_bw = list()
      state_bw = initial_state_bw
      with tf.variable_scope('Backward'):
        inputs = tf.reverse(inputs, [1])
        for step in range(num_steps):
          if step > 0:
            tf.get_variable_scope().reuse_variables()
          (output_bw, state_bw) = lstm_cells_bw(inputs[:, step, :], state_bw)
          outputs_bw.append(output_bw)
      outputs_bw = tf.reverse(outputs_bw, [0])
      end_points['backward'] = outputs_bw

    return end_points


def bidirectional_lstm(inputs,
                       coefficients,
                       batch_size=4,
                       num_steps=95,
                       num_layers=2,
                       hidden_size=256,
                       num_classes=2,
                       is_training=True,
                       input_keep_prob=1,
                       output_keep_prob=0.5,
                       prediction_fn=slim.softmax,
                       classify_other_steps=False,
                       state_is_tuple=True,
                       reuse=None,
                       scope='BidirectionalLSTM'):
  with tf.variable_scope(scope, 'BidirectionalLSTM', [inputs], reuse=reuse):
    end_points = bidirectional_lstm_base(inputs,
                                         batch_size=batch_size,
                                         num_steps=num_steps,
                                         num_layers=num_layers,
                                         hidden_size=hidden_size,
                                         is_training=is_training,
                                         input_keep_prob=input_keep_prob,
                                         output_keep_prob=output_keep_prob,
                                         state_is_tuple=state_is_tuple)
    with tf.variable_scope('Logits'):
      forward = end_points['forward']
      backward = end_points['backward']
      # concat outputs to [num_steps, batch_size, hidden_size*2]
      outputs = tf.concat([forward, backward], 2)
      # transpose outputs from [num_steps, batch_size, hidden_size] to [batch_size, num_steps, hidden_size*2]
      # outputs = tf.transpose(outputs, perm=[1, 0, 2])
      end_points['output'] = outputs
      outputs = end_points['output']

      if classify_other_steps:
        steps_logits = list()
        steps_predictions = list()
        with slim.arg_scope(inception_arg_scope(batch_norm_decay=0.99)):
          coefficient, _ = coefficient_net(coefficients, keep_prob=output_keep_prob, is_training=is_training)
        for step in range(num_steps):
          all = tf.concat([outputs[step], coefficient], 1)
          logits = slim.fully_connected(all, num_classes, activation_fn=None, scope='Logits',
                                        reuse=tf.AUTO_REUSE)
          steps_logits.append(logits)
          predictions = prediction_fn(logits, scope='Predictions')
          steps_predictions.append(predictions)
          end_points['Logits'] = logits
          end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        end_points['StepsLogits'] = steps_logits
        end_points['StepsPredictions'] = steps_predictions
      else:
        # [batch_size, hidden_size * 2]
        final_output = outputs[-1]
        # [batch_size, 1024]
        with slim.arg_scope(inception_arg_scope(batch_norm_decay=0.99)):
          coefficient, _ = coefficient_net(coefficients, keep_prob=output_keep_prob, is_training=is_training)
        all = tf.concat([final_output, coefficient], 1)
        logits = slim.fully_connected(all, num_classes, activation_fn=None, scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points
