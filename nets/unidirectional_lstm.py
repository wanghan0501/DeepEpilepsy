# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/3/21 11:03.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from .lstm_utils import lstm_cell


def unidirectional_lstm_base(inputs,
                             batch_size=4,
                             num_steps=95,
                             num_layers=2,
                             hidden_size=256,
                             is_training=True,
                             dropout_keep_prob=0.5,
                             state_is_tuple=True,
                             scope='UnidirectionalLSTM'):
  end_points = {}

  with tf.variable_scope(scope, 'UnidirectionalLSTM', [inputs]):
    lstm_cells_fw = tf.nn.rnn_cell.MultiRNNCell(
      [lstm_cell(is_training, hidden_size, dropout_keep_prob) for _ in range(num_layers)],
      state_is_tuple=state_is_tuple)
    initial_state_fw = lstm_cells_fw.zero_state(batch_size, tf.float32)

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
  return end_points


def unidirectional_lstm(inputs,
                        batch_size=4,
                        num_steps=95,
                        num_layers=2,
                        hidden_size=256,
                        num_classes=2,
                        is_training=True,
                        dropout_keep_prob=0.5,
                        prediction_fn=slim.softmax,
                        state_is_tuple=True,
                        reuse=None,
                        classify_other_steps=False,
                        scope='UnidirectionalLSTM'):
  with tf.variable_scope(scope, 'UnidirectionalLSTM', [inputs], reuse=reuse):
    end_points = unidirectional_lstm_base(inputs,
                                          batch_size=batch_size,
                                          num_steps=num_steps,
                                          num_layers=num_layers,
                                          hidden_size=hidden_size,
                                          is_training=is_training,
                                          dropout_keep_prob=dropout_keep_prob,
                                          state_is_tuple=state_is_tuple)
    with tf.variable_scope('Logits'):
      outputs = end_points['forward']
      if classify_other_steps:
        steps_logits = list()
        steps_predictions = list()
        for step in outputs:
          logits = slim.fully_connected(step, num_classes, activation_fn=None, scope='Logits', reuse=tf.AUTO_REUSE)
          steps_logits.append(logits)
          predictions = prediction_fn(logits, scope='Predictions')
          steps_predictions.append(predictions)
          end_points['Logits'] = logits
          end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        end_points['StepsLogits'] = steps_logits
        end_points['StepsPredictions'] = steps_predictions
      else:
        final_output = outputs[-1]
        logits = slim.fully_connected(final_output, num_classes, activation_fn=None, scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points
