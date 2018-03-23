# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/3/21 11:00.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf


def lstm_cell(is_training=False, hidden_size=256, input_keep_prob=1, output_keep_prob=0.5):
  cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
  if is_training:
    return tf.nn.rnn_cell.DropoutWrapper(
      cell,
      input_keep_prob=input_keep_prob,
      output_keep_prob=output_keep_prob)
  else:
    return cell
