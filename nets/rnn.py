# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/18 14:15.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .cnn import epilepsy_3d_cnn, epilepsy_3d_cnn_arg_scope


def epilepsy_3d_cnn_cell(inputs,
                         is_training=True,
                         reuse=None,
                         scope='Epilepsy_3D_CNN_Cell'):
    with slim.arg_scope(epilepsy_3d_cnn_arg_scope(batch_norm_decay=0.99)):
        net, _ = epilepsy_3d_cnn(inputs, scope=scope,
                                 num_classes=None, is_training=is_training, reuse=reuse)
        net = tf.squeeze(net, [1, 2, 3], name='SpatialSqueeze')
    return net


def epilepsy_3d_rnn(inputs,
                    num_steps=190,
                    num_layers=2,
                    hidden_size=1024,
                    num_classes=2,
                    is_training=True,
                    dropout_keep_prob=0.8,
                    prediction_fn=slim.softmax,
                    reuse=None,
                    scope='Epilepsy_3D_RNN'):
    end_points = {}

    with tf.variable_scope(scope, 'Epilepsy_3D_RNN', [inputs], reuse=reuse):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0, state_is_tuple=True)
        if is_training and dropout_keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

        # get batch size
        batch_size = tf.shape(inputs)[0]
        initial_state = cell.zero_state(batch_size, tf.float32)

        outputs = list()
        state = initial_state
        with tf.variable_scope('LSTM'):
            for step in range(num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                # get cnn cell
                cur_input = tf.reshape(inputs[:, :, :, :, step], (-1, 61, 73, 61))
                if step > 0:
                    cnn_cell = epilepsy_3d_cnn_cell(cur_input, is_training=is_training, reuse=True, scope='CNN_Cell')
                else:
                    cnn_cell = epilepsy_3d_cnn_cell(cur_input, is_training=is_training, scope='CNN_Cell')
                # restore the state of LSTM cell
                cell_output, state = cell(cnn_cell, state)
                outputs.append(cell_output)
        final_state = outputs[-1]

        with tf.variable_scope('Logits'):
            logits = slim.fully_connected(final_state, num_classes, activation_fn=None, scope='Logits')
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


epilepsy_3d_rnn.default_image_size = (61, 73, 61, 190)
