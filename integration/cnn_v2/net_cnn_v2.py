# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/10 16:56.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def epilepsy_3d_cnn_base(inputs,
                         final_endpoint='MaxPool_3_3x3x3',
                         scope=None):
    end_points = {}
    with tf.variable_scope(scope, 'Epilepsy_3D_CNN', [inputs]):
        with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                            stride=2,
                            padding='SAME'):
            # 31 x 37 x 31 x 128
            end_point = 'Conv3d_1_3x3x3'
            net = slim.conv3d(inputs,
                              num_outputs=128,
                              kernel_size=[3, 3, 3],
                              weights_initializer=trunc_normal(1.0),
                              scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # 16 x 19 x 16 x 128
            end_point = 'MaxPool_1_3x3x3'
            net = slim.max_pool3d(net,
                                  kernel_size=[3, 3, 3],
                                  scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # 8 x 10 x 8 x 256
            end_point = 'Con3d_2_3x3x3'
            net = slim.conv3d(net,
                              num_outputs=256,
                              kernel_size=[3, 3, 3],
                              weights_initializer=trunc_normal(0.1),
                              scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # 4 x 5 x 4 x 256
            end_point = 'MaxPool_2_3x3x3'
            net = slim.max_pool3d(net,
                                  kernel_size=[2, 2, 2],
                                  scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # 2 x 3 x 2 x 512
            end_point = 'Con3d_3_2x2x2'
            net = slim.conv3d(net,
                              num_outputs=512,
                              kernel_size=[2, 2, 2],
                              weights_initializer=trunc_normal(0.09),
                              scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # 1 x 2 x 1 x 512
            end_point = 'MaxPool_3_3x3x3'
            net = slim.max_pool3d(net,
                                  kernel_size=[2, 2, 2],
                                  scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint {}'.format(final_endpoint))


def epilepsy_3d_cnn(inputs,
                    num_classes=2,
                    is_training=True,
                    dropout_keep_prob=0.8,
                    prediction_fn=slim.softmax,
                    reuse=None,
                    scope='Epilepsy_3D_CNN'):
    with tf.variable_scope(scope, 'Epilepsy_3D_CNN', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.dropout, slim.batch_norm],
                            is_training=is_training):
            net, end_points = epilepsy_3d_cnn_base(inputs, scope=scope)

            with tf.variable_scope('Logits'):
                # 1 x 1024
                net = slim.flatten(net, scope='Flatten')
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1')
                net = slim.fully_connected(net, num_outputs=512, activation_fn=None, scope='Fc_1_512')
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_2')
                logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


epilepsy_3d_cnn.default_image_size = (61, 73, 61)


def epilepsy_3d_cnn_arg_scope(weight_decay=0.00004,
                              use_batch_norm=True,
                              batch_norm_decay=0.9997,
                              batch_norm_epsilon=0.001,
                              activation_fn=tf.nn.relu):
    """Defines the default arg scope for inception models.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
      use_batch_norm: "If `True`, batch_norm is applied after each convolution.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
      activation_fn: Activation function for conv3d.
    Returns:
      An `arg_scope` to use for the inception models.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # use fused batch norm if possible.
        'fused': None,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv3d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv3d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params) as sc:
            return sc
