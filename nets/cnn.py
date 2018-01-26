# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/10 16:56.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# def epilepsy_3d_cnn_base(inputs,
#                          final_endpoint='MaxPool_3_3x3x3',
#                          scope=None):
#     end_points = {}
#     with tf.variable_scope(scope, 'Epilepsy_3D_CNN', [inputs]):
#         with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
#                             stride=2,
#                             padding='SAME'):
#             # 31 x 37 x 31 x 128
#             end_point = 'Conv3d_1_3x3x3'
#             net = slim.conv3d(inputs,
#                               num_outputs=128,
#                               kernel_size=[3, 3, 3],
#                               weights_initializer=trunc_normal(1.0),
#                               scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_endpoint: return net, end_points
#
#             # 16 x 19 x 16 x 128
#             end_point = 'MaxPool_1_3x3x3'
#             net = slim.max_pool3d(net,
#                                   kernel_size=[3, 3, 3],
#                                   scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_endpoint: return net, end_points
#
#             # 8 x 10 x 8 x 256
#             end_point = 'Con3d_2_3x3x3'
#             net = slim.conv3d(net,
#                               num_outputs=256,
#                               kernel_size=[3, 3, 3],
#                               weights_initializer=trunc_normal(0.1),
#                               scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_endpoint: return net, end_points
#
#             # 4 x 5 x 4 x 256
#             end_point = 'MaxPool_2_3x3x3'
#             net = slim.max_pool3d(net,
#                                   kernel_size=[2, 2, 2],
#                                   scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_endpoint: return net, end_points
#
#             # 2 x 3 x 2 x 512
#             end_point = 'Con3d_3_2x2x2'
#             net = slim.conv3d(net,
#                               num_outputs=512,
#                               kernel_size=[2, 2, 2],
#                               weights_initializer=trunc_normal(0.09),
#                               scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_endpoint: return net, end_points
#
#             # 1 x 2 x 1 x 512
#             end_point = 'MaxPool_3_3x3x3'
#             net = slim.max_pool3d(net,
#                                   kernel_size=[2, 2, 2],
#                                   scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_endpoint: return net, end_points
#     raise ValueError('Unknown final endpoint {}'.format(final_endpoint))
#
#
# def epilepsy_3d_cnn(inputs,
#                     num_classes=2,
#                     is_training=True,
#                     dropout_keep_prob=0.8,
#                     prediction_fn=slim.softmax,
#                     reuse=None,
#                     scope='Epilepsy_3D_CNN'):
#     with tf.variable_scope(scope, 'Epilepsy_3D_CNN', [inputs], reuse=reuse) as scope:
#         with slim.arg_scope([slim.dropout, slim.batch_norm],
#                             is_training=is_training):
#             net, end_points = epilepsy_3d_cnn_base(inputs, scope=scope)
#
#             with tf.variable_scope('Logits'):
#                 # 1 x 1024
#                 net = slim.flatten(net, scope='Flatten')
#                 net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1')
#                 net = slim.fully_connected(net, num_outputs=512, activation_fn=None, scope='Fc_1_512')
#                 net = slim.dropout(net, dropout_keep_prob, scope='Dropout_2')
#                 logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
#                 end_points['Logits'] = logits
#                 end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
#
#     return logits, end_points



def epilepsy_3d_cnn_base(inputs,
                         final_endpoint='Mixed_5c',
                         min_depth=16,
                         depth_multiplier=1.0,
                         scope=None):
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    concat_dim = 4
    with tf.variable_scope(scope, 'Epilepsy_3D_CNN', [inputs]):
        with slim.arg_scope(
                [slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                stride=1,
                padding='SAME',
        ):

            # 61 x 73 x 61 x 2
            end_point = 'Conv2d_1a_7x7x7'

            net = slim.conv3d(
                inputs,
                depth(64), [7, 7, 7],
                stride=2,
                weights_initializer=trunc_normal(1.0),
                scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # 31 x 37 x 31 x 64
            end_point = 'MaxPool_2a_3x3x3'
            net = slim.max_pool3d(net, [3, 3, 3], scope=end_point, stride=2)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 16 x 19 x 16 x 64
            end_point = 'Conv2d_2b_1x1x1'
            net = slim.conv3d(net, depth(64), [1, 1, 1], scope=end_point,
                              weights_initializer=trunc_normal(0.1))
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 16 x 19 x 16 x 64
            end_point = 'Conv2d_2c_3x3x3'
            net = slim.conv3d(net, depth(192), [3, 3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 16 x 19 x 16 x 192
            end_point = 'MaxPool_3a_3x3x3'
            net = slim.max_pool3d(net, [3, 3, 3], scope=end_point, stride=2)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 8 x 10 x 8 x 192
            # Inception module.
            end_point = 'Mixed_3b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(64), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(64), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(64), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(64), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(32), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 8 x 10 x 8 x 256
            end_point = 'Mixed_3c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(64), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(64), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(96), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(64), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(64), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 8 x 10 x 8 x 320
            end_point = 'Mixed_4a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(
                        net, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_0 = slim.conv3d(branch_0, depth(160), [3, 3, 3], stride=2,
                                           scope='Conv2d_1a_3x3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(64), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(
                        branch_1, depth(96), [3, 3, 3], scope='Conv2d_0b_3x3x3')
                    branch_1 = slim.conv3d(
                        branch_1, depth(96), [3, 3, 3], stride=2, scope='Conv2d_1a_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool3d(
                        net, [3, 3, 3], stride=2, scope='MaxPool_1a_3x3x3')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 4 x 5 x 4 x 576
            end_point = 'Mixed_4b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(224), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(64), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(
                        branch_1, depth(96), [3, 3, 3], scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(96), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(128), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(128), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 4 x 5 x 4 x 576
            end_point = 'Mixed_4c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(96), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(128), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(96), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(128), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(128), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 4 x 5 x 4 x 576
            end_point = 'Mixed_4d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(160), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(160), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(160), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(160), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(96), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 4 x 5 x 4 x 576
            end_point = 'Mixed_4e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(96), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(192), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(160), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(192), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(192), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(96), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 4 x 5 x 4 x 576
            end_point = 'Mixed_5a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(
                        net, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_0 = slim.conv3d(branch_0, depth(192), [3, 3, 3], stride=2,
                                           scope='Conv2d_1a_3x3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(192), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(256), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_1 = slim.conv3d(branch_1, depth(256), [3, 3, 3], stride=2,
                                           scope='Conv2d_1a_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool3d(net, [3, 3, 3], stride=2,
                                               scope='MaxPool_1a_3x3x3')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 2 x 3 x 2 x 1024
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(352), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(192), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(320), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(160), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(224), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(224), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
            # 2 x 3 x 2 x 1024
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv3d(net, depth(352), [1, 1, 1], scope='Conv2d_0a_1x1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv3d(
                        net, depth(192), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_1 = slim.conv3d(branch_1, depth(320), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv3d(
                        net, depth(192), [1, 1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1x1')
                    branch_2 = slim.conv3d(branch_2, depth(224), [3, 3, 3],
                                           scope='Conv2d_0b_3x3x3')
                    branch_2 = slim.conv3d(branch_2, depth(224), [3, 3, 3],
                                           scope='Conv2d_0c_3x3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3x3')
                    branch_3 = slim.conv3d(
                        branch_3, depth(128), [1, 1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1x1')
                net = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
        raise ValueError('Unknown final endpoint {}'.format(final_endpoint))


def epilepsy_3d_cnn(inputs,
                    num_classes=2,
                    is_training=True,
                    dropout_keep_prob=0.8,
                    min_depth=16,
                    depth_multiplier=1.0,
                    prediction_fn=slim.softmax,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='Epilepsy_3D_CNN'):
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    with tf.variable_scope(scope, 'Epilepsy_3D_CNN', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = epilepsy_3d_cnn_base(
                inputs, scope=scope,
                min_depth=min_depth,
                depth_multiplier=depth_multiplier)

            with tf.variable_scope('Logits'):
                # Pooling with a fixed kernel size.
                kernel_size = _reduced_kernel_size_for_small_input(net, [2, 3, 2])
                net = slim.avg_pool3d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                end_points['AvgPool_1a'] = net

                if not num_classes:
                    return net, end_points
                # 1 x 1 x 1 x 1024
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv3d(net, num_classes, [1, 1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2, 3], name='SpatialSqueeze')
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


epilepsy_3d_cnn.default_image_size = (61, 73, 61)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None or shape[3] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1]),
                           min(shape[3], kernel_size[2])]
    return kernel_size_out


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
