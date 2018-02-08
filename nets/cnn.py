# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/10 16:56.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


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
