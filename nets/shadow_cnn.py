# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/5/10 10:38.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def epilepsy_3d_cnn_base(inputs,
                         final_endpoint='MaxPool_6_2x2x2',
                         dropout_keep_prob=1.0,
                         scope=None):
  end_points = {}
  with tf.variable_scope(scope, 'Epilepsy_3D_CNN', [inputs]):
    with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                        stride=2,
                        padding='SAME'):
      # 61 x 73 x 61 x 2
      end_point = 'Conv3d_1_5x5x5'
      net = slim.conv3d(inputs,
                        num_outputs=128,
                        kernel_size=[5, 5, 5],
                        weights_initializer=trunc_normal(1.0),
                        scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # 31 x 37 x 31 x 128
      end_point = 'Conv3d_2_3x3x3'
      net = slim.conv3d(net,
                        num_outputs=256,
                        kernel_size=[3, 3, 3],
                        weights_initializer=trunc_normal(0.5),
                        scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # 16 x 19 x 16 x 256
      end_point = 'MaxPool_3_3x3x3'
      net = slim.max_pool3d(net,
                            kernel_size=[3, 3, 3],
                            scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      net = slim.dropout(net, dropout_keep_prob, scope='Dropout')

      # 8 x 10 x 8 x 256
      end_point = 'Con3d_4_3x3x3'
      net = slim.conv3d(net,
                        num_outputs=512,
                        kernel_size=[3, 3, 3],
                        weights_initializer=trunc_normal(0.1),
                        scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # 4 x 5 x 4 x 512
      end_point = 'Con3d_5_2x2x2'
      net = slim.conv3d(net,
                        num_outputs=1024,
                        kernel_size=[2, 2, 2],
                        stride=1,
                        weights_initializer=trunc_normal(0.09),
                        scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # 4 x 5 x 4 x 1024
      end_point = 'MaxPool_6_2x2x2'
      net = slim.max_pool3d(net,
                            kernel_size=[2, 2, 2],
                            scope=end_point)
      end_points[end_point] = net
      net = slim.dropout(net, dropout_keep_prob, scope='Dropout')
      # 2 x 3 x 2 x 1024
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
    with slim.arg_scope([slim.dropout], is_training=is_training):
      net, end_points = epilepsy_3d_cnn_base(inputs, scope=scope, dropout_keep_prob=dropout_keep_prob)

      with tf.variable_scope('Logits'):
        net = slim.avg_pool3d(net, [2, 3, 2], padding='VALID', scope='AvgPool_7_2x3x2')
        logits = slim.conv3d(net, num_classes, [1, 1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv3d_8_1x1x1')
        logits = tf.squeeze(logits, [1, 2, 3], name='SpatialSqueeze')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
