# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
Created by Wang Han on 07/02/2018 16:49.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All right Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_3d_v3_base(inputs,
                         final_endpoint='Mixed_7c',
                         min_depth=16,
                         depth_multiplier=1.0,
                         scope=None):
  end_points = {}
  concat_dim = 4
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  '''
  in original inception_v3，the padding method of initial layers (from Conv3d_1a_3x3x3 to MaxPool_5a_3x3x3) are 
  'Valid', however, the data size of epilepsy is (61,73,61,2), so I change the padding method from 'Valid' to 'SAME' 
  to avoid getting the small feature size after convolution operation. 
  '''
  with tf.variable_scope(scope, 'Inception3DV3', [inputs]):
    with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                        stride=1, padding='SAME'):
      # 61 x 73 x 61 x 2
      end_point = 'Conv3d_1a_3x3x3'
      net = slim.conv3d(inputs, depth(32), [3, 3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 31 x 37 x 31 x 32
      end_point = 'Conv3d_2a_3x3x3'
      net = slim.conv3d(net, depth(32), [3, 3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 31 x 37 x 31 x 32
      end_point = 'Conv3d_2b_3x3x3'
      net = slim.conv3d(net, depth(64), [3, 3, 3], padding='SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 31 x 37 x 31 x 64
      end_point = 'MaxPool_3a_3x3x3'
      net = slim.max_pool3d(net, [3, 3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 16 x 19 x 16 x 64
      end_point = 'Conv3d_3b_1x1x1'
      net = slim.conv3d(net, depth(80), [1, 1, 1], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 16 x 19 x 16 x 80
      end_point = 'Conv3d_4a_3x3x3'
      net = slim.conv3d(net, depth(192), [3, 3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 16 x 19 x 16 x 192
      end_point = 'MaxPool_5a_3x3x3'
      net = slim.max_pool3d(net, [3, 3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

    # Inception blocks
    with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                        stride=1, padding='SAME'):
      # mixed:  # 8 x 10 x 8 x 192
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(64), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(48), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(64), [5, 5, 5],
                                 scope='Conv3d_0b_5x5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(64), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                 scope='Conv3d_0b_3x3x3')
          branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                 scope='Conv3d_0c_3x3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(branch_3, depth(32), [1, 1, 1],
                                 scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_1: 8 x 10 x 8 x 256.
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(64), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(48), [1, 1, 1], scope='Conv3d_0b_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(64), [5, 5, 5],
                                 scope='Conv_1_0c_5x5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(64), [1, 1, 1],
                                 scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                 scope='Conv3d_0b_3x3x3')
          branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                 scope='Conv3d_0c_3x3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(branch_3, depth(64), [1, 1, 1],
                                 scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_2: 8 x 10 x 8 x 288.
      end_point = 'Mixed_5d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(64), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(48), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(64), [5, 5, 5],
                                 scope='Conv3d_0b_5x5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(64), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                 scope='Conv3d_0b_3x3x3')
          branch_2 = slim.conv3d(branch_2, depth(96), [3, 3, 3],
                                 scope='Conv3d_0c_3x3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(branch_3, depth(64), [1, 1, 1],
                                 scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_3: 8 x 10 x 8 x 288.
      end_point = 'Mixed_6a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(384), [3, 3, 3], padding='VALID',
                                 scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(64), [1, 1, 1],
                                 scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(96), [3, 3, 3],
                                 scope='Conv3d_0b_3x3x3')
          branch_1 = slim.conv3d(branch_1, depth(96), [3, 3, 3], padding='VALID',
                                 scope='Conv3d_1a_1x1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool3d(net, [3, 3, 3], padding='VALID',
                                     scope='MaxPool_1a_3x3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed4: 6 x 8 x 6 x 768.
      end_point = 'Mixed_6b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(128), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(128), [1, 1, 5],
                                 scope='Conv3d_0b_1x1x5')
          branch_1 = slim.conv3d(branch_1, depth(128), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [5, 1, 1],
                                 scope='Conv3d_0d_5x1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(128), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(branch_2, depth(128), [5, 1, 1],
                                 scope='Conv3d_0b_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(128), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(128), [1, 1, 5],
                                 scope='Conv3d_0d_1x1x5')
          branch_2 = slim.conv3d(branch_2, depth(128), [5, 1, 1],
                                 scope='Conv3d_0e_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(128), [1, 5, 1],
                                 scope='Conv3d_0f_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [1, 1, 5],
                                 scope='Conv3d_0g_1x1x5')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(branch_3, depth(192), [1, 1, 1],
                                 scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_5: 6 x 8 x 6 x 768.
      end_point = 'Mixed_6c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(160), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(160), [1, 1, 5],
                                 scope='Conv3d_0b_1x1x5')
          branch_1 = slim.conv3d(branch_1, depth(160), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [5, 1, 1],
                                 scope='Conv3d_0d_5x1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(160), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [5, 1, 1],
                                 scope='Conv3d_0b_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [1, 1, 5],
                                 scope='Conv3d_0d_1x1x5')
          branch_2 = slim.conv3d(branch_2, depth(160), [5, 1, 1],
                                 scope='Conv3d_0e_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [1, 5, 1],
                                 scope='Conv3d_0f_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [1, 1, 5],
                                 scope='Conv3d_0g_1x1x5')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(branch_3, depth(192), [1, 1, 1],
                                 scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_6: 6 x 8 x 6 x 768.
      end_point = 'Mixed_6d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(160), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(160), [1, 1, 5],
                                 scope='Conv3d_0b_1x1x5')
          branch_1 = slim.conv3d(branch_1, depth(160), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [5, 1, 1],
                                 scope='Conv3d_0d_5x1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(160), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [5, 1, 1],
                                 scope='Conv3d_0b_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [1, 1, 5],
                                 scope='Conv3d_0d_1x1x5')
          branch_2 = slim.conv3d(branch_2, depth(160), [5, 1, 1],
                                 scope='Conv3d_0e_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(160), [1, 5, 1],
                                 scope='Conv3d_0f_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [1, 1, 5],
                                 scope='Conv3d_0g_1x1x5')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(branch_3, depth(192), [1, 1, 1],
                                 scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_7: 6 x 8 x 6 x 768.
      end_point = 'Mixed_6e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [1, 1, 5],
                                 scope='Conv3d_0b_1x1x5')
          branch_1 = slim.conv3d(branch_1, depth(192), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [5, 1, 1],
                                 scope='Conv3d_0d_5x1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [5, 1, 1],
                                 scope='Conv3d_0b_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [1, 5, 1],
                                 scope='Conv3d_0c_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [1, 1, 5],
                                 scope='Conv3d_0d_1x1x5')
          branch_2 = slim.conv3d(branch_2, depth(192), [5, 1, 1],
                                 scope='Conv3d_0e_5x1x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [1, 5, 1],
                                 scope='Conv3d_0f_1x5x1')
          branch_2 = slim.conv3d(branch_2, depth(192), [1, 1, 5],
                                 scope='Conv3d_0g_1x1x5')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(branch_3, depth(192), [1, 1, 1],
                                 scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_8: 6 x 8 x 6 x 768.
      end_point = 'Mixed_7a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_0 = slim.conv3d(branch_0, depth(320), [3, 3, 3], stride=2,
                                 padding='VALID', scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(192), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [1, 1, 7],
                                 scope='Conv3d_0b_1x1x7')
          branch_1 = slim.conv3d(branch_1, depth(192), [1, 7, 1],
                                 scope='Conv3d_0c_1x7x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [7, 1, 1],
                                 scope='Conv3d_0d_7x1x1')
          branch_1 = slim.conv3d(branch_1, depth(192), [3, 3, 3], stride=2,
                                 padding='VALID', scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool3d(net, [3, 3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      '''
      in original inception_v3，the depth of Mixed_7b/Brach_1 is 384. And I change the depth of it from '384' to
       '256' to keep the number of channel. 
      '''
      # mixed_9: 2 x 3 x 2 x 1280.
      end_point = 'Mixed_7b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(320), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(384), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = tf.concat(axis=concat_dim, values=[
            slim.conv3d(branch_1, depth(256), [1, 1, 3], scope='Conv3d_0b_1x1x3'),
            slim.conv3d(branch_1, depth(256), [1, 3, 1], scope='Conv3d_0c_1x3x1'),
            slim.conv3d(branch_1, depth(256), [3, 1, 1], scope='Conv3d_0d_3x1x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(384), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(
            branch_2, depth(256), [3, 3, 3], scope='Conv3d_0b_3x3x3')
          branch_2 = tf.concat(axis=concat_dim, values=[
            slim.conv3d(branch_2, depth(256), [1, 1, 3], scope='Conv3d_0c_1x1x3'),
            slim.conv3d(branch_2, depth(256), [1, 3, 1], scope='Conv3d_0d_1x3x1'),
            slim.conv3d(branch_2, depth(256), [3, 1, 1], scope='Conv3d_0e_3x1x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3x3')
          branch_3 = slim.conv3d(
            branch_3, depth(192), [1, 1, 1], scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_10: 2 x 3 x 2 x 2048.
      end_point = 'Mixed_7c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, depth(320), [1, 1, 1], scope='Conv3d_0a_1x1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, depth(256), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = tf.concat(axis=concat_dim, values=[
            slim.conv3d(branch_1, depth(256), [1, 1, 3], scope='Conv3d_0b_1x1x3'),
            slim.conv3d(branch_1, depth(256), [1, 3, 1], scope='Conv3d_0c_1x3x1'),
            slim.conv3d(branch_1, depth(256), [3, 1, 1], scope='Conv3d_0d_3x1x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv3d(net, depth(384), [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_2 = slim.conv3d(
            branch_2, depth(256), [3, 3, 3], scope='Conv3d_0b_3x3x3')
          branch_2 = tf.concat(axis=concat_dim, values=[
            slim.conv3d(branch_2, depth(256), [1, 1, 3], scope='Conv3d_0c_1x1x3'),
            slim.conv3d(branch_2, depth(256), [1, 1, 3], scope='Conv3d_0d_1x3x1'),
            slim.conv3d(branch_2, depth(256), [3, 1, 1], scope='Conv3d_0f_3x1x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool3d(net, [3, 3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv3d(
            branch_3, depth(192), [1, 1, 1], scope='Conv3d_0b_1x1x1')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_3d_v3(inputs,
                    num_classes=2,
                    is_training=True,
                    dropout_keep_prob=0.8,
                    min_depth=16,
                    depth_multiplier=1.0,
                    prediction_fn=slim.softmax,
                    spatial_squeeze=True,
                    reuse=None,
                    create_aux_logits=True,
                    scope='Inception3DV3',
                    global_pool=False):
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'Inception3DV3', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_3d_v3_base(
        inputs, scope=scope,
        min_depth=min_depth,
        depth_multiplier=depth_multiplier)

      # Auxiliary Head logits
      if create_aux_logits and num_classes:
        with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                            stride=1, padding='SAME'):
          aux_logits = end_points['Mixed_6e']
          with tf.variable_scope('AuxLogits'):
            kernel_size = _reduced_kernel_size_for_small_input(net, [5, 5, 5])
            aux_logits = slim.avg_pool3d(aux_logits, kernel_size, padding='VALID',
                                         scope='AvgPool_1a_{}x{}x{}'.format(*kernel_size))
            aux_logits = slim.conv3d(aux_logits, depth(128), [1, 1, 1],
                                     scope='Conv3d_1b_1x1x1')
            # shape of feature map before the final layer.
            kernel_size = _reduced_kernel_size_for_small_input(
              aux_logits, [5, 6, 5])
            aux_logits = slim.conv3d(
              aux_logits, depth(768), kernel_size,
              weights_initializer=trunc_normal(0.01),
              padding='VALID', scope='Conv3d_2a_{}x{}x{}'.format(*kernel_size))
            aux_logits = slim.conv3d(
              aux_logits, num_classes, [1, 1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
              scope='Conv3d_2b_1x1x1')
            if spatial_squeeze:
              aux_logits = tf.squeeze(aux_logits, [1, 2, 3], name='SpatialSqueeze')
            end_points['AuxLogits'] = aux_logits

      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2, 3], keepdims=True, name='GlobalPool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [2, 3, 2])
          net = slim.avg_pool3d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_{}x{}x{}'.format(*kernel_size))
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = slim.conv3d(net, num_classes, [1, 1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv3d_1c_1x1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2, 3], name='SpatialSqueeze')
        # 2
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None or shape[3] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1]),
                       min(shape[3], kernel_size[2])]
  return kernel_size_out
