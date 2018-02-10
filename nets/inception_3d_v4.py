# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
Created by Wang Han on 09/02/2018 17:01.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All right Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def block_inception_a(inputs, scope=None, reuse=None):
  """Builds Inception-A block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv3d, slim.avg_pool3d, slim.max_pool3d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv3d(inputs, 96, [1, 1, 1], scope='Conv3d_0a_1x1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv3d(inputs, 64, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_1 = slim.conv3d(branch_1, 96, [3, 3, 3], scope='Conv3d_0b_3x3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv3d(inputs, 64, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_2 = slim.conv3d(branch_2, 96, [3, 3, 3], scope='Conv3d_0b_3x3x3')
        branch_2 = slim.conv3d(branch_2, 96, [3, 3, 3], scope='Conv3d_0c_3x3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool3d(inputs, [3, 3, 3], scope='AvgPool_0a_3x3x3')
        branch_3 = slim.conv3d(branch_3, 96, [1, 1, 1], scope='Conv3d_0b_1x1x1')
      return tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_a(inputs, scope=None, reuse=None):
  """Builds Reduction-A block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv3d, slim.avg_pool3d, slim.max_pool3d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv3d(inputs, 384, [3, 3, 3], stride=2,
                               scope='Conv3d_1a_3x3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv3d(inputs, 192, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_1 = slim.conv3d(branch_1, 224, [3, 3, 3], scope='Conv3d_0b_3x3x3')
        branch_1 = slim.conv3d(branch_1, 256, [3, 3, 3], stride=2,
                               scope='Conv3d_1a_3x3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool3d(inputs, [3, 3, 3], stride=2,
                                   scope='MaxPool_1a_3x3x3')
      return tf.concat(axis=4, values=[branch_0, branch_1, branch_2])


def block_inception_b(inputs, scope=None, reuse=None):
  """Builds Inception-B block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv3d, slim.avg_pool3d, slim.max_pool3d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv3d(inputs, 384, [1, 1, 1], scope='Conv3d_0a_1x1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv3d(inputs, 192, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_1 = slim.conv3d(branch_1, 224, [1, 1, 5], scope='Conv3d_0b_1x1x5')
        branch_1 = slim.conv3d(branch_1, 224, [1, 5, 1], scope='Conv3d_0c_1x5x1')
        branch_1 = slim.conv3d(branch_1, 256, [5, 1, 1], scope='Conv3d_0d_5x1x1')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv3d(inputs, 192, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_2 = slim.conv3d(branch_2, 192, [5, 1, 1], scope='Conv3d_0b_5x1x1')
        branch_2 = slim.conv3d(branch_2, 224, [1, 5, 1], scope='Conv3d_0c_1x5x1')
        branch_2 = slim.conv3d(branch_2, 224, [1, 1, 5], scope='Conv3d_0d_1x1x5')
        branch_2 = slim.conv3d(branch_2, 224, [5, 1, 1], scope='Conv3d_0e_5x1x1')
        branch_2 = slim.conv3d(branch_2, 224, [1, 5, 1], scope='Conv3d_0f_1x5x1')
        branch_2 = slim.conv3d(branch_2, 256, [1, 1, 5], scope='Conv3d_0g_1x1x5')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool3d(inputs, [3, 3, 3], scope='AvgPool_0a_3x3x3')
        branch_3 = slim.conv3d(branch_3, 128, [1, 1, 1], scope='Conv3d_0b_1x1x1')
      return tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_b(inputs, scope=None, reuse=None):
  """Builds Reduction-B block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv3d, slim.avg_pool3d, slim.max_pool3d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv3d(inputs, 192, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_0 = slim.conv3d(branch_0, 192, [3, 3, 3], stride=2,
                               scope='Conv3d_1a_3x3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv3d(inputs, 256, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_1 = slim.conv3d(branch_1, 256, [5, 1, 1], scope='Conv3d_0b_5x1x1')
        branch_1 = slim.conv3d(branch_1, 288, [1, 5, 1], scope='Conv3d_0c_1x5x1')
        branch_1 = slim.conv3d(branch_1, 320, [1, 1, 5], scope='Conv3d_0d_1x1x5')
        branch_1 = slim.conv3d(branch_1, 320, [3, 3, 3], stride=2,
                               scope='Conv3d_1a_3x3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool3d(inputs, [3, 3, 3], stride=2,
                                   scope='MaxPool_1a_3x3x3')
      return tf.concat(axis=4, values=[branch_0, branch_1, branch_2])


def block_inception_c(inputs, scope=None, reuse=None):
  """Builds Inception-C block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv3d, slim.avg_pool3d, slim.max_pool3d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv3d(inputs, 192, [1, 1, 1], scope='Conv3d_0a_1x1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv3d(inputs, 384, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_1 = tf.concat(axis=4, values=[
          slim.conv3d(branch_1, 192, [1, 1, 3], scope='Conv3d_0b_1x1x3'),
          slim.conv3d(branch_1, 192, [1, 3, 1], scope='Conv3d_0c_1x3x1'),
          slim.conv3d(branch_1, 192, [3, 1, 1], scope='Conv3d_0d_3x1x1')])
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv3d(inputs, 384, [1, 1, 1], scope='Conv3d_0a_1x1x1')
        branch_2 = slim.conv3d(branch_2, 448, [3, 1, 1], scope='Conv3d_0b_3x1x1')
        branch_2 = slim.conv3d(branch_2, 512, [1, 3, 1], scope='Conv3d_0c_1x3x1')
        branch_2 = slim.conv3d(branch_2, 576, [1, 1, 3], scope='Conv3d_0d_1x1x3')
        branch_2 = tf.concat(axis=4, values=[
          slim.conv3d(branch_2, 192, [1, 1, 3], scope='Conv3d_0e_1x1x3'),
          slim.conv3d(branch_2, 192, [1, 3, 1], scope='Conv3d_0f_1x3x1'),
          slim.conv3d(branch_2, 192, [3, 1, 1], scope='Conv3d_0g_3x1x1')
        ])
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool3d(inputs, [3, 3, 3], scope='AvgPool_0a_3x3x3')
        branch_3 = slim.conv3d(branch_3, 192, [1, 1, 1], scope='Conv3d_0b_1x1x1')
      return tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])


def inception_3d_v4_base(inputs, final_endpoint='Mixed_7d', scope=None):
  """Creates the Inception V4 network up to the given final endpoint.
  Args:
    inputs: a 5-D tensor of size [batch_size, length, width, height, 2].
  """
  end_points = {}
  concat_dim = 4

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'Inception3DV4', [inputs]):
    with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                        stride=1, padding='SAME'):
      # 61 x 73 x 61 x 2
      end_point = 'Conv3d_1a_3x3x3'
      net = slim.conv3d(inputs, 32, [3, 3, 3], stride=2, scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 31 x 37 x 31 x 32
      end_point = 'Conv3d_2a_3x3x3'
      net = slim.conv3d(net, 32, [3, 3, 3], scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 31 x 37 x 31 x 32
      end_point = 'Conv3d_2b_3x3x3'
      net = slim.conv3d(net, 64, [3, 3, 3], scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 31 x 37 x 31 x 64
      end_point = 'Mixed_3a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.max_pool3d(net, [3, 3, 3], stride=2,
                                     scope='MaxPool_0a_3x3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, 96, [3, 3, 3], stride=2,
                                 scope='Conv3d_0a_3x3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1])
        if add_and_check_final(end_point, net): return net, end_points

      # 16 x 19 x 16 x 160
      end_point = 'Mixed_4a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, 64, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_0 = slim.conv3d(branch_0, 96, [3, 3, 3], padding='VALID',
                                 scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv3d(net, 64, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          branch_1 = slim.conv3d(branch_1, 64, [1, 1, 7], scope='Conv3d_0b_1x1x7')
          branch_1 = slim.conv3d(branch_1, 64, [1, 7, 1], scope='Conv3d_0c_1x7x1')
          branch_1 = slim.conv3d(branch_1, 64, [7, 1, 1], scope='Conv3d_0d_7x1x1')
          branch_1 = slim.conv3d(branch_1, 96, [3, 3, 3], padding='VALID',
                                 scope='Conv3d_1a_3x3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1])
        if add_and_check_final(end_point, net): return net, end_points

      # 14 x 17 x 14 x 192
      end_point = 'Mixed_5a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv3d(net, 192, [3, 3, 3], stride=2,
                                 scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.max_pool3d(net, [3, 3, 3], stride=2,
                                     scope='MaxPool_1a_3x3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1])
        if add_and_check_final(end_point, net): return net, end_points

      # 7 x 9 x 7 x 384
      # 4 x Inception-A blocks
      for idx in range(4):
        block_scope = 'Mixed_5' + chr(ord('b') + idx)
        net = block_inception_a(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points

      # 7 x 9 x 7 x 384
      # Reduction-A block
      net = block_reduction_a(net, 'Mixed_6a')
      if add_and_check_final('Mixed_6a', net): return net, end_points

      # 4 x 5 x 4 x 1024
      # 7 x Inception-B blocks
      for idx in range(7):
        block_scope = 'Mixed_6' + chr(ord('b') + idx)
        net = block_inception_b(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points

      # 4 x 5 x 4 x 1024
      # Reduction-B block
      net = block_reduction_b(net, 'Mixed_7a')
      if add_and_check_final('Mixed_7a', net): return net, end_points

      # 2 x 3 x 2 x 1536
      # 3 x Inception-C blocks
      for idx in range(3):
        block_scope = 'Mixed_7' + chr(ord('b') + idx)
        net = block_inception_c(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_3d_v4(inputs,
                    num_classes=2,
                    is_training=True,
                    dropout_keep_prob=0.5,
                    reuse=None,
                    scope='Inception3DV4',
                    prediction_fn=slim.softmax,
                    create_aux_logits=True):
  end_points = {}
  with tf.variable_scope(scope, 'Inception3DV4', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_3d_v4_base(inputs, scope=scope)

      with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                          stride=1, padding='SAME'):
        # Auxiliary Head logits
        if create_aux_logits and num_classes:
          with tf.variable_scope('AuxLogits'):
            # 4 x 5 x 4 x 1024
            aux_logits = end_points['Mixed_6h']
            kernel_size = _reduced_kernel_size_for_small_input(net, [5, 5, 5])
            aux_logits = slim.avg_pool3d(aux_logits, kernel_size, stride=2,
                                         padding='VALID',
                                         scope='AvgPool_1a_{}x{}x{}'.format(*kernel_size))
            aux_logits = slim.conv3d(aux_logits, 128, [1, 1, 1],
                                     scope='Conv3d_1b_1x1x1')
            aux_logits = slim.conv3d(aux_logits, 768,
                                     aux_logits.get_shape()[1:4],
                                     padding='VALID', scope='Conv3d_2a')
            aux_logits = slim.flatten(aux_logits)
            aux_logits = slim.fully_connected(aux_logits, num_classes,
                                              activation_fn=None,
                                              scope='Aux_logits')
            end_points['AuxLogits'] = aux_logits

        # Final pooling and prediction
        # consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        with tf.variable_scope('Logits'):
          # 2 x 3 x 2 x 1856
          kernel_size = net.get_shape()[1:4]
          if kernel_size.is_fully_defined():
            net = slim.avg_pool3d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
          else:
            net = tf.reduce_mean(net, [1, 2, 3], keep_dims=True,
                                 name='global_pool')
          end_points['global_pool'] = net
          if not num_classes:
            return net, end_points
          # 1 x 1 x 1536
          net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
          net = slim.flatten(net, scope='PreLogitsFlatten')
          end_points['PreLogitsFlatten'] = net
          # 1536
          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        scope='Logits')
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


inception_3d_v4.default_image_size = (61, 73, 61, 2)
