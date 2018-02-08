# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
Created by Wang Han on 08/02/2018 11:36.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All right Reserved.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv3d(net, 32, [1, 1, 1], scope='Conv3d_1x1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv3d(net, 32, [1, 1, 1], scope='Conv3d_0a_1x1x1')
      tower_conv1_1 = slim.conv3d(tower_conv1_0, 32, 3, scope='Conv3d_0b_3x3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv3d(net, 32, [1, 1, 1], scope='Conv3d_0a_1x1x1')
      tower_conv2_1 = slim.conv3d(tower_conv2_0, 48, [3, 3, 3], scope='Conv3d_0b_3x3x3')
      tower_conv2_2 = slim.conv3d(tower_conv2_1, 64, [3, 3, 3], scope='Conv3d_0c_3x3x3')
    mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv3d(mixed, net.get_shape()[4], [1, 1, 1], normalizer_fn=None,
                     activation_fn=None, scope='Conv3d_1x1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv3d(net, 192, [1, 1, 1], scope='Conv3d_1x1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv3d(net, 128, [1, 1, 1], scope='Conv3d_0a_1x1x1')
      tower_conv1_1 = slim.conv3d(tower_conv1_0, 160, [1, 1, 7],
                                  scope='Conv3d_0b_1x1x7')
      tower_conv1_2 = slim.conv3d(tower_conv1_1, 160, [1, 7, 1],
                                  scope='Conv3d_0c_1x7x1')
      tower_conv1_3 = slim.conv3d(tower_conv1_2, 192, [7, 1, 1],
                                  scope='Conv3d_0d_7x1x1')
    mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_3])
    up = slim.conv3d(mixed, net.get_shape()[4], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv3d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv3d(net, 192, [1, 1, 1], scope='Conv3d_1x1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv3d(net, 192, [1, 1, 1], scope='Conv3d_0a_1x1x1')
      tower_conv1_1 = slim.conv3d(tower_conv1_0, 224, [1, 1, 3],
                                  scope='Conv3d_0b_1x1x3')
      tower_conv1_2 = slim.conv3d(tower_conv1_1, 224, [1, 3, 1],
                                  scope='Conv3d_0b_1x3x1')
      tower_conv1_3 = slim.conv3d(tower_conv1_2, 256, [3, 1, 1],
                                  scope='Conv3d_0c_3x1x1')
    mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_3])
    up = slim.conv3d(mixed, net.get_shape()[4], [1, 1, 1], normalizer_fn=None,
                     activation_fn=None, scope='Conv3d_1x1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_3d_v2_base(inputs,
                                final_endpoint='Conv3d_7b_1x1x1',
                                output_stride=16,
                                align_feature_maps=False,
                                scope=None,
                                activation_fn=tf.nn.relu):
  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  padding = 'SAME' if align_feature_maps else 'VALID'

  end_points = {}
  concat_dim = 4

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnet3DV2', [inputs]):
    with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                        stride=1, padding='SAME'):
      # 149 x 149 x 32
      end_point = 'Conv3d_1a_3x3x3'
      net = slim.conv3d(inputs, 32, [3, 3, 3], stride=2, padding=padding, scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 147 x 147 x 32
      end_point = 'Conv3d_2a_3x3x3'
      net = slim.conv3d(net, 32, [3, 3, 3], padding=padding, scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 147 x 147 x 64
      end_point = 'Conv3d_2b_3x3x3'
      net = slim.conv3d(net, 64, [3, 3, 3], scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 73 x 73 x 64
      end_point = 'MaxPool_3a_3x3x3'
      net = slim.max_pool3d(net, [3, 3, 3], stride=2, padding=padding, scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 73 x 73 x 80
      end_point = 'Conv3d_3b_1x1x1'
      net = slim.conv3d(net, 80, [1, 1, 1], padding=padding, scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 71 x 71 x 192
      end_point = 'Conv3d_4a_3x3x3'
      net = slim.conv3d(net, 192, [3, 3, 3], padding=padding, scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points
      # 35 x 35 x 192
      end_point = 'MaxPool_5a_3x3x3'
      net = slim.max_pool3d(net, 3, stride=2, padding=padding, scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points

      # 35 x 35 x 320
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv3d(net, 96, [1, 1, 1], scope='Conv3d_1x1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv3d(net, 48, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          tower_conv1_1 = slim.conv3d(tower_conv1_0, 64, [5, 5, 5],
                                      scope='Conv3d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv3d(net, 64, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          tower_conv2_1 = slim.conv3d(tower_conv2_0, 96, [3, 3, 3],
                                      scope='Conv3d_0b_3x3x3')
          tower_conv2_2 = slim.conv3d(tower_conv2_1, 96, [3, 3, 3],
                                      scope='Conv3d_0c_3x3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool3d(net, [3, 3, 3], stride=1, padding='SAME',
                                       scope='AvgPool_0a_3x3x3')
          tower_pool_1 = slim.conv3d(tower_pool, 64, [1, 1, 1],
                                     scope='Conv3d_0b_1x1x1')
        net = tf.concat(values=[tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], axis=concat_dim)

      if add_and_check_final(end_point, net): return net, end_points
      # register intermediate endpoints
      net = slim.repeat(net, 10, block35, scale=0.17,
                        activation_fn=activation_fn)

      # 17 x 17 x 1088 if output_stride == 8,
      # 33 x 33 x 1088 if output_stride == 16
      use_atrous = output_stride == 8
      end_point = 'Mixed_6a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv3d(net, 384, 3, stride=1 if use_atrous else 2,
                                   padding=padding,
                                   scope='Conv3d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv3d(net, 256, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          tower_conv1_1 = slim.conv3d(tower_conv1_0, 256, [3, 3, 3],
                                      scope='Conv3d_0b_3x3')
          tower_conv1_2 = slim.conv3d(tower_conv1_1, 384, [3, 3, 3],
                                      stride=1 if use_atrous else 2,
                                      padding=padding,
                                      scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool3d(net, [3, 3, 3], stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3x3')
        net = tf.concat(values=[tower_conv, tower_conv1_2, tower_pool], axis=concat_dim)

      if add_and_check_final(end_point, net): return net, end_points

      # register intermediate endpoints
      with slim.arg_scope([slim.conv3d], rate=2 if use_atrous else 1):
        net = slim.repeat(net, 20, block17, scale=0.10,
                          activation_fn=activation_fn)
      if add_and_check_final('PreAuxLogits', net): return net, end_points

      if output_stride == 8:
        # properly support output_stride for the rest of the net.
        raise ValueError('output_stride==8 is only supported up to the '
                         'PreAuxlogits end_point for now.')

      # 8 x 8 x 2080
      end_point = 'Mixed_7a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv3d(net, 256, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          tower_conv_1 = slim.conv3d(tower_conv, 384, [3, 3, 3], stride=2,
                                     padding=padding,
                                     scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv3d(net, 256, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          tower_conv1_1 = slim.conv3d(tower_conv1, 288, [3, 3, 3], stride=2,
                                      padding=padding,
                                      scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv3d(net, 256, [1, 1, 1], scope='Conv3d_0a_1x1x1')
          tower_conv2_1 = slim.conv3d(tower_conv2, 288, [3, 3, 3],
                                      scope='Conv3d_0b_3x3x3')
          tower_conv2_2 = slim.conv3d(tower_conv2_1, 320, [3, 3, 3], stride=2,
                                      padding=padding,
                                      scope='Conv3d_1a_3x3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool3d(net, [3, 3, 3], stride=2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3x3')
        net = tf.concat(
          [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], concat_dim)

      if add_and_check_final(end_point, net): return net, end_points

      # register intermediate endpoints
      net = slim.repeat(net, 9, block8, scale=0.20, activation_fn=activation_fn)
      net = block8(net, activation_fn=None)

      # 8 x 8 x 1536
      end_point = 'Conv3d_7b_1x1x1'
      net = slim.conv3d(net, 1536, [1, 1, 1], scope=end_point)
      if add_and_check_final(end_point, net): return net, end_points

    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)


def inception_resnet_3d_v2(inputs, num_classes=1001, is_training=True,
                           dropout_keep_prob=0.8,
                           reuse=None,
                           scope='InceptionResnet3DV2',
                           create_aux_logits=True,
                           activation_fn=tf.nn.relu):
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnet3DV2', [inputs],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

      net, end_points = inception_resnet_3d_v2_base(inputs, scope=scope,
                                                    activation_fn=activation_fn)

      if create_aux_logits and num_classes:
        end_points = 'AuxLogits'
        with tf.variable_scope(end_points):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool3d(aux, 5, stride=3, padding='VALID',
                                scope='Conv3d_1a_3x3x3')
          aux = slim.conv3d(aux, 128, 1, scope='Conv3d_1b_1x1x1')
          aux = slim.conv3d(aux, 768, aux.get_shape()[1:4],
                            padding='VALID', scope='Conv3d_2a_5x5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points[end_points] = aux

      with tf.variable_scope('Logits'):
        # Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        kernel_size = net.get_shape()[1:4]
        if kernel_size.is_fully_defined():
          net = slim.avg_pool3d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_8x8x8')
        else:
          net = tf.reduce_mean(net, [1, 2, 3], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
        if not num_classes:
          return net, end_points
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')
        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points


inception_resnet_3d_v2.default_image_size = (61, 73, 61, 2)


def inception_resnet_3d_v2_arg_scope(weight_decay=0.00004,
                                     batch_norm_decay=0.9997,
                                     batch_norm_epsilon=0.001,
                                     activation_fn=tf.nn.relu):
  """Returns the scope with the default parameters for inception_resnet_v2.
  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Activation function for Conv3d.
  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in Conv3d and fully_connected layers.
  with slim.arg_scope([slim.conv3d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):
    batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'fused': None,  # Use fused batch norm if possible.
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv3d], activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope
