# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/5/8 21:50.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf


def WeightedCrossEntropyLoss(onehot_labels, logits, size_average=True):
  f_y = tf.sigmoid(logits)
  shape = onehot_labels.shape
  BetaP = tf.reduce_sum(onehot_labels) / tf.cast((shape[0] * shape[1]), tf.float32)
  BetaN = 1 - BetaP
  xx = BetaN * tf.reduce_sum(- onehot_labels * tf.log(f_y), axis=1)
  yy = BetaP * tf.reduce_sum(- (1 - onehot_labels) * tf.log(1 - f_y), axis=1)
  if size_average:
    return (xx + yy) / tf.cast((shape[0]), tf.float32)
  else:
    return xx + yy
