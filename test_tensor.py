# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/1/13 11:23.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os

import numpy as np
import scipy.io as scio
import tensorflow as tf
from tensorflow.contrib import slim

from nets.model import Epilepsy3dCnn
from utils.config import CNNConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataFile = 'data/epilepsy/P001.mat'
data1 = scio.loadmat(dataFile)
for key in data1.keys():
    if not key.startswith('__'):
        data1 = data1[key]
data1 = data1[np.newaxis, :, :, :, :]
print(data1.shape)

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 61, 73, 61, 2])

net = slim.conv3d(inputs,
                  num_outputs=256,
                  kernel_size=[3, 3, 3],
                  stride=2,
                  padding='SAME')
net = slim.max_pool3d(net,
                      kernel_size=[3, 3, 3],
                      stride=2,
                      padding='SAME')
net = slim.conv3d(net,
                  num_outputs=512,
                  kernel_size=[3, 3, 3],
                  stride=2,
                  padding='SAME')
net = slim.max_pool3d(net,
                      kernel_size=[2, 2, 2],
                      stride=2,
                      padding='SAME')
net = slim.conv3d(net,
                  num_outputs=256,
                  kernel_size=[2, 2, 2],
                  stride=2,
                  padding='SAME')

# 1 x 2 x 1 x 256
end_point = 'MaxPool_3_3x3x3'
net = slim.max_pool3d(net,
                      kernel_size=[2, 2, 2],
                      stride=2,
                      padding='SAME')

net = slim.flatten(net, scope='Flatten')
conf = CNNConfig(is_training=True)
model = Epilepsy3dCnn(config=conf)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    output = sess.run(net, {inputs: data1})
    print(np.shape(output))
