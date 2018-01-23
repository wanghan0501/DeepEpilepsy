# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/23 19:42.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os

import tensorflow as tf

from dataset.tfrecord import read_from_tfrecord
from integration.inference_integration import Inference

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

inference = Inference()
inference.setup()

TEST_TFRECODE = 'tfdata/cnn_tfdata/epilepsy_cnn_test.tfrecords'
TEST_RECODE_NUM = 60

queue = tf.train.string_input_producer(
    [TEST_TFRECODE],num_epochs=1)
curr_image, curr_label = read_from_tfrecord(queue)
# get test batch data
test_batch_images, test_batch_labels = tf.train.batch(
    [curr_image, curr_label], capacity=100, batch_size=1)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    TP, TN, FP, FN = 0, 0, 0, 0
    for item in range(TEST_RECODE_NUM):
        curr_test_image, curr_test_label = sess.run(
            [test_batch_images, test_batch_labels])
        inference_prediction, inference_class = inference.inference(curr_test_image)
        if inference_class == curr_test_label:
            if inference_class == 0:
                TN += 1
            else:
                TP += 1
        else:
            if inference_class == 0 and curr_test_label == 1:
                FN += 1
            else:
                FP += 1

    print('TP:{}, TN:{}, FP:{}, FN:{}'.format(TP, TN, FP, FN))
    print('Accuracy:{:.6f}'.format((TP + TN) / (TP + TN + FP + FN)))
    coord.request_stop()
    coord.join(threads)
