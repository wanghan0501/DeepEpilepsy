# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/20 16:15.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os
import sys
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from dataset.tfrecord import write_to_tfrecord, read_nii
    from utils.log import Logger

from dataset.tfrecord import write_to_tfrecord, read_nii
from utils.log import Logger

NORMAL_DIR = '/root/workspace/epilepsy_data/normal/FunImgARW/'
EPILEPSY_DIR = '/root/workspace/epilepsy_data/epilepsy/FunImgARW/'
TF_DIR_PREFIX = '../tfdata/'
SAMPLING_RATE = 0.8


def load_data():
    normal_data = []
    for dir in os.listdir(NORMAL_DIR):
        cur_path = NORMAL_DIR + dir + '/'
        for item in os.listdir(cur_path):
            if item.endswith('nii'):
                normal_data.append(cur_path + item)
    normal_labels = list(np.zeros(len(normal_data)))

    epilepsy_data = []
    for dir in os.listdir(EPILEPSY_DIR):
        cur_path = EPILEPSY_DIR + dir + '/'
        for item in os.listdir(cur_path):
            if item.endswith('nii'):
                epilepsy_data.append(cur_path + item)
    epilepsy_labels = list(np.ones(len(epilepsy_data)))

    data = normal_data + epilepsy_data
    labels = normal_labels + epilepsy_labels

    return data, labels


def sampling():
    data, labels = load_data()
    train_X, test_X, train_Y, test_Y = train_test_split(data, labels,
                                                        test_size=1 - SAMPLING_RATE)
    TF_DIR = TF_DIR_PREFIX + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '/'

    # create data dir
    if not os.path.exists(TF_DIR):
        os.mkdir(TF_DIR)

    logger = Logger(filename=TF_DIR + 'tf_rnn_sampling.log').get_logger()
    logger.info('RNN SAMPLING LOG')
    logger.info('sampling rate: {}'.format(SAMPLING_RATE))
    logger.info('normal data dir: ' + NORMAL_DIR)
    logger.info('epilepsy data idr: ' + EPILEPSY_DIR)
    logger.info("original total set: {}".format(len(data)))
    logger.info("original training set: {}".format(len(train_X)))
    logger.info("original testing set: {}".format(len(test_X)))
    logger.info("******************")

    # write train tfrecords
    write_to_tfrecord(TF_DIR + "epilepsy_rnn_train.tfrecords",
                      datas=train_X,
                      labels=train_Y,
                      logger=logger,
                      read_function=read_nii,
                      img_shape=(61, 73, 61, 190))

    # write test tfrecords
    write_to_tfrecord(TF_DIR + "epilepsy_rnn_test.tfrecords",
                      datas=test_X,
                      labels=test_Y,
                      logger=logger,
                      read_function=read_nii,
                      img_shape=(61, 73, 61, 190))


if __name__ == '__main__':
    sampling()
