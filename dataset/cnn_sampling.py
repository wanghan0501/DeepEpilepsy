# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/18 20:45.
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
  from dataset.tfrecord import write_to_tfrecord
  from utils.log import Logger

from dataset.tfrecord import write_to_tfrecord
from utils.log import Logger

NORMAL_DIR = '../data/normal/'
EPILEPSY_DIR = '../data/epilepsy/'
TF_DIR_PREFIX = '../tfdata/'
SAMPLING_RATE = 0.8


def load_data():
  normal_data = list(filter(lambda x: x.endswith('mat'), os.listdir(NORMAL_DIR)))
  normal_data = list(map(lambda x: NORMAL_DIR + x, normal_data))
  normal_labels = list(np.zeros(len(normal_data)))

  epilepsy_data = list(filter(lambda x: x.endswith('mat'), os.listdir(EPILEPSY_DIR)))
  epilepsy_data = list(map(lambda x: EPILEPSY_DIR + x, epilepsy_data))
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

  logger = Logger(filename=TF_DIR + 'tf_sampling.log').get_logger()
  logger.info('SAMPLING LOG')
  logger.info('sampling rate: {}'.format(SAMPLING_RATE))
  logger.info('normal data dir: ' + NORMAL_DIR)
  logger.info('epilepsy data idr: ' + EPILEPSY_DIR)
  logger.info("original total set: {}".format(len(data)))
  logger.info("original training set: {}".format(len(train_X)))
  logger.info("original testing set: {}".format(len(test_X)))
  logger.info("******************")

  # write train tfrecords
  write_to_tfrecord(TF_DIR + "epilepsy_cnn_train.tfrecords",
                    datas=train_X,
                    labels=train_Y,
                    logger=logger)

  # write test tfrecords
  write_to_tfrecord(TF_DIR + "epilepsy_cnn_test.tfrecords",
                    datas=test_X,
                    labels=test_Y,
                    logger=logger)


if __name__ == '__main__':
  sampling()
