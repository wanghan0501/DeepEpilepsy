# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/28 20:44.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os
import sys
from datetime import datetime

import pandas as pd

if __name__ == '__main__':
  PACKAGE_PARENT = '..'
  SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
  sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
  from dataset.tfrecord import write_to_tfrecord
  from utils.log import Logger
from dataset.tfrecord import write_to_tfrecord
from utils.log import Logger

SAMPLING_RATE = 0.8
MAT_DIR = "../data/alff&reho_20180509/"
TRAIN_LABEL_DIR = "../data/cnn_alff&reho_20180509_train.txt"
TEST_LABEL_DIR = '../data/cnn_alff&reho_20180509_test.txt'
TF_DIR_PREFIX = '../tfdata/'

TF_DIR_PREFIX = '../tfdata/'

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_data():
  train_data = pd.read_csv(
    TRAIN_LABEL_DIR, delimiter=',')
  test_data = pd.read_csv(
    TEST_LABEL_DIR, delimiter=',')
  return train_data, test_data


def sampling():
  train_data, test_data = load_data()
  # sampling
  # train_X, test_X, train_Y, test_Y = train_test_split(
  #   label_data['name'],
  #   label_data['label'],
  #   test_size=1 - SAMPLING_RATE)

  TF_DIR = TF_DIR_PREFIX + start_time + '/'

  # create data dir
  if not os.path.exists(TF_DIR):
    os.mkdir(TF_DIR)

  logger = Logger(filename=TF_DIR + 'cnn_epilepsy_sampling.log').get_logger()
  logger.info('SAMPLING LOG')
  logger.info('sampling start time: {}'.format(start_time))
  logger.info('sampling rate: {}'.format(SAMPLING_RATE))
  logger.info('data dir: ' + MAT_DIR)
  logger.info('train lable dir: ' + TRAIN_LABEL_DIR)
  logger.info('test label dir: ' + TEST_LABEL_DIR)
  logger.info("original training set: {}".format(train_data.shape))
  logger.info("original testing set: {}".format(test_data.shape))
  logger.info("******************")

  # write train tfrecords
  write_to_tfrecord(
    TF_DIR + "epilepsy_cnn_train.tfrecords",
    datas=train_data['name'].apply(lambda x: MAT_DIR + x).tolist(),
    labels=train_data['label'].tolist(),
    logger=logger)
  # write test tfrecords
  write_to_tfrecord(
    TF_DIR + "epilepsy_cnn_test.tfrecords",
    datas=test_data['name'].apply(lambda x: MAT_DIR + x).tolist(),
    labels=test_data['label'].tolist(),
    logger=logger)


if __name__ == '__main__':
  sampling()
