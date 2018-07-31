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
  from dataset.tfrecord_rnn import write_to_tfrecord
  from utils.log import Logger
from dataset.tfrecord_rnn import write_to_tfrecord
from utils.log import Logger

SAMPLING_RATE = 0.8
MAT_DIR = "../data/rnn_20180527/"
TRAIN_LABEL_DIR = "../data/rnn_20180527_train.txt"
TEST_LABEL_DIR = '../data/rnn_20180527_test.txt'
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

  TF_DIR = TF_DIR_PREFIX + start_time + '/'

  # create data dir
  if not os.path.exists(TF_DIR):
    os.mkdir(TF_DIR)

  logger = Logger(filename=TF_DIR + 'rnn_epilepsy_sampling.log').get_logger()
  logger.info('SAMPLING LOG')
  logger.info('sampling start time: {}'.format(start_time))
  logger.info('sampling rate: {}'.format(SAMPLING_RATE))
  logger.info('data dir: ' + MAT_DIR)
  logger.info('train lable dir: ' + TRAIN_LABEL_DIR)
  logger.info('test label dir: ' + TEST_LABEL_DIR)
  logger.info("original training set: {}".format(train_data.shape))
  logger.info("original testing set: {}".format(test_data.shape))
  logger.info("******************")

  temp = train_data['name'].tolist()
  train_datas = []
  for item in temp:
    train_datas.append([MAT_DIR + 'ROISignals_' + item + '.mat',
                        MAT_DIR + 'ROICorrelation_' + item + '.mat'])
  # write train tfrecords
  write_to_tfrecord(
    TF_DIR + "epilepsy_rnn_train.tfrecords",
    datas=train_datas,
    labels=train_data['label'].tolist(),
    img_shape=(190, 160),
    logger=logger,
    use_avg=False)

  temp = test_data['name'].tolist()
  test_datas = []
  for item in temp:
    test_datas.append([MAT_DIR + 'ROISignals_' + item + '.mat',
                        MAT_DIR + 'ROICorrelation_' + item + '.mat'])
  # write test tfrecords
  write_to_tfrecord(
    TF_DIR + "epilepsy_rnn_test.tfrecords",
    datas=test_datas,
    labels=test_data['label'].tolist(),
    img_shape=(95, 160),
    logger=logger,
    use_avg=True)


if __name__ == '__main__':
  sampling()
