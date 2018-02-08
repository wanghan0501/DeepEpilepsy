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
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
  PACKAGE_PARENT = '..'
  SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
  sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
  from dataset.tfrecord import write_to_tfrecord
  from utils.log import Logger
from dataset.tfrecord import write_to_tfrecord
from utils.log import Logger

SAMPLING_RATE = 0.8
MAT_DIR = "../data/roi_20180201/"
LABEL_DIR = "../data/rnn_roi_20180201.txt"
TF_DIR_PREFIX = '../tfdata/'

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_data():
  label_data = pd.read_csv(
    LABEL_DIR, names=['mat_path', 'label'], delimiter=',')
  return label_data


def sampling():
  label_data = load_data()
  # sampling
  train_X, test_X, train_Y, test_Y = train_test_split(
    label_data['mat_path'],
    label_data['label'],
    test_size=1 - SAMPLING_RATE)

  TF_DIR = TF_DIR_PREFIX + start_time + '/'

  # create data dir
  if not os.path.exists(TF_DIR):
    os.mkdir(TF_DIR)

  logger = Logger(filename=TF_DIR + 'rnn_epilepsy_sampling.log').get_logger()
  logger.info('SAMPLING LOG')
  logger.info('sampling start time: {}'.format(start_time))
  logger.info('sampling rate: {}'.format(SAMPLING_RATE))
  logger.info('data dir: ' + MAT_DIR)
  logger.info('lable idr: ' + LABEL_DIR)
  logger.info("original total set: {}".format(label_data.shape))
  logger.info("original training set: {}".format(train_X.shape))
  logger.info("original testing set: {}".format(test_X.shape))
  logger.info("******************")

  # write train tfrecords
  write_to_tfrecord(
    TF_DIR + "epilepsy_rnn_train.tfrecords",
    datas=train_X.apply(lambda x: MAT_DIR + x).tolist(),
    labels=train_Y.tolist(),
    img_shape=(95, 160),
    logger=logger,
    use_avg=True)
  # write test tfrecords
  write_to_tfrecord(
    TF_DIR + "epilepsy_rnn_test.tfrecords",
    datas=test_X.apply(lambda x: MAT_DIR + x).tolist(),
    labels=test_Y.tolist(),
    img_shape=(95, 160),
    logger=logger,
    use_avg=True)


if __name__ == '__main__':
  sampling()
