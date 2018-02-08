# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/28 20:44.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os
from datetime import datetime

import pandas as pd
import sys
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
MAT_DIR = "../data/data_20180128/"
LABEL_DIR = "../data/cnn_label_20180128.txt"
TF_DIR_PREFIX = '../tfdata/'

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_data():
  mat_files = os.listdir(MAT_DIR)
  mat_files = list(filter(lambda x: x.endswith('mat'), mat_files))
  label_data = pd.read_csv(
    LABEL_DIR, names=['mat_path', 'label'], delimiter=',')

  # 排除存在于label.txt但不在文件夹中存在的数据
  #     tmp = []
  #     for index, row in label_data.iterrows():
  #         if row['data_path'] in data_files:
  #             tmp.append(row)
  #     label_data = pd.DataFrame(tmp, columns=['data_path', 'label'])
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

  logger = Logger(filename=TF_DIR + 'cnn_epilepsy_sampling.log').get_logger()
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
    TF_DIR + "epilepsy_cnn_train.tfrecords",
    datas=train_X.apply(lambda x: MAT_DIR + x).tolist(),
    labels=train_Y.tolist(),
    logger=logger)
  # write test tfrecords
  write_to_tfrecord(
    TF_DIR + "epilepsy_cnn_test.tfrecords",
    datas=test_X.apply(lambda x: MAT_DIR + x).tolist(),
    labels=test_Y.tolist(),
    logger=logger)


if __name__ == '__main__':
  sampling()
