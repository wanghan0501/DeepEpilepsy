# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/18 14:42.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os

import numpy as np
import scipy.io as scio
import tensorflow as tf

from utils.log import Logger


def read_mat(mat_path):
  dict = scio.loadmat(mat_path)
  for key in dict.keys():
    if not key.startswith('__'):
      return dict[key]


def avg_mat(mat, rate=2):
  res = []
  mat = mat / 1000
  mat = mat.T
  try:
    for row in range(mat.shape[0]):
      res.append(np.mean(np.reshape(mat[row, :], [mat.shape[1] // rate, rate]), axis=1))
    return np.array(res).T
  except ValueError:
    return np.array([])


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_tfrecord(file_name, datas, labels, logger=None, read_function=read_mat, img_shape=(95, 160),
                      use_avg=False):
  dir_name = os.path.dirname(file_name)
  if logger is None:
    logger = Logger(filename=dir_name + '/tfrecord_sampling.log', filemode='a').get_logger()
  if os.path.exists(file_name):
    logger.info('The data file {} exists !!'.format(file_name))
    return

  writer = tf.python_io.TFRecordWriter(file_name)

  count = 0
  for i in range(len(datas)):
    if not i % 100:
      logger.info('Write data: {:.5} %% '.format((float(i) / len(datas)) * 100))
    singal = read_function(datas[i][0])
    coefficient = read_function(datas[i][1])
    coefficient = coefficient[:, :, np.newaxis]
    if use_avg:
      singal = avg_mat(singal)

    if singal.shape != img_shape:
      logger.info('File {} shape {} error!'.format(datas[i][0], singal.shape))
      count += 1
      continue

    if singal is None:
      continue

    if coefficient.shape != (160, 160, 1):
      logger.info('File {} shape {} error!'.format(datas[i][1], coefficient.shape))
      count += 1
      continue

    singal = singal.astype(np.float32)
    coefficient = coefficient.astype(np.float32)
    label = np.int(labels[i])
    feature = {'label': _int64_feature(label),
               'image': _bytes_feature(singal.tobytes()),
               'coefficient': _bytes_feature(coefficient.tobytes()),
               'path': _bytes_feature(datas[i][0].encode('utf8'))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
  writer.close()
  logger.info('Data file {} finished writing! Total record number is {}'.format(file_name, len(datas) - count))


def read_from_tfrecord(filename_queue, img_shape=(95, 160), name='read_tfrecord'):
  with tf.name_scope(name=name):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={'label': tf.FixedLenFeature([], tf.int64),
                                                 'image': tf.FixedLenFeature([], tf.string),
                                                 'coefficient': tf.FixedLenFeature([], tf.string),
                                                 'path': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image'], tf.float32, name='image')
    image = tf.reshape(image, img_shape, name='img_reshape')
    coefficient = tf.decode_raw(features['coefficient'], tf.float32, name='coefficient')
    coefficient = tf.reshape(coefficient, (160, 160, 1), name='coeff_reshape')
    label = tf.cast(features['label'], dtype=tf.int64, name='label')
    path = features['path']
    return image, coefficient, label, path


def get_shuffle_batch(filename, batch_size, model_config, name='shuffle_batch'):
  with tf.name_scope(name=name):
    queue = tf.train.string_input_producer([filename])

    cur_images, cur_coefficients, cur_labels, cur_paths = read_from_tfrecord(queue, model_config.image_shape)
    batch_images, batch_coefficients, batch_labels, batch_paths = tf.train.shuffle_batch(
      [cur_images, cur_coefficients, cur_labels, cur_paths],
      batch_size=batch_size,
      capacity=model_config.capacity,
      num_threads=model_config.num_threads,
      min_after_dequeue=model_config.min_after_dequeue)
    return batch_images, batch_coefficients, batch_labels, batch_paths


def get_batch(filename, batch_size, model_config, name='batch'):
  with tf.name_scope(name=name):
    queue = tf.train.string_input_producer([filename])
    cur_images, cur_coefficients, cur_labels, cur_paths = read_from_tfrecord(queue, model_config.image_shape)
    batch_images, batch_coefficients, batch_labels, batch_paths = tf.train.batch(
      [cur_images, cur_coefficients, cur_labels, cur_paths],
      batch_size=batch_size,
      capacity=model_config.capacity,
      num_threads=1)
    return batch_images, batch_coefficients, batch_labels, batch_paths
