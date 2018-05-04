# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/18 14:42.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os

import nipy
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


def read_nii(nii_path):
  nii_img = nipy.load_image(nii_path)
  return np.array(nii_img.get_data())


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_tfrecord(file_name, datas, labels, logger=None, read_function=read_mat, img_shape=(61, 73, 61, 2),
                      img_type=np.float32, use_avg=False):
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
    img = read_function(datas[i])
    if use_avg:
      img = avg_mat(img)

    if img.shape != img_shape:
      logger.info('File {} shape {} error!'.format(datas[i], img.shape))
      count += 1
      continue

    if img is None:
      continue

    image = img.astype(img_type)
    label = np.int(labels[i])
    feature = {'label': _int64_feature(label),
               'image': _bytes_feature(image.tobytes()),
               'path': _bytes_feature(datas[i].encode('utf8'))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
  writer.close()
  logger.info('Data file {} finished writing! Total record number is {}'.format(file_name, len(datas) - count))


def read_from_tfrecord(filename_queue, img_shape=(61, 73, 61, 2), img_type=tf.float32, use_path=False,
                       name='read_tfrecord', ):
  with tf.name_scope(name=name):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    if use_path:
      features = tf.parse_single_example(serialized_example,
                                         features={'label': tf.FixedLenFeature([], tf.int64),
                                                   'image': tf.FixedLenFeature([], tf.string),
                                                   'path': tf.FixedLenFeature([], tf.string)})
      image = tf.decode_raw(features['image'], img_type)
      image = tf.reshape(image, img_shape)
      label = tf.cast(features['label'], dtype=tf.int64, name='label')
      path = features['path']
      return image, label, path
    else:
      features = tf.parse_single_example(serialized_example,
                                         features={'label': tf.FixedLenFeature([], tf.int64),
                                                   'image': tf.FixedLenFeature([], tf.string)})
      image = tf.decode_raw(features['image'], img_type)
      image = tf.reshape(image, img_shape)
      label = tf.cast(features['label'], dtype=tf.int64, name='label')
      return image, label


def get_shuffle_batch(filename, batch_size, model_config, name='shuffle_batch', use_path=False):
  with tf.name_scope(name=name):
    queue = tf.train.string_input_producer([filename])
    if use_path:
      cur_images, cur_labels, cur_paths = read_from_tfrecord(queue, model_config.image_shape, use_path=True)
      batch_images, batch_labels, batch_paths = tf.train.shuffle_batch(
        [cur_images, cur_labels, cur_paths],
        batch_size=batch_size,
        capacity=model_config.capacity,
        num_threads=model_config.num_threads,
        min_after_dequeue=model_config.min_after_dequeue)
      return batch_images, batch_labels, batch_paths
    else:
      cur_images, cur_labels = read_from_tfrecord(queue, model_config.image_shape)
      batch_images, batch_labels = tf.train.shuffle_batch(
        [cur_images, cur_labels],
        batch_size=batch_size,
        capacity=model_config.capacity,
        num_threads=model_config.num_threads,
        min_after_dequeue=model_config.min_after_dequeue)
      return batch_images, batch_labels


def get_batch(filename, batch_size, model_config, name='batch', use_path=False):
  with tf.name_scope(name=name):
    queue = tf.train.string_input_producer([filename])
    if use_path:
      cur_images, cur_labels, cur_paths = read_from_tfrecord(queue, model_config.image_shape, use_path=True)
      batch_images, batch_labels, batch_paths = tf.train.batch(
        [cur_images, cur_labels, cur_paths],
        batch_size=batch_size,
        capacity=model_config.capacity,
        num_threads=1,
        allow_smaller_final_batch=True)
      return batch_images, batch_labels, batch_paths
    else:
      cur_images, cur_labels = read_from_tfrecord(queue, model_config.image_shape)
      batch_images, batch_labels = tf.train.batch(
        [cur_images, cur_labels],
        batch_size=batch_size,
        capacity=model_config.capacity,
        num_threads=1,
        allow_smaller_final_batch=True)
    return batch_images, batch_labels
