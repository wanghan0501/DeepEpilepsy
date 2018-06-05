# -*- coding: utf-8 -*-  

"""
This program supports python3

Created by Wang Han on 2018/1/10 14:46.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset.tfrecord import get_batch, get_shuffle_batch
from nets.model import Epilepsy3dInceptionResnetV2, Epilepsy3dInceptionV2, Epilepsy3dInceptionV3, Epilepsy3dInceptionV4, \
  Epilepsy3dShadow
from utils import config
from utils.log import Logger
from utils.plot import plot

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

cur_run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

conf = config.CNNConfig(
  model_name='shadow_3d',
  dropout_keep_prob=0.35,
  is_training=True,
  num_classes=3,
  optimizer=tf.train.AdamOptimizer,
  image_shape=(61, 73, 61, 2),
  lr=0.0005,
  batch_norm_decay=0.95,
  use_tensorboard=False,
  train_batch_size=8,
  test_batch_size=8,
  max_epoch=500,
  capacity=240,
  num_threads=4,
  min_after_dequeue=120,
  train_data_path='tfdata/cnn_tfdata_20180509/epilepsy_cnn_train.tfrecords',
  test_data_path='tfdata/cnn_tfdata_20180509/epilepsy_cnn_test.tfrecords', )

# get train batch data
train_batch_images, train_batch_labels, _ = get_shuffle_batch(conf.train_data_path, conf.train_batch_size, conf,
                                                              name='train_shuffle_batch', use_path=True)
# estimate 'train' progress batch data
estimate_train_images, estimate_train_labels, _ = get_batch(conf.train_data_path, conf.train_batch_size, conf,
                                                            name='estimate_train_batch', use_path=True)
# estimate 'test' progress batch data
estimate_test_images, estimate_test_labels, _ = get_batch(conf.test_data_path, conf.test_batch_size, conf,
                                                          name='estimate_test_batch', use_path=True)

# set train
conf.train_data_length = 343
conf.test_data_length = 88
if conf.model_name == 'shadow_3d':
  model = Epilepsy3dShadow(config=conf)
elif conf.model_name == 'inception_3d_v2':
  model = Epilepsy3dInceptionV2(config=conf)
elif conf.model_name == 'inception_3d_v3':
  model = Epilepsy3dInceptionV3(config=conf)
elif conf.model_name == 'inception_3d_v4':
  model = Epilepsy3dInceptionV4(config=conf)
elif conf.model_name == 'inception_resnet_3d_v2':
  model = Epilepsy3dInceptionResnetV2(config=conf)
else:
  model = Epilepsy3dInceptionV2(config=conf)

# create path to save model
conf.save_model_path = 'saved_models/{}_{}/'.format(conf.model_name, cur_run_timestamp)
if not os.path.exists(conf.save_model_path):
  os.mkdir(conf.save_model_path)

conf.logger_path = 'logs/{}_{}.log'.format(conf.model_name, cur_run_timestamp)
conf.tensorboard_path = 'summaries/{}_{}'.format(conf.model_name, cur_run_timestamp)
logger = Logger(filename=conf.logger_path).get_logger()
logger.info(str(conf))

epoch_train_acc, epoch_test_acc = [], []
epoch_train_sens, epoch_test_sens = [], []
epoch_train_spec, epoch_test_spec = [], []
with tf.Session(config=config_gpu) as sess:
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)

  if conf.use_tensorboard:
    writer = tf.summary.FileWriter(conf.tensorboard_path)
    writer.add_graph(sess.graph2)

  acc_saver = tf.train.Saver()
  f1_saver = tf.train.Saver()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord=coord)

  max_test_acc, max_test_acc_epoch = 0.0, 0
  for epoch_idx in range(conf.max_epoch):
    # train op
    for batch_idx in tqdm(range(int(conf.train_data_length / conf.train_batch_size))):
      cur_train_image, cur_train_label = sess.run([train_batch_images, train_batch_labels])
      _ = sess.run([model.train_op], feed_dict={model.inputs: cur_train_image,
                                                model.labels: cur_train_label})

    # estimate 'train' progress
    train_acc_array = []
    train_loss_array = []
    train_confusion_matrix = np.zeros([3, 3], dtype=int)
    for batch_idx in tqdm(range(int(conf.train_data_length / conf.train_batch_size))):
      cur_train_image, cur_train_label = sess.run([estimate_train_images, estimate_train_labels])
      cur_train_acc, cur_train_loss, cur_train_confusion_matrix = sess.run(
        [model.test_accuracy, model.test_loss, model.test_confusion_matrix],
        feed_dict={model.inputs: cur_train_image,
                   model.labels: cur_train_label})
      train_acc_array.append(cur_train_acc)
      train_loss_array.append(cur_train_loss)
      train_confusion_matrix += cur_train_confusion_matrix
    # [[TN, FP], [FN, TP]] = train_confusion_matrix
    [[T11, F12, F13], [F21, T22, F23], [F31, F32, T33]] = train_confusion_matrix
    # train_metrics = Confusion(train_confusion_matrix)
    logger.info(
      '[Train] Epoch:{}, T11:{}/{}, F12:{}, F13:{}, T22:{}/{}, F21:{}, F23：{}, T33:{}/{}, F31：{}, F32：{}, Loss:{:.6f}, Accuracy:{:.6f}'.format(
        epoch_idx,
        T11, T11 + F12 + F13, F12, F13, T22, F21 + T22 + F23, F21, F23, T33, F31 + F32 + T33, F31, F32,
        np.average(train_loss_array),
        np.average(train_acc_array)))
    epoch_train_acc.append(np.average(train_acc_array))

    # logger.info('[Train] Epoch:{}, TP:{}, TN:{}, FP:{}, FN:{}, Loss:{:.6f}, Accuracy:{:.6f}'.format(
    #   epoch_idx,
    #   TP, TN, FP, FN,
    #   np.average(train_loss_array),
    #   np.average(train_acc_array)))
    # epoch_train_acc.append(train_metrics.accuracy())
    # epoch_train_sens.append(train_metrics.sensibility(1))
    # epoch_train_spec.append(train_metrics.specificity(1))

    # estimate 'test' progress
    test_acc_array = []
    test_loss_array = []
    test_confusion_matrix = np.zeros([3, 3], dtype=int)
    for batch_idx in tqdm(range(int(conf.test_data_length / conf.test_batch_size))):
      cur_test_image, cur_test_label = sess.run([estimate_test_images, estimate_test_labels])
      cur_test_loss, cur_test_acc, cur_test_confusion_matrix = sess.run(
        [model.test_loss, model.test_accuracy, model.test_confusion_matrix],
        feed_dict={model.inputs: cur_test_image,
                   model.labels: cur_test_label})
      test_acc_array.append(cur_test_acc)
      test_loss_array.append(cur_test_loss)
      test_confusion_matrix += cur_test_confusion_matrix
    # [[TN, FP], [FN, TP]] = test_confusion_matrix
    # test_metrics = Confusion(test_confusion_matrix)
    [[T11, F12, F13], [F21, T22, F23], [F31, F32, T33]] = test_confusion_matrix
    # for the whole 'test' progress
    avg_test_acc = np.average(test_acc_array)
    avg_test_loss = np.average(test_loss_array)
    if max_test_acc < avg_test_acc:
      max_test_acc_epoch = epoch_idx
      max_test_acc = avg_test_acc
      model_save_path = conf.save_model_path + 'epoch_{}_acc_{:.6f}.ckpt'.format(
        epoch_idx, avg_test_acc)
      save_path = acc_saver.save(sess, model_save_path)
      print('Epoch {} model has been saved with test accuracy is {:.6f}'.format(epoch_idx, avg_test_acc))
    logger.info(
      '[Test] Epoch:{}, T11:{}/{}, F12:{}, F13:{}, T22:{}/{}, F21:{}, F23：{}, T33:{}/{}, F31：{}, F32：{}, Loss:{:.6f}, Accuracy:{:.6f}'.format(
        epoch_idx,
        T11, T11 + F12 + F13, F12, F13, T22, F21 + T22 + F23, F21, F23, T33, F31 + F32 + T33, F31, F32,
        avg_test_loss,
        avg_test_acc))
    # logger.info('[Test] Epoch:{}, TP:{}, TN:{}, FP:{}, FN:{}, Loss:{:.6f}, Accuracy:{:.6f}'.format(
    #   epoch_idx,
    #   TP, TN, FP, FN,
    #   avg_test_loss,
    #   avg_test_acc))
    logger.info('[Info] The max test accuracy is {:.6f} at epoch {}'.format(
      max_test_acc,
      max_test_acc_epoch))
    epoch_test_acc.append(avg_test_acc)
    # epoch_test_sens.append(test_metrics.sensibility(1))
    # epoch_test_spec.append(test_metrics.specificity(1))

  print('Model {} final epoch has been finished!'.format(conf.model_name))
  coord.request_stop()
  coord.join(threads)

# plot
print('Starting plotting.')
plot(epoch_train_acc, epoch_test_acc, conf.save_model_path + 'acc.png', title=conf.model_name, xlabel='epoch',
     ylabel='Accuracy')
# plot(epoch_train_sens, epoch_test_sens, conf.save_model_path + 'sens.png', title=conf.model_name, xlabel='epoch',
#      ylabel='Sensibility')
# plot(epoch_train_spec, epoch_test_spec, conf.save_model_path + 'spec.png', title=conf.model_name, xlabel='epoch',
#      ylabel='Specificity')
# print('Ending plotting.')
