# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/23 16:41.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import tensorflow as tf

from .model_v3 import Epilepsy3dCnn_V3
from utils.config import CNNConfig


class Inference(object):
    def __init__(self, model_path='integration/cnn_v3/optimization_model/epoch_23_acc_1.000000_f1_1.000000.ckpt', ):
        self.model_path = model_path
        # get model config
        model_config = CNNConfig(model_name='cnn_v3',
                                 image_shape=(61, 73, 61, 2),
                                 batch_size=1,
                                 is_training=False)
        self.g = tf.Graph()
        with self.g.as_default():
            self.model = Epilepsy3dCnn_V3(model_config)
            self.sess = tf.Session()

    def setup(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)

    def inference(self, inputs):
        predictions = self.sess.run([self.model.test_predictions],
                                    feed_dict={self.model.inputs: inputs})
        predictions = predictions[0]
        return predictions
