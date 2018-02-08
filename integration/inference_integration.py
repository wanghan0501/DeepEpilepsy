# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/23 17:22.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import numpy as np

from .cnn_v1 import inference_v1
from .cnn_v2 import inference_v2
from .cnn_v3 import inference_v3


class Inference:
  def __init__(self):
    v1 = inference_v1.Inference()
    v2 = inference_v2.Inference()
    v3 = inference_v3.Inference()
    self.model_list = [v1, v2, v3]

  def setup(self):
    count = 0
    for model in self.model_list:
      model.setup()
      count += 1
      print('No.{} model setup.'.format(count))

  def inference(self, inputs):
    cur_prediction = np.zeros([1, 2], np.float32)
    for model in self.model_list:
      cur_prediction += model.inference(inputs)
    cur_prediction /= len(self.model_list)
    cur_class = np.argmax(cur_prediction, axis=1)
    return cur_prediction, cur_class
