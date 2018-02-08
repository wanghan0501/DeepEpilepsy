# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/28 10:09.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import scipy.io as scio


def read_mat(mat_path):
  dict = scio.loadmat(mat_path)
  for key in dict.keys():
    print(key)
    if not key.startswith('__'):
      return dict[key]
