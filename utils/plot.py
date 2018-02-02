# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/2/2 10:09.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import os

import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt


def read_lines(filename, read_type="r", encoding="utf-8"):
    tmp_fp = open(filename, read_type, encoding=encoding)
    lines = tmp_fp.readlines()
    tmp_fp.close()

    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
        lines[i] = lines[i].strip("\r")
    return lines


def plot(train_data, test_data, save_path, title='Neural Network', xlabel='Epoch', ylabel='Accuracy', font_size=6):
    plt.title(title, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)

    train_data_len = len(train_data)
    test_data_len = len(test_data)
    assert train_data_len == test_data_len
    x = list(range(1, len(train_data) + 1))

    plt.plot(x, train_data, 'b--', label='train {}'.format(str.lower(ylabel)), linewidth=1.0)
    plt.plot(x, test_data, 'r', label='test {}'.format(str.lower(ylabel)), linewidth=1.0)
    plt.legend(loc="lower right", fontsize=font_size)
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
