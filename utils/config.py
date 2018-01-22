# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/10 14:48.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""


class CNNConfig(object):
    def __init__(self,
                 model_name='Epilepsy_3D_CNN',
                 is_training=False,
                 max_epoch=200,
                 gpu_num=1,
                 lr=1,
                 num_classes=2,
                 image_shape=(61, 73, 61, 2),
                 batch_size=1,
                 plot_batch=25,
                 dropout_keep_prob=0.5,
                 train_data_path=None,
                 test_data_path=None,
                 train_data_length=None,
                 test_data_length=None,
                 logger_path=None,
                 save_model_path=None,
                 capacity=50,
                 num_threads=1,
                 min_after_dequeue=5,
                 ):
        self.model_name = model_name
        self.is_training = is_training
        self.max_epoch = max_epoch
        self.gpu_num = gpu_num
        self.lr = lr
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.plot_batch = plot_batch
        self.dropout_keep_prob = dropout_keep_prob
        self.train_data_path = train_data_path
        self.train_data_length = train_data_length
        self.test_data_path = test_data_path
        self.test_data_length = test_data_length
        self.logger_path = logger_path
        self.save_model_path = save_model_path
        self.capacity = capacity
        self.num_threads = num_threads
        self.min_after_dequeue = min_after_dequeue

    def __str__(self):
        str = '''
=====Config====
model_name = {}  
is_training = {}      
max_epoch = {}
gpu_num = {}
lr = {}
num_classes = {}
batch_size = {}
image_shape = {}
plot_batch = {}
dropout_keep_prob = {}
train_data_path = {}
train_data_length = {}
test_data_path = {}
test_data_length = {}
logger_path = {}
save_model_path = {}
=====Config====
'''.format(
            self.model_name,
            self.is_training,
            self.max_epoch,
            self.gpu_num,
            self.lr,
            self.num_classes,
            self.batch_size,
            self.image_shape,
            self.plot_batch,
            self.dropout_keep_prob,
            self.train_data_path,
            self.train_data_length,
            self.test_data_path,
            self.test_data_length,
            self.logger_path,
            self.save_model_path)
        return str


class RNNConfig(object):
    def __init__(self,
                 model_name='Epilepsy_3D_RNN',
                 is_training=False,
                 max_epoch=200,
                 num_gpu=1,
                 num_layers=2,
                 num_steps=190,
                 hidden_size=512,
                 num_classes=2,
                 lr=1,
                 image_shape=(61, 73, 61, 190),
                 batch_size=1,
                 plot_batch=25,
                 dropout_keep_prob=0.5,
                 train_data_path=None,
                 test_data_path=None,
                 train_data_length=None,
                 test_data_length=None,
                 logger_path=None,
                 save_model_path=None,
                 capacity=20,
                 num_threads=1,
                 min_after_dequeue=3
                 ):
        self.model_name = model_name
        self.is_training = is_training
        self.max_epoch = max_epoch
        self.num_gpu = num_gpu
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.plot_batch = plot_batch
        self.dropout_keep_prob = dropout_keep_prob
        self.train_data_path = train_data_path
        self.train_data_length = train_data_length
        self.test_data_path = test_data_path
        self.test_data_length = test_data_length
        self.logger_path = logger_path
        self.save_model_path = save_model_path
        self.capacity = capacity
        self.num_threads = num_threads
        self.min_after_dequeue = min_after_dequeue

    def __str__(self):
        str = '''
=====Config====
model_name = {}  
is_training = {}      
max_epoch = {}
num_gpu = {}
num_layers = {}
num_steps = {}
hidden_size = {}
num_classes = {}
lr = {}
batch_size = {}
image_shape = {}
plot_batch = {}
dropout_keep_prob = {}
train_data_path = {}
train_data_length = {}
test_data_path = {}
test_data_length = {}
logger_path = {}
save_model_path = {}
capacity = {}
num_threads = {}
min_after_dequeue = {}
=====Config====
'''.format(
            self.model_name,
            self.is_training,
            self.max_epoch,
            self.num_gpu,
            self.num_layers,
            self.num_steps,
            self.hidden_size,
            self.num_classes,
            self.lr,
            self.batch_size,
            self.image_shape,
            self.plot_batch,
            self.dropout_keep_prob,
            self.train_data_path,
            self.train_data_length,
            self.test_data_path,
            self.test_data_length,
            self.logger_path,
            self.save_model_path,
            self.capacity,
            self.num_threads,
            self.min_after_dequeue
        )
        return str
