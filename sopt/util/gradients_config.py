#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: gd_config.py
@time: 2018/06/09 20:41
@description:
config for GD
"""

class BASIC_CONFIG:
    def __init__(self):
        self.lr = 1e-3
        self.adagrad_lr = 10
        self.rmsprop_lr = 10
        self.adam_lr = 10
        self.momentum_beta = 0.9
        self.rmsprop_beta = 0.9
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.9
        self.delta = 1e-8
        self.eps = 1e-8
        self.epochs = 100
        self.min_step = -1.
        self.max_step = 1.
        self.step_size = 0.001
        self.init_variables = None
        self.func_type_min = 'min'
        self.func_type_max = 'max'

gradients_config = BASIC_CONFIG()
