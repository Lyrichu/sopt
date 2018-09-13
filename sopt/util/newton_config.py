#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: newton_config.py
@time: 2018/09/11 21:26
@description:
config for newton series method
"""

class BASIC_CONFIG:
    def __init__(self):
        self.delta = 1e-8
        self.eps = 1e-6
        self.epochs = 10
        self.init_variables = None
        self.func_type_min = 'min'
        self.func_type_max = 'max'
        self.max_step = 1.
        self.min_step = -1.
        self.step_size = 0.01

newton_config = BASIC_CONFIG()



