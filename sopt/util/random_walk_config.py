#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: random_walk_config.py
@time: 2018/06/09 16:08
@description:
config for random walk
"""
class BASIC_CONFIG:
    def __init__(self):
        self.func_type_min = 'min'
        self.func_type_max = 'max'
        self.generations = 100
        self.init_step = 10.0
        self.eps = 1e-4
        self.vectors_num = 10
        self.init_pos = None
        self.complex_constraints = None

class COMPLEX_CONSTRAINTS_METHOD:
    def __init__(self):
        self.loop = 'loop'

random_walk_config = BASIC_CONFIG()
complex_constraints_method = COMPLEX_CONSTRAINTS_METHOD()
