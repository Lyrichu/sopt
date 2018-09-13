#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: sa_config.py
@time: 2018/06/09 14:47
@description:
config for SA
"""

class BASIC_CONFIG:
    def __init__(self):
        self.func_type_min = 'min'
        self.func_type_max = 'max'
        self.T_start = 100.
        self.T_end = 1e-6
        self.q = 0.9
        self.L = 100
        self.init_pos = None
        self.complex_constraints = None

class COMPLEX_CONSTRAINTS_METHOD:
    def __init__(self):
        self.loop = 'loop'

sa_config = BASIC_CONFIG()
complex_constraints_method = COMPLEX_CONSTRAINTS_METHOD()