#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: pso_config.py
@time: 2018/06/07 23:11
@description:
config settings for PSO
"""

class BASIC_CONFIG:
    def __init__(self):
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.generations = 100
        self.population_size = 50
        self.vmax = 1
        self.vmin = -1
        self.func_type_min = 'min'
        self.func_type_max = 'max'
        self.w = 1
        self.w_start = 0.9
        self.w_end = 0.4
        self.complex_constraints = None

class PSO_W_METHOD:
    def __init__(self):
        self.constant = 'constant'
        self.linear_decrease = 'linear_decrease'
        self.square1 = 'square1_decrease'
        self.square2 = 'square2_decrease'
        self.exp = 'exp_decrease'

class COMPLEX_CONSTRAINTS_METHOD:
    def __init__(self):
        # self.penalty = 'penalty'
        self.loop = 'loop'

pso_config = BASIC_CONFIG()
pso_w_method = PSO_W_METHOD()
complex_constraints_method = COMPLEX_CONSTRAINTS_METHOD()


