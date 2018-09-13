#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: config.py
@time: 2018/06/07 13:26
@description:
basic config for GA
"""
class BASIC_CONFIG:
    def __init__(self):
        self.cross_rate = 0.7
        self.mutation_rate = 0.1
        self.population_size = 100
        self.generations = 200
        self.binary_code_length = 10
        self.func_type_min = 'min'
        self.func_type_max = 'max'
        self.cross_rate_exp = 1
        self.mutation_rate_exp = 1
        self.tournament_num = 2
        self.arithmetic_cross_alpha = 0.1
        self.arithmetic_cross_exp = 1
        self.none_uniform_mutation_rate = 1
        self.complex_constraints_C = 1e6
        self.M = 1e8
        self.eps = 1e4


class CODE_TYPE:
    def __init__(self):
        self.binary = 'binary'
        self.gray = 'gray'
        self.real = 'real'

class SELECT_METHOD:
    def __init__(self):
        self.proportion = 'proportion'
        self.keep_best = 'keep_best'
        self.determinate_sampling = 'determinate_sampling'
        self.rssr = 'rssr'
        self.rank = 'rank'
        self.stochastic_tournament = 'stochastic_tournament'

class CROSS_METHOD:
    def __init__(self):
        self.one_point = 'one_point'
        self.two_point = 'two_point'
        self.uniform = 'uniform'
        self.arithmetic = 'arithmetic'


class MUTATION_METHOD:
    def __init__(self):
        self.simple = 'simple'
        self.uniform = 'uniform'
        self.boundary = 'boundary'
        self.none_uniform = 'none_uniform'
        self.gaussian = 'gaussian'

class COMPLEX_CONSTRAINTS_METHOD:
    def __init__(self):
        self.penalty = 'penalty'

ga_config = BASIC_CONFIG()
code_type = CODE_TYPE()
select_method = SELECT_METHOD()
cross_method = CROSS_METHOD()
mutation_method = MUTATION_METHOD()
complex_constraints_method = COMPLEX_CONSTRAINTS_METHOD()






