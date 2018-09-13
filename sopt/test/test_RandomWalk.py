#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_RandomWalk.py
@time: 2018/06/07 16:17
@description:
test for RandomWalk
"""
import sys
sys.path.append("..")
from time import time
from sopt.util.functions import *
from sopt.util.constraints import *
from sopt.util.random_walk_config import *
from sopt.Optimizers.RandomWalk import RandomWalk

class TestRandomWalk:
    def __init__(self):
        self.func = Rosenbrock
        self.func_type = Rosenbrock_func_type
        self.variables_num = Rosenbrock_variables_num
        self.lower_bound = Rosenbrock_lower_bound
        self.upper_bound = Rosenbrock_upper_bound
        self.generations = 10
        self.init_step = 10
        self.eps = 1e-2
        self.vectors_num = 10
        self.init_pos = None
        # self.complex_constraints = [constraints1,constraints2,constraints3]
        # self.complex_constraints_method = complex_constraints_method.loop
        self.RandomWalk = RandomWalk(**self.__dict__)

    def test(self):
        start_time = time()
        self.RandomWalk.random_walk()
        print("random walk costs %.4f seconds!" %(time() - start_time))
        self.RandomWalk.save_plot()
        self.RandomWalk.show_result()


if __name__ == '__main__':
    TestRandomWalk().test()

