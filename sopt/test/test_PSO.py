#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_PSO.py
@time: 2018/06/08 23:31
@description:
test for PSO
"""
import sys
sys.path.append("..")
from time import time
from sopt.util.functions import *
from sopt.util.pso_config import *
from sopt.PSO.PSO import PSO
from sopt.util.constraints import *


class TestPSO:
    def __init__(self):
        self.func = quadratic50
        self.func_type = quadratic50_func_type
        self.variables_num = quadratic50_variables_num
        self.lower_bound = quadratic50_lower_bound
        self.upper_bound = quadratic50_upper_bound
        self.c1 = basic_config.c1
        self.c2 = basic_config.c2
        self.generations = 200
        self.population_size = 100
        self.vmax = 1
        self.vmin = -1
        self.w = 1
        self.w_start = 0.9
        self.w_end = 0.4
        self.w_method = pso_w_method.linear_decrease
        # self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints = None
        self.complex_constraints_method = complex_constraints_method.loop
        self.PSO = PSO(**self.__dict__)

    def test(self):
        start_time = time()
        self.PSO.run()
        print("PSO costs %.4f seconds!" %(time()-start_time))
        self.PSO.save_plot()
        self.PSO.show_result()


if __name__ == '__main__':
    TestPSO().test()
