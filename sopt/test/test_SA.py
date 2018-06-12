#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_SA.py
@time: 2018/06/07 19:03
@description:
test for SA
"""
from time import time
import sys
sys.path.append("..")
from sopt.util.functions import *
from sopt.Optimizers.SA import SA
from sopt.util.sa_config import *
from sopt.util.constraints import *

class TestSA:
    def __init__(self):
        self.func = Rosenbrock
        self.func_type = Rosenbrock_func_type
        self.variables_num = Rosenbrock_variables_num
        self.lower_bound = Rosenbrock_lower_bound
        self.upper_bound = Rosenbrock_upper_bound
        self.T_start = 100
        self.T_end = 1e-6
        self.q = 0.9
        self.L = 100
        self.init_pos = None
        #self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints_method = complex_constraints_method.loop
        self.SA = SA(**self.__dict__)


    def test(self):
        start_time = time()
        self.SA.run()
        print("SA costs %.4f seconds!" %(time()-start_time))
        self.SA.save_plot()
        self.SA.show_result()


if __name__ == '__main__':
    TestSA().test()




