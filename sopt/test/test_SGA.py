#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_SGA.py
@time: 2018/06/06 08:00
@description:
test for SGA
"""
from time import time
import sys
sys.path.append("..")
from sopt.SGA.SGA import SGA
from sopt.util.functions import *

class TestSGA:
    def __init__(self):
        self.lower_bound = shubert2_lower_bound
        self.upper_bound = shubert2_upper_bound
        self.variables_num = shubert2_variables_num
        self.func_type = shubert2_func_type
        self.cross_rate = 0.8
        self.mutation_rate = 0.1
        self.generations = 200
        self.population_size = 100
        self.binary_code_length = 20
        self.func = shubert2
        self.SGA = SGA(**self.__dict__)

    def test(self):
        start_time = time()
        self.SGA.run()
        print("SGA costs %.4f seconds!" % (time()-start_time))
        self.SGA.save_plot()
        self.SGA.show_result()



if __name__ == '__main__':
    TestSGA().test()
