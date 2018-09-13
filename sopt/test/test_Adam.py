#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_Adam.py
@time: 2018/06/09 23:59
@description:
test for Adam
"""
import sys
sys.path.append("..")
from time import time
from sopt.util.gradients_config import gradients_config
from sopt.util.functions import *
from sopt.Optimizers.Gradients import Adam


class TestAdam:
    def __init__(self):
        self.func = quadratic100
        self.func_type = quadratic100_func_type
        self.variables_num = quadratic100_variables_num
        self.init_variables = None
        self.lr = 10
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.eps = gradients_config.eps
        self.epochs = 1000
        self.Adam = Adam(**self.__dict__)

    def test(self):
        start_time = time()
        self.Adam.run()
        print("Adam costs %.4f seconds!" %(time()-start_time))
        self.Adam.save_plot()
        self.Adam.show_result()



if __name__ == '__main__':
    TestAdam().test()
