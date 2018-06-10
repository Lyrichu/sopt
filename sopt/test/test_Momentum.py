#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_Momentum.py
@time: 2018/06/09 23:04
@description:
test for Momentum Optimizer
"""
import sys
sys.path.append("..")
from time import time
from sopt.util.gradients_config import *
from sopt.util.functions import *
from sopt.Optimizers.Gradients import Momentum


class TestMomentum:
    def __init__(self):
        self.func = quadratic50
        self.func_type = quadratic50_func_type
        self.variables_num = quadratic50_variables_num
        self.init_variables = None
        self.lr = 1e-3
        self.beta = 0.9
        self.epochs = 2000
        self.Momentum = Momentum(**self.__dict__)

    def test(self):
        start_time = time()
        self.Momentum.run()
        print("Momentum costs %.4f seconds!" %(time()-start_time))
        self.Momentum.save_plot()
        self.Momentum.show_result()



if __name__ == '__main__':
    TestMomentum().test()

