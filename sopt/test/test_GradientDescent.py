#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_GradientDescent.py
@time: 2018/06/09 22:20
@description:
test for Gradient Descent
"""
import sys
sys.path.append("..")
from time import time
from sopt.util.gradients_config import gradients_config
from sopt.util.functions import *
from sopt.Optimizers.Gradients import GradientDescent


class TestGradientDescent:
    def __init__(self):
        self.func = quadratic50
        self.func_type = quadratic50_func_type
        self.variables_num = quadratic50_variables_num
        self.init_variables = None
        self.lr = 1e-3
        self.epochs = 5000
        self.GradientDescent = GradientDescent(**self.__dict__)

    def test(self):
        start_time = time()
        self.GradientDescent.run()
        print("Gradient Descent costs %.4f seconds!" %(time()-start_time))
        self.GradientDescent.save_plot()
        self.GradientDescent.show_result()



if __name__ == '__main__':
    TestGradientDescent().test()


