#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_AdaGrad.py
@time: 2018/06/09 23:26
@description:
test for AdaGrad
"""
import sys
sys.path.append("..")
from time import time
from sopt.util.gradients_config import gradients_config
from sopt.util.functions import *
from sopt.Optimizers.Gradients import AdaGrad


class TestAdaGrad:
    def __init__(self):
        self.func = quadratic50
        self.func_type = quadratic50_func_type
        self.variables_num = quadratic50_variables_num
        self.init_variables = None
        self.lr = 10
        self.eps = gradients_config.eps
        self.epochs = 2000
        self.AdaGrad = AdaGrad(**self.__dict__)

    def test(self):
        start_time = time()
        self.AdaGrad.run()
        print("Adagrad costs %.4f seconds!" %(time()-start_time))
        self.AdaGrad.save_plot()
        self.AdaGrad.show_result()



if __name__ == '__main__':
    TestAdaGrad().test()
