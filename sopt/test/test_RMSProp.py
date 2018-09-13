#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_RMSProp.py
@time: 2018/06/09 23:41
@description:
test for RMSProp
"""

import sys
sys.path.append("..")
from time import time
from sopt.util.gradients_config import gradients_config
from sopt.util.functions import *
from sopt.Optimizers.Gradients import RMSProp


class TestRMSProp:
    def __init__(self):
        self.func = quadratic100
        self.func_type = quadratic100_func_type
        self.variables_num = quadratic100_variables_num
        self.init_variables = None
        self.lr = 10
        self.beta = 0.9
        self.eps = gradients_config.eps
        self.epochs = 1000
        self.RMSProp = RMSProp(**self.__dict__)

    def test(self):
        start_time = time()
        self.RMSProp.run()
        print("RMSProp costs %.4f seconds!" %(time()-start_time))
        self.RMSProp.save_plot()
        self.RMSProp.show_result()



if __name__ == '__main__':
    TestRMSProp().test()
