#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_DFP.py
@time: 2018/09/11 22:03
@description:
test for newton dfp
"""

import sys
sys.path.append("..")
from time import time
from sopt.util.newton_config import newton_config
from sopt.util.functions import *
from sopt.Optimizers.Newton import DFP


class TestDFP:
    def __init__(self):
        self.func = quadratic100
        self.func_type = quadratic100_func_type
        self.variables_num = quadratic100_variables_num
        self.init_variables = None
        self.min_step = newton_config.min_step
        self.max_step = newton_config.max_step
        self.step_size = newton_config.step_size
        self.eps = newton_config.eps
        self.epochs = newton_config.epochs
        self.DFP = DFP(**self.__dict__)

    def test(self):
        start_time = time()
        self.DFP.run()
        print("dfp costs %.4f seconds!" %(time()-start_time))
        self.DFP.save_plot()
        self.DFP.show_result()



if __name__ == '__main__':
    TestDFP().test()


