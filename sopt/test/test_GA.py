#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_GA.py
@time: 2018/06/06 16:45
@description:
test for GA
"""
from time import time
import sys
sys.path.append("..")
from sopt.GA.GA import GA
from sopt.util.functions import *
from sopt.util.ga_config import *
from sopt.util.constraints import *

class TestGA:
    def __init__(self):
        self.func = Rosenbrock
        self.func_type = Rosenbrock_func_type
        self.variables_num = Rosenbrock_variables_num
        self.lower_bound = Rosenbrock_lower_bound
        self.upper_bound = Rosenbrock_upper_bound
        self.cross_rate = 0.8
        self.mutation_rate = 0.1
        self.generations = 300
        self.population_size = 200
        self.binary_code_length = 20
        self.cross_rate_exp = 1
        self.mutation_rate_exp = 1
        self.code_type = code_type.real 
        self.cross_code = False
        self.select_method = select_method.proportion
        self.rank_select_probs = None
        self.tournament_num = 2
        self.cross_method = cross_method.uniform
        self.arithmetic_cross_alpha = 0.1
        self.arithmetic_cross_exp = 1
        self.mutation_method = mutation_method.uniform
        self.none_uniform_mutation_rate = 1
        self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints_method = complex_constraints_method.penalty
        self.complex_constraints_C = 1e8
        self.M = 1e8
        self.GA = GA(**self.__dict__)

    def test(self):
        start_time = time()
        self.GA.run()
        print("GA costs %.4f seconds!" % (time()-start_time))
        self.GA.save_plot()
        self.GA.show_result()



if __name__ == '__main__':
    TestGA().test()
