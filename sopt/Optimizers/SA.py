#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: SA.py
@time: 2018/06/07 17:27
@description:
simple simulated annealing algorithm(SA)
"""
from warnings import warn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sopt.util.sa_config import *

class SA:
    def __init__(self,
                 func,
                 variables_num,
                 lower_bound,
                 upper_bound,
                 func_type = sa_config.func_type_min,
                 T_start=sa_config.T_start,
                 T_end = sa_config.T_end,
                 q = sa_config.q,
                 L = sa_config.L,
                 init_pos = None,
                 complex_constraints = sa_config.complex_constraints,
                 complex_constraints_method = complex_constraints_method.loop
                 ):
        '''
        :param func: target function
        :param variables_num: the numbers of variables
        :param lower_bound: int or array of int/float
        :param upper_bound: int or array of int/float
        :param func_type: 'min' or 'max'
        :param T_start: the starting temperature of SA
        :param T_end: the endding temperature of SA
        :param q: the SA factor
        :param L: the length of SA link
        :param init_pos: the initial position
        :param complex_constraints: the complex constraints,default is None
        :param complex_constraints_method:currentlly only support 'loop' method
        '''
        self.func = func
        self.func_type = func_type
        self.variables_num = variables_num
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if isinstance(self.lower_bound,(int,float)):
            self.lower_bound = np.array([self.lower_bound]*self.variables_num)
        assert len(self.lower_bound) == self.variables_num and type(self.lower_bound) is np.ndarray,\
                "lower bound should be an array of size %d of int/float!" % self.variables_num
        self.upper_bound = upper_bound
        if isinstance(self.upper_bound,(int,float)):
            self.upper_bound = np.array([self.upper_bound]*self.variables_num)
        assert len(self.upper_bound) == self.variables_num and type(self.upper_bound) is np.ndarray, \
            "upper bound should be an array of size %d of int/float!" % self.variables_num
        self.T_start = T_start
        self.T_end = T_end
        self.q = q
        self.L = L
        self.init_pos = init_pos
        self.complex_constraints = complex_constraints
        self.complex_constraints_method = complex_constraints_method
        if self.complex_constraints_method != 'loop':
            warn("%s complex constrains method is currentlly not supported!Use 'loop' instead!" % self.complex_constraints_method)
        if self.init_pos is None:
            self.init_pos = self.lower_bound + (self.upper_bound-self.lower_bound)*np.random.rand(self.variables_num)
        self.steps = 0
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0
        self.generations_best_targets = []
        self.generations_best_points = []

    def random_disturb(self):
        '''
        random disturb function to generate new solutions
        :return:
        '''
        init_pos_copy = deepcopy(self.init_pos)
        if np.random.random() > 0.5 :
            if self.complex_constraints is None:
                init_pos_copy = self.lower_bound + np.random.rand(self.variables_num)*(self.upper_bound-self.lower_bound)
            else:
                init_pos_copy = self.lower_bound + np.random.rand(self.variables_num) * (self.upper_bound - self.lower_bound)
                while self._check_constraints(init_pos_copy) > 0:
                    init_pos_copy = self.lower_bound + np.random.rand(self.variables_num) * (self.upper_bound - self.lower_bound)
        return init_pos_copy

    def _check_constraints(self,data):
        if self.complex_constraints is None:
            return 0
        res = 0
        for constraint in self.complex_constraints:
            if constraint(data) > 0:
                res += constraint(data)
        return res


    def run(self):
        '''
        run SA
        :return:
        '''
        T = self.T_start
        while(T > self.T_end):
            for i in range(self.L):
                init_pos_disturb = self.random_disturb()
                delta = self.func(init_pos_disturb)-self.func(self.init_pos)
                if (self.func_type == sa_config.func_type_min and delta<0) \
                        or (self.func_type == sa_config.func_type_max and delta > 0):
                    self.init_pos = init_pos_disturb
                else:
                    if self.func_type == sa_config.func_type_min:
                        sign = 1
                    else:
                        sign = -1
                    rnd = np.exp(-sign*delta/T)
                    if np.random.random() < rnd:
                        # use a small probability to accept the worse solution
                        self.init_pos = init_pos_disturb
                self.steps += 1
                self.generations_best_targets.append(self.func(self.init_pos))
                self.generations_best_points.append(self.init_pos)
            T *= self.q
        if self.func_type == sa_config.func_type_min:
            self.global_best_index = np.argmin(self.generations_best_targets)
            self.global_best_target = np.min(self.generations_best_targets)
            self.global_best_point = self.generations_best_points[int(np.argmin(np.array(self.generations_best_targets)))]
        else:
            self.global_best_index = np.argmax(self.generations_best_targets)
            self.global_best_target = np.max(self.generations_best_targets)
            self.global_best_point = self.generations_best_points[int(np.argmax(np.array(self.generations_best_targets)))]


    def save_plot(self,save_name = "SA.png"):
        plt.plot(self.generations_best_targets,'r-')
        plt.xlabel("steps")
        plt.ylabel("best target function value")
        plt.title("SA %d steps simulation" % self.steps)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "SA config is:", "-" * 20)
        for k,v in self.__dict__.items():
            if k not in ['global_best_target','global_best_point','global_best_index',
                         'generations_best_targets','generations_best_points']:
                print("%s:%s" %(k,v))
        print("-" * 20, "SA calculation result is:", "-" * 20)
        print("global best generation index/total generations:%s/%s" % (self.global_best_index,self.steps))
        print("global best point:", self.global_best_point)
        print("global best target:",self.global_best_target)



