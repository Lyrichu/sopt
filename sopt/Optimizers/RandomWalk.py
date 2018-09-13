#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: RandomWalk.py
@time: 2018/06/07 15:41
@description:
random walk optimizers for non constrainted optimization problem
"""
from warnings import warn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sopt.util.random_walk_config import *
import numpy as np

class RandomWalk:
    def __init__(self,
                 variables_num,
                 lower_bound,
                 upper_bound,
                 func,
                 func_type = random_walk_config.func_type_min,
                 generations = random_walk_config.generations,
                 init_step = random_walk_config.init_step,
                 eps = random_walk_config.eps,
                 vectors_num = random_walk_config.vectors_num,
                 init_pos=random_walk_config.init_pos,
                 complex_constraints = random_walk_config.complex_constraints,
                 complex_constraints_method = complex_constraints_method.loop
    ):
        '''
        :param variables_num: numbers of variables
        :param lower_bound: int of array of lower bound for each variable
        :param upper_bound: int of array of upper bound for each variable
        :param func:target function
        :param generations: how many generations you want to run
        :param init_step:the initial step
        :param eps: the stop epsilon value
        :param vectors_num: numbers of vectors every time generate
        :param init_pos: the initial position
        :param complex_constraints: the complex constraints,default is None
        :param complex_constraints_method:currentlly only support 'loop' method 
        '''
        self.generations = generations
        self.init_step = init_step
        self.eps = eps
        self.variables_num = variables_num
        self.lower_bound = lower_bound
        if isinstance(self.lower_bound,(int,float)):
            self.lower_bound = np.array([self.lower_bound]*self.variables_num)
        assert len(self.lower_bound) == self.variables_num and type(self.lower_bound) is np.ndarray,\
                "lower bound should be an array of size %d of int/float!" % self.variables_num
        self.upper_bound = upper_bound
        if isinstance(self.upper_bound,(int,float)):
            self.upper_bound = np.array([self.upper_bound]*self.variables_num)
        assert len(self.upper_bound) == self.variables_num and type(self.upper_bound) is np.ndarray, \
            "upper bound should be an array of size %d of int/float!" % self.variables_num
        self.init_pos = init_pos
        if self.init_pos is None:
            self.init_pos = self.lower_bound + (self.upper_bound-self.lower_bound)*np.random.rand(self.variables_num)
        self.complex_constraints = complex_constraints
        self.complex_constraints_method = complex_constraints_method
        if self.complex_constraints_method != 'loop':
            warn("%s complex constrains method is currentlly not supported!Use 'loop' instead!" % self.complex_constraints_method)
        self.func = func
        self.func_type = func_type
        self.walk_nums = 0
        self.generations_nums = 0
        self.vectors_num = vectors_num
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0
        self.steps_best_targets = []
        self.steps_best_points = []
        self.generations_best_targets = []
        self.generations_best_points = []

    def _check_constraints(self,data):
        if self.complex_constraints is None:
            return 0
        res = 0
        for constraint in self.complex_constraints:
            if constraint(data) > 0:
                res += constraint(data)
        return res

    def _loop_check_constraints(self,data,step):
        while self._check_constraints(data) > 0:
            # generate random vectors
            v = np.random.uniform(low=-1., high=1., size=self.variables_num)
            # standarize the vectors
            v1 = v / np.sqrt(np.sum(v ** 2))
            data += step*v1
            for i in range(self.variables_num):
                if data[i] < self.lower_bound[i]:
                    data[i] = self.lower_bound[i]
                if data[i] > self.upper_bound[i]:
                    data[i] = self.upper_bound[i]
        return data


    def random_walk(self,show_info = True):
        '''
        basic random walk
        :return: the found best solution
        '''
        self.generations_nums = 0
        self.walk_nums = 0
        self.steps_best_targets = []
        self.steps_best_points = []
        self.generations_best_targets = []
        self.generations_best_points = []
        # start random walk
        x = self.init_pos
        step = self.init_step
        while step > self.eps:
            k = 1  # initilize the counter
            while k < self.generations:
                # generate random vectors
                v = np.random.uniform(low=-1.,high=1.,size=self.variables_num)
                # standarize the vectors
                v1 = v/np.sqrt(np.sum(v**2))
                x1 = x + step*v1
                if self.complex_constraints is None:
                    for i in range(self.variables_num):
                        if x1[i] < self.lower_bound[i]:
                            x1[i] = self.lower_bound[i]
                        if x1[i] > self.upper_bound[i]:
                            x1[i] = self.upper_bound[i]
                else:
                    x1 = self._loop_check_constraints(x1,step)
                # if we find a better solution
                if (self.func_type == random_walk_config.func_type_min and self.func(x1) < self.func(x)) or (self.func_type == random_walk_config.func_type_max and self.func(x1) > self.func(x)):
                    k = 1
                    x = x1
                else:
                    k += 1
                self.generations_nums += 1
                self.generations_best_points.append(x)
                self.generations_best_targets.append(self.func(x))
            step /= 2.
            self.steps_best_points.append(x)
            self.steps_best_targets.append(self.func(x))
            self.walk_nums += 1
            if show_info:
                print("Finish %d random walk!" % self.walk_nums)
        if self.func_type == random_walk_config.func_type_min:
            self.global_best_index = np.argmin(self.generations_best_targets)
            self.global_best_target = self.generations_best_targets[int(self.global_best_index)]
            self.global_best_point = self.generations_best_points[int(self.global_best_index)]
        else:
            self.global_best_index = np.argmax(self.generations_best_targets)
            self.global_best_target = self.generations_best_targets[int(self.global_best_index)]
            self.global_best_point = self.generations_best_points[int(self.global_best_index)]
        return x

    def improved_random_walk(self,show_info = True):
        '''
        improved random walk
        :return: the found best solution
        '''
        self.generations_nums = 0
        self.walk_nums = 0
        self.steps_best_targets = []
        self.steps_best_points = []
        self.generations_best_targets = []
        self.generations_best_points = []
        # start random walk
        x = self.init_pos
        step = self.init_step
        while step > self.eps:
            k = 1
            while k < self.generations:
                # generate n vectors
                x1_list = []
                for i in range(self.vectors_num):
                    v = np.random.uniform(-1.,1.,size=self.variables_num)
                    # v1 is the standarized vectors
                    v1 = v/np.sqrt(np.sum(v**2))
                    x1 = x + step*v1
                    if self.complex_constraints is None:
                        for i in range(self.variables_num):
                            if x1[i] < self.lower_bound[i]:
                                x1[i] = self.lower_bound[i]
                            if x1[i] > self.upper_bound[i]:
                                x1[i] = self.upper_bound[i]
                    else:
                        x1 = self._loop_check_constraints(x1,step)
                    x1_list.append(x1)
                f1_list = [self.func(x1) for x1 in x1_list]
                f1_min = min(f1_list)
                f1_index = f1_list.index(f1_min)
                x11 = x1_list[f1_index]
                if (self.func_type == random_walk_config.func_type_min and self.func(x1) < self.func(x)) or (self.func_type == random_walk_config.func_type_max and self.func(x1) > self.func(x)):
                    k = 1
                    x = x11
                else:
                    k += 1
                self.generations_nums += 1
                self.generations_best_points.append(x)
                self.generations_best_targets.append(self.func(x))
            self.steps_best_points.append(x)
            self.steps_best_targets.append(self.func(x))
            step /= 2.
            self.walk_nums += 1
            if show_info:
                print("Finish %d random walk!" % self.walk_nums)
        if self.func_type == random_walk_config.func_type_min:
            self.global_best_index = np.argmin(self.generations_best_targets)
            self.global_best_target = self.generations_best_targets[int(self.global_best_index)]
            self.global_best_point = self.generations_best_points[int(self.global_best_index)]
        else:
            self.global_best_index = np.argmax(self.generations_best_targets)
            self.global_best_target = self.generations_best_targets[int(self.global_best_index)]
            self.global_best_point = self.generations_best_points[int(self.global_best_index)]
        return x

    def save_plot(self,save_name = "random_walk.png"):
        plt.subplot(121)
        plt.plot(self.generations_best_targets,'r-')
        plt.xlabel('generations')
        plt.ylabel("best target function value")
        plt.title("random walk with %d generations" % self.generations_nums)
        plt.subplot(122)
        plt.plot(self.steps_best_targets, 'b-')
        plt.xlabel('steps')
        plt.ylabel("best target function value")
        plt.title("random walk with %d steps" % self.walk_nums)
        plt.savefig(save_name)

    def show_result(self):
        print("-"*20,"random walk config is:","-"*20)
        for k, v in self.__dict__.items():
            if k not in ['steps_best_targets','steps_best_points','init_pos',
                         'global_best_index','global_best_point','global_best_target',
                         'generations_best_targets','generations_best_points']:
                print("%s:%s" %(k,v))

        print("-"*20,"random walk caculation result is:","-"*20)
        print("global best generation index/total generations:%s/%s" %(self.global_best_index,self.generations_nums))
        print("global best point is:",self.global_best_point)
        print("global best target is:",self.global_best_target)



