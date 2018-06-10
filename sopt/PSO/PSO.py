#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: PSO.py
@time: 2018/06/07 23:07
@description:
paricle swarm optimizer(PSO)
"""
from warnings import warn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sopt.util.pso_config import *

class PSO:
    def __init__(self,
                 variables_num,
                 lower_bound,
                 upper_bound,
                 func,
                 func_type = basic_config.func_type_min,
                 c1 = basic_config.c1,
                 c2 = basic_config.c2,
                 generations = basic_config.generations,
                 population_size = basic_config.population_size,
                 vmax = basic_config.vmax,
                 vmin = basic_config.vmin,
                 w = 1,
                 w_start = 0.4,
                 w_end = 0.9,
                 w_method = pso_w_method.constant,
                 complex_constraints=None,
                 complex_constraints_method=complex_constraints_method.loop
                 ):
        '''
        :param variables_num: the numbers of variables
        :param lower_bound: numbers or array of numbers
        :param upper_bound: numbers or array of numbers
        :param func: the target function
        :param func_type: 'min' or 'max'
        :param c1: the individual learning factor
        :param c2: the social learning factor
        :param generations: the generations
        :param population_size: the PSO population size
        :param vmax: the max speed
        :param vmin: the min speed
        :param w: the w value
        :param w_start: the starting w value
        :param w_end: the ending w value
        :param w_method:'constant' or 'linear_decrease' or 'square1_decrease'
        or 'square2_decrease' or 'exp_decrease'
        '''
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
        self.func = func
        self.func_type = func_type
        self.c1 = c1
        self.c2 = c2
        self.generations = generations
        self.population_size = population_size
        self.vmax = vmax
        self.vmin = vmin
        self.w = w
        self.w_start = w_start
        self.w_end  = w_end
        self.w_method = w_method
        self.complex_constraints = complex_constraints
        self.complex_constraints_method = complex_constraints_method
        if self.complex_constraints_method != 'loop':
            warn("%s complex constraints method is currently not supported!Use 'loop' method instead!" % self.complex_constraints_method)
        self.best_population_target = 0
        self.best_population_point = None
        self.generations_best_targets = []
        self.generations_best_points = []


    def init_population(self):
        self.population = self.lower_bound + (self.upper_bound-self.lower_bound)*np.random.rand(self.population_size,self.variables_num)
        if self.complex_constraints is not None:
            for i in range(self.population_size):
                if self._check_constraints(self.population[i]) > 0:
                    self.population[i] = self._loop_random_modify(self.population[i])
        self.v = self.vmin + (self.vmax-self.vmin)*np.random.rand(self.population_size,self.variables_num)

    def _check_constraints(self,data):
        res = 0
        for constraint in self.complex_constraints:
            if constraint(data) > 0:
                res += constraint(data)
        return res

    def _loop_random_modify(self,data):
        while self._check_constraints(data) > 0:
            data = self.lower_bound + (self.upper_bound-self.lower_bound)*np.random.rand(self.variables_num)
        return data

    def _loop_v_modify(self,data,index,w_):
        '''
        loop to modify the data by the v[index] to
        satisfy the complex constraints
        :param data: the input data
        :param index: the index of v
        :return: the satisfied complex constraints data
        '''
        while self._check_constraints(data) > 0:
            self.v[index] = w_*self.v[index] + self.c1*np.random.rand(self.variables_num)*(self.best_individuals_points[index]-self.population[index]) + \
                self.c2*np.random.rand(self.variables_num)*(self.best_population_point-self.population[index])
            self.v[index][self.v[index] > self.vmax] = self.vmax
            self.v[index][self.v[index] < self.vmin] = self.vmin
            data += self.v[index]
            for i in range(self.variables_num):
                if data[i] > self.upper_bound[i]:
                    data[i] = self.upper_bound[i]
                if data[i] < self.lower_bound[i]:
                    data[i] = self.lower_bound[i]
        return data



    def run(self):
        self.init_population()
        self._calculate_best_population()
        self.best_individuals_points = deepcopy(self.population)
        for i in range(self.generations):
            if self.w_method == pso_w_method.constant:
                w_ = self.w
            elif self.w_method == pso_w_method.linear_decrease:
                w_ = self.w_end +(self.w_start-self.w_end)*(self.generations-i)/float(self.generations)
            elif self.w_method == pso_w_method.square1:
                w_ = self.w_start -(self.w_start-self.w_end)*(float(i)/self.generations)**2
            elif self.w_method == pso_w_method.square2:
                w_ = self.w_start + (self.w_start-self.w_end)*(2*float(i)/self.generations-(i/self.generations)**2)
            else:
                w_ = self.w_end*((self.w_start/self.w_end)**(1/(1+10*i/self.generations)))
            self.v = w_*self.v + self.c1*np.random.rand(self.population_size,self.variables_num)*(self.best_individuals_points-self.population) + \
                self.c2*np.random.rand(self.population_size,self.variables_num)*(self.best_population_point-self.population)
            self.v[self.v > self.vmax] = self.vmax
            self.v[self.v < self.vmin] = self.vmin
            self.population += self.v
            for k in range(self.population_size):
                for j in range(self.variables_num):
                    if self.population[k][j] > self.upper_bound[j]:
                        self.population[k][j] = self.upper_bound[j]
                    if self.population[k][j] < self.lower_bound[j]:
                        self.population[k][j] = self.lower_bound[j]
            # loop to process the complex constraints
            if self.complex_constraints is not None:
                for i in range(self.population_size):
                    if self._check_constraints(self.population[i]) > 0:
                        self.population[i] = self._loop_v_modify(self.population[i],i,w_)

            for k in range(self.population_size):
                # update the best_individuals points
                if (self.func(self.population[k]) < self.func(self.best_individuals_points[k]) and self.func_type == basic_config.func_type_min) or \
                    (self.func(self.population[k]) > self.func(self.best_individuals_points[k]) and self.func_type == basic_config.func_type_max):
                    self.best_individuals_points[k] = self.population[k]
                if (self.func(self.population[k]) < self.func(self.best_population_point) and self.func_type == basic_config.func_type_min) or \
                        (self.func(self.population[k]) > self.func(self.best_population_point) and self.func_type == basic_config.func_type_max):
                    self.best_population_point = self.population[k]
                    self.best_population_target = self.func(self.best_population_point)
            self.generations_best_targets.append(self.best_population_target)
            self.generations_best_points.append(self.best_population_point)

        if self.func_type == basic_config.func_type_min:
            self.global_best_index = np.argmin(self.generations_best_targets)
        else:
            self.global_best_index = np.argmax(self.generations_best_targets)







    def _calculate_best_population(self):
        targets_func = np.zeros(self.population_size)
        for i in range(self.population_size):
            targets_func[i] = self.func(self.population[i])
        if self.func_type == basic_config.func_type_min:
            self.best_population_target = np.min(targets_func)
            best_population_index = np.argmin(targets_func)
            self.best_population_point = self.population[best_population_index]
        else:
            self.best_population_target = np.max(targets_func)
            best_population_index = np.argmax(targets_func)
            self.best_population_point = self.population[best_population_index]

    def save_plot(self,save_name="PSO.png"):
        plt.plot(self.generations_best_targets,'r-')
        plt.xlabel("generations")
        plt.ylabel("best target function value")
        plt.title("PSO with %d generations" %self.generations)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "PSO config is:", "-" * 20)
        for k,v in self.__dict__.items():
            if k not in ['best_population_target','best_population_point','population',
                         'best_individuals_points','v','global_best_index',
                         'generations_best_targets','generations_best_points']:
                print("%s:%s" %(k,v))
        print("-" * 20, "PSO calculation result is:", "-" * 20)
        print("global best generation index/total generations: %s/%s" %(self.global_best_index,self.generations))
        print("global best point:", self.best_population_point)
        print("global best target:",self.best_population_target)



