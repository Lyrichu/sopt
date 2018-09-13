#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: SGA.py
@time: 2018/06/05 21:03
@description:
simple Genetic Algorithm(SGA)
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sopt.util.ga_config import *

class SGA:
    def __init__(self,lower_bound,upper_bound,variables_num,func,
                 cross_rate = ga_config.cross_rate,
                 mutation_rate = ga_config.mutation_rate,
                 population_size = ga_config.population_size,
                 generations = ga_config.generations,
                 binary_code_length= ga_config.binary_code_length,
                 func_type = ga_config.func_type_min
                 ):
        '''
        :param lower_bound: the lower bound of variables,real number or list of real numbers
        :param upper_boound: the upper bound of variables,real number or list of real numbers
        :param variables_num: the number of variables
        :param func: the target function
        :param cross_rate: GA cross rate
        :param mutation_rate: GA mutation rate
        :param population_size: the size of GA population
        :param generations: the GA generations count
        :param binary_code_length: the binary code length to represent a real number
        :param func_type:'min' means to evaluate the minimum target function;'max'
         means to evaluate the maximum target function
        '''
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.variables_num = variables_num
        if isinstance(self.lower_bound,(int,float)):
            self.lower_bound = [self.lower_bound]*self.variables_num
        if isinstance(self.upper_bound,(int,float)):
            self.upper_bound = [self.upper_bound]*self.variables_num
        self.func = func
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.binary_code_length = binary_code_length
        self.func_type = func_type
        self.global_best_target = None
        self.global_best_point = None
        self.generations_best_targets = []
        self.global_best_index = 0
        self.generations_best_points = []
        self.global_best_raw_point = None
    def init_population(self):
        '''
        init the population
        :return: None
        '''
        self.population = np.random.randint(0,2,(self.population_size,self.variables_num*self.binary_code_length))
        self.calculate(self.population)

    def calculate(self,population,real_population = None,complex_constraints = None,complex_constraints_C = ga_config.complex_constraints_C):
        '''
        calculate the global_best_target,global_best_points,
        generations_best_target,generations_best_points
        :param population:
        :param real_population: if the population is the real code type population
        :param add_generations:if add to generations_best_target and generations_best_points
        :return: None
        '''
        if real_population is None:
            real_population = self._convert_binary_to_real(population)
        targets_func = np.zeros(self.population_size)
        for i in range(self.population_size):
            if complex_constraints is None:
                targets_func[i] = self.func(real_population[i])
            else:
                targets_func[i] = self.func(real_population[i])
                tmp_plus = self._check_constraints(real_population[i],complex_constraints)
                if  tmp_plus > 0:
                    if self.func_type == ga_config.func_type_min:
                        targets_func[i] += complex_constraints_C*tmp_plus
                    else:
                        targets_func[i] -= complex_constraints_C*tmp_plus
        if self.func_type == ga_config.func_type_min:
            if self.global_best_target is None:
                self.global_best_target = np.min(targets_func)
                self.global_best_raw_point = population[targets_func.argmin()]
                self.global_best_point = real_population[targets_func.argmin()]
            else:
                if self.global_best_target > np.min(targets_func):
                    self.global_best_target = np.min(targets_func)
                    self.global_best_raw_point = population[targets_func.argmin()]
                    self.global_best_point = real_population[targets_func.argmin()]
            self.generations_best_targets.append(np.min(targets_func))
            self.generations_best_points.append(real_population[targets_func.argmin()])
        else:
            if self.global_best_target is None:
                self.global_best_target = np.max(targets_func)
                self.global_best_raw_point = population[targets_func.argmax()]
                self.global_best_point = real_population[targets_func.argmax()]
            else:
                if self.global_best_target < np.max(targets_func):
                    self.global_best_target = np.max(targets_func)
                    self.global_best_raw_point = population[targets_func.argmax()]
                    self.global_best_point = real_population[targets_func.argmax()]
            self.generations_best_targets.append(np.max(targets_func))
            self.generations_best_points.append(real_population[targets_func.argmax()])


    def _check_constraints(self,data,constraints):
        res = 0
        for constraint in constraints:
            if constraint(data) > 0:
                res += constraint(data)
        return res


    def select(self,probs = None,complex_constraints = None,complex_constraints_C = ga_config.complex_constraints_C,M = ga_config.M):
        if probs is None:
            real_population = self._convert_binary_to_real(self.population)
            targets_func = np.zeros(self.population_size)
            for i in range(self.population_size):
                if complex_constraints is None:
                    targets_func[i] = self.func(real_population[i])
                else:
                    targets_func[i] = self.func(real_population[i])
                    tmp_plus = self._check_constraints(real_population[i],complex_constraints)
                    if self.func_type == ga_config.func_type_min:
                        targets_func[i] += tmp_plus*complex_constraints_C
                    else:
                        targets_func[i] -= tmp_plus*complex_constraints_C
                    if targets_func[i] < 0:
                        if self.func_type == ga_config.func_type_min:
                            raise ValueError("Please make sure that the target function value is > 0!")
                        else:
                            targets_func = 1./(abs(targets_func[i])*M)
            assert (np.all(targets_func > 0) == True),"Please make sure that the target function value is > 0!"
            if self.func_type == ga_config.func_type_min:
                targets_func = 1./targets_func
            prob_func = targets_func/np.sum(targets_func)
        else:
            assert (len(probs) == self.population_size and abs(sum(probs)-1) < ga_config.eps), "rank_select_probs should be list or array of size %d,and sum to 1!" % self.population_size
            prob_func = probs
        new_population = np.zeros_like(self.population)
        for i in range(self.population_size):
            choice = np.random.choice(self.population_size,p = prob_func)
            new_population[i] = self.population[choice]
        self.population = new_population

    def _convert_binary_to_real(self,population,cross_code = False):
        '''
        convert binary population to real population
        :param population: binary population,shape:(self.population_size,self.binary_code_length*self.variables_num)
        :return:real_population,shape:(self.population_size,self.variables_num)
        '''
        if cross_code:
            population = self._convert_from_cross_code(population)
        real_population = np.zeros((self.population_size,self.variables_num))
        base_max = float(int("1"*self.binary_code_length,2))
        for i in range(self.population_size):
            for j in range(self.variables_num):
                binary_arr = population[i][j*self.binary_code_length:(j+1)*self.binary_code_length]
                real_value = int("".join(map(str,list(binary_arr))),2)
                real_population[i][j] = real_value*(self.upper_bound[j]-self.lower_bound[j])/base_max + self.lower_bound[j]
        return real_population

    def _convert_from_cross_code(self,population):
        new_population = np.zeros_like(population)
        for i in range(self.population_size):
            tmp = []
            for j in range(self.variables_num):
                tmp.append(list(population[i][j*self.binary_code_length:(j+1)*self.binary_code_length]))
            #print(tmp)
            tmp = list(zip(tmp))
            res = []
            for t in tmp:
                res += list(t[0])
            new_population[i] = np.array(res)
        return new_population


    def cross(self,cross_rate=None):
        if cross_rate is None:
            cross_rate = self.cross_rate
        indices = list(range(self.population_size))
        np.random.shuffle(indices)
        first_indices = indices[:self.population_size//2]
        second_indices = indices[self.population_size//2:]
        for i in range(self.population_size//2):
            if np.random.random() < cross_rate:
                # generate position to cross,using single point cross method
                cross_pos = np.random.choice(self.binary_code_length*self.variables_num)
                self.population[first_indices[i],cross_pos:],self.population[second_indices[i],cross_pos:] = self.population[second_indices[i],cross_pos:],self.population[first_indices[i],cross_pos:]


    def mutate(self,mutation_rate = None):
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        # using numpy vectorized method to accelarate the computing speed
        r_matrix = np.random.rand(self.population_size,self.binary_code_length*self.variables_num)
        bool_matrix = r_matrix < mutation_rate
        self.population[bool_matrix] ^= 1

    def run(self):
        self.init_population()
        for i in range(self.generations):
            self.select()
            self.cross()
            self.mutate()
            self.calculate(self.population)
        if self.func_type == ga_config.func_type_min:
            self.global_best_index = np.array(self.generations_best_targets).argmin()
        else:
            self.global_best_index = np.array(self.generations_best_targets).argmax()

    def save_plot(self,save_name = "SGA.png"):
        plt.plot(self.generations_best_targets,'r-')
        plt.title("Best target function value for %d generations" % self.generations)
        plt.xlabel("generations")
        plt.ylabel("best target function value")
        plt.savefig(save_name)

    def show_result(self):
        print("-"*20,"SGA config is:","-"*20)
        # iterate all the class attributes
        for k,v in self.__dict__.items():
            if k not in ['population','generations_best_points','global_best_point','generations_best_targets','rank_select_probs',
                         'global_best_index','global_best_target','global_best_raw_point']:
                print("%s:%s" %(k,v))
        print("-"*20,"SGA caculation result is:","-"*20)
        print("global best generation index/total generations:%s/%s" % (self.global_best_index,self.generations))
        print("global best point:%s" % self.global_best_point)
        print("global best target:%s" % self.global_best_target)
        





