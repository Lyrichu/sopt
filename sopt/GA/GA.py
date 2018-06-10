#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: GA.py
@time: 2018/06/06 10:18
@description:
More general genetic algorithm(GA)
"""
from warnings import warn
import numpy as np
import sys
sys.path.append("..")
from sopt.SGA.SGA import SGA
from sopt.util.ga_config import *

class GA(SGA):
    def __init__(self,lower_bound,upper_bound,variables_num,func,
                 cross_rate = basic_config.cross_rate,
                 mutation_rate = basic_config.mutation_rate,
                 population_size = basic_config.population_size,
                 generations = basic_config.generations,
                 binary_code_length= basic_config.binary_code_length,
                 func_type = basic_config.func_type_min,
                 # below is the new attribute of GA
                 cross_rate_exp = basic_config.cross_rate_exp,
                 mutation_rate_exp = basic_config.mutation_rate_exp,
                 code_type = code_type.binary,
                 cross_code = False,
                 select_method = select_method.proportion,
                 rank_select_probs = None,
                 tournament_num = basic_config.tournament_num,
                 cross_method = cross_method.one_point,
                 arithmetic_cross_alpha = basic_config.arithmetic_cross_alpha,
                 arithmetic_cross_exp = basic_config.arithmetic_cross_exp,
                 mutation_method = mutation_method.uniform,
                 none_uniform_mutation_rate = basic_config.none_uniform_mutation_rate,
                 complex_constraints = None,
                 complex_constraints_method = complex_constraints_method.penalty,
                 complex_constraints_C = basic_config.complex_constraints_C,
                 M = basic_config.M
                 ):
        '''
                :param lower_bound: the lower bound of variables,real number or list of real numbers
                :param upper_bound: the upper bound of variables,real number or list of real numbers
                :param variable_num: the number of variables
                :param func: the target function
                :param cross_rate: GA cross rate
                :param mutation_rate: GA mutation rate
                :param population_size: the size of GA population
                :param generations: the GA generations count
                :param binary_code_length: the binary code length to represent a real number
                :param func_type:'min' means to evaluate the minimum target function;'max'
                 means to evaluate the maximum target function
                 :param cross_rate_exp: use `(cross_rate_exp^generation)*cross_rate` to sightly
                 increase cross_rate as GA generation increase
                 :param mutation_rate_exp:use `(mutation_rate_exp^generation)*mutation_rate` to sightly
                 increase mutation_rate as GA generation increase
                 :param code_type: using what kind of variable codings,it can be 'binary' or 'gray' or 'real'
                 :param cross_code: True means using cross coding method,False means using concanate coding method
                 :param select_method: it can be:
                 1.'proportion':means using proportion select method
                 2.'keep_best': means keep the best solution,which do not use cross or mutate,and replace the
                 worsest solution with the best solution
                 3.'determinate_sampling': using determinate sampling select method
                 4.'rssr': rssr means:remainder stochastic sampling with replacement
                 5.'rank': using rank based select method
                 6.'stochastic_tournament': using stochastic tournament select method
                 :param rank_select_probs: if using rank select method,then you should give a list or array of probs,which has size `self.population_size`
                 :param tournament_num:tournament numbers,default is 2
                 :param cross_method: it can be:
                 1.'one_point': using one-point crossover method
                 2.'two_point':using two-point crossover method
                 3.'uniform':using uniform crossover method
                 4.'arithmetic':using arithmetic crossover method
                 :param arithmetic_cross_alpha:the alpha parameter value of arithmetic crossover
                 :param arithmetic_cross_exp:the exp parameter value of arithmetic crossover
                 :param mutation_method: it can be:
                 1.'simple':using simple mutation method
                 2.'uniform': using uniform mutation method
                 3.'boundary':using boundary mutation method
                 4.'none_uniform':using none uniform mutation method
                 5.'gaussian':using gaussian mutation
                 :param none_uniform_mutation_rate:none uniform mutation rate value
                 :param complex_constraints: list of complex constraints func,or None(means no complex constraints)
                 :param complex_constraints_method:it can be:
                 1.'penalty': using penalty method to target function to satisfy the complex constraints,
                 other methods like 'loop' or 'proportion' is temperately not supported
                 :param complex_constraints_C: the penalty alpha value
                 :param M: the divided max value,default is 1e8
                '''
        super(GA,self).__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            variables_num=variables_num,
            func=func,
            cross_rate=cross_rate,
            mutation_rate=mutation_rate,
            population_size=population_size,
            generations = generations,
            binary_code_length = binary_code_length,
            func_type = func_type
        )
        self.cross_rate_exp = cross_rate_exp
        self.mutation_rate_exp = mutation_rate_exp
        self.code_type = code_type
        self.cross_code = cross_code
        self.select_method = select_method
        self.rank_select_probs = rank_select_probs
        self.tournament_num = tournament_num
        self.cross_method = cross_method
        self.arithmetic_cross_alpha = arithmetic_cross_alpha
        self.arithmetic_cross_exp = arithmetic_cross_exp
        self.mutation_method = mutation_method
        self.none_uniform_mutation_rate = none_uniform_mutation_rate
        self.complex_constraints = complex_constraints
        self.complex_constraints_method = complex_constraints_method
        if self.complex_constraints_method != 'penalty':
            warn("Currently not support '%s' complex_constraints_method!Using 'penalty' instead!" % self.complex_constraints_method)
        self.complex_constraints_C = complex_constraints_C
        self.global_generations_step = 0
        self.M = M

    def init_population(self):
        if self.code_type == code_type.binary or self.code_type == code_type.gray:
            self.population = np.random.randint(0, 2,(self.population_size, self.variables_num * self.binary_code_length))
            self.calculate(self.population,complex_constraints=self.complex_constraints,complex_constraints_C=self.complex_constraints_C)
        else:
            self.population = []
            for i in range(self.variables_num):
                self.population.append(self.lower_bound[i] + (self.upper_bound[i]-self.lower_bound[i])*np.random.rand(self.population_size))
            self.population = np.array(self.population).transpose()
            self.calculate(self.population,self.population,complex_constraints=self.complex_constraints,complex_constraints_C=self.complex_constraints_C)

    def calculate(self,population,real_population = None,complex_constraints = None,complex_constraints_C = basic_config.complex_constraints_C):
        if self.code_type == code_type.binary:
            super(GA,self).calculate(population,complex_constraints = complex_constraints,complex_constraints_C = complex_constraints_C)
        elif self.code_type == code_type.gray:
            binary_population = self._convert_gray_to_binary(population,self.cross_code)
            super(GA, self).calculate(binary_population,complex_constraints = complex_constraints,complex_constraints_C = complex_constraints_C)
        else:
            super(GA,self).calculate(population,real_population,complex_constraints = complex_constraints,complex_constraints_C = complex_constraints_C)

    def _convert_gray_to_binary(self,population,cross_code = False):
        if cross_code is True:
            population = self._convert_from_cross_code(population)
        binary_population = np.zeros_like(population)
        for i in range(self.population_size):
            for j in range(self.variables_num*self.binary_code_length):
                if j == 0:
                    binary_population[i][j] = population[i][j]
                else:
                    binary_population[i][j] = binary_population[i][j-1]^population[i][j]
        return binary_population

    def select(self,probs = None,complex_constraints = None,complex_constraints_C = basic_config.complex_constraints_C,M=basic_config.M):
        if self.code_type == code_type.binary:
            real_population = self._convert_binary_to_real(self.population, self.cross_code)
        elif self.code_type == code_type.gray:
            binary_population = self._convert_gray_to_binary(self.population, self.cross_code)
            real_population = self._convert_binary_to_real(binary_population)
        else:
            real_population = self.population
        targets_func = np.zeros(self.population_size)
        for i in range(self.population_size):
            targets_func[i] = self.func(real_population[i])
            if complex_constraints is not None:
                tmp_plus = self._check_constraints(real_population[i], complex_constraints)
                if self.func_type == basic_config.func_type_min:
                    targets_func[i] += tmp_plus * complex_constraints_C
                else:
                    targets_func[i] -= tmp_plus * complex_constraints_C
                if targets_func[i] < 0:
                    if self.func_type == basic_config.func_type_min:
                        raise ValueError("Please make sure that the target function value is > 0!")
                    else:
                        targets_func[i] = 1 / (abs(targets_func[i])*M)
        assert (np.all(targets_func > 0) == True), "Please make sure that the target function value is > 0!"
        if self.func_type == basic_config.func_type_min:
            targets_func = 1. / targets_func
        if self.select_method == select_method.proportion or self.select_method == select_method.keep_best:
            prob_func = targets_func/np.sum(targets_func)
            super(GA,self).select(prob_func,complex_constraints = complex_constraints,complex_constraints_C = complex_constraints_C,M=M)
        elif self.select_method == select_method.determinate_sampling or self.select_method ==select_method.rssr:
            prob_func = targets_func / np.sum(targets_func)
            new_population = np.zeros_like(self.population)
            index = 0
            for i in range(self.population_size):
                N = int(self.population_size*prob_func[i])
                if N > 0:
                    new_population[index:(index+N)] = self.population[i]
                    index += N
            if index < self.population_size:
                if self.select_method == select_method.determinate_sampling:
                    sorted_prob_index = np.argsort(prob_func)
                    new_population[index:] = self.population[sorted_prob_index[index:]]
                else:
                    remain_prob = prob_func - (prob_func*self.population_size).astype('int32')/self.population_size
                    remain_prob /= sum(remain_prob)
                    for i in range(index,self.population_size):
                        choice = np.random.choice(self.population_size,p=remain_prob)
                        new_population[i] = self.population[choice]
            self.population = new_population

        elif self.select_method == select_method.rank:
            targets_rank = np.argsort(targets_func)
            if self.rank_select_probs is None:
                self.rank_select_probs = np.arange(1,self.population_size+1)/float(np.sum(np.arange(1,self.population_size+1)))
                self.rank_select_probs = self.rank_select_probs[targets_rank]
            else:
                self.rank_select_probs = np.sort(self.rank_select_probs)
                assert (len(self.rank_select_probs) == self.population_size and abs(np.sum(self.rank_select_probs)-1) < 1e4),"rank_select_probs should be an array of size %d that sum to 1!" % self.population_size
                self.rank_select_probs = self.rank_select_probs[targets_rank]
            super(GA,self).select(self.rank_select_probs,complex_constraints = complex_constraints,complex_constraints_C = complex_constraints_C,M=M)
        else:
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                choice_indices = np.random.choice(self.population_size,self.tournament_num)
                if self.func_type == basic_config.func_type_min:
                    choice = choice_indices[np.argmin(targets_func[choice_indices])]
                else:
                    choice = choice_indices[np.argmax(targets_func[choice_indices])]
                new_population[i] = self.population[choice]
            self.population = new_population



    def cross(self,cross_rate = None):
        cross_rate = self.cross_rate*(self.cross_rate_exp**self.global_generations_step)
        if self.cross_method == cross_method.one_point:
            if self.code_type in (code_type.binary,code_type.gray):
                super(GA,self).cross(cross_rate)
            else:
                indices = list(range(self.population_size))
                np.random.shuffle(indices)
                first_indices = indices[:self.population_size // 2]
                second_indices = indices[self.population_size // 2:]
                for i in range(self.population_size // 2):
                    if np.random.random() < cross_rate:
                        # generate position to cross,using single point cross method
                        cross_pos = np.random.choice(self.variables_num)
                        self.population[first_indices[i], cross_pos:], self.population[second_indices[i],cross_pos:]  = \
                        self.population[second_indices[i],cross_pos:],self.population[first_indices[i], cross_pos:]
        elif self.cross_method == cross_method.two_point:
            indices = list(range(self.population_size))
            np.random.shuffle(indices)
            first_indices = indices[:self.population_size // 2]
            second_indices = indices[self.population_size // 2:]
            for i in range(self.population_size // 2):
                if np.random.random() < cross_rate:
                    # generate position to cross,using single point cross method
                    if self.code_type == code_type.real:
                        cross_pos1,cross_pos2 = np.random.choice(self.variables_num+1,2)
                    else:
                        cross_pos1, cross_pos2 = np.random.choice(self.variables_num*self.binary_code_length+1, 2)
                    cross_pos1,cross_pos2 = min(cross_pos1,cross_pos2),max(cross_pos1,cross_pos2)
                    self.population[first_indices[i], cross_pos1:cross_pos2], self.population[second_indices[i], cross_pos1:cross_pos2] = \
                    self.population[second_indices[i], cross_pos1:cross_pos2], self.population[first_indices[i],cross_pos1:cross_pos2]
        elif self.cross_method == cross_method.uniform:
            indices = list(range(self.population_size))
            np.random.shuffle(indices)
            first_indices = indices[:self.population_size // 2]
            second_indices = indices[self.population_size // 2:]
            for i in range(self.population_size // 2):
                if self.code_type == code_type.real:
                    length = self.variables_num
                else:
                    length = self.variables_num*self.binary_code_length
                for j in range(length):
                    if np.random.random() < cross_rate:
                        # generate position to cross,using single point cross method
                        self.population[first_indices[i],j],self.population[second_indices[i],j] = \
                        self.population[second_indices[i], j],self.population[first_indices[i],j]

        else:
            assert self.code_type == code_type.real,"arithmetic crossover must use real codings!"
            arithmetic_cross_alpha = self.arithmetic_cross_alpha*(self.arithmetic_cross_exp**self.global_generations_step)
            indices = list(range(self.population_size))
            np.random.shuffle(indices)
            first_indices = indices[:self.population_size // 2]
            second_indices = indices[self.population_size // 2:]
            for i in range(self.population_size // 2):
                if np.random.random() < cross_rate:
                    first = arithmetic_cross_alpha*self.population[first_indices[i]] + (1-arithmetic_cross_alpha)*self.population[second_indices[i]]
                    second = arithmetic_cross_alpha*self.population[second_indices[i]] + (1-arithmetic_cross_alpha)*self.population[first_indices[i]]
                    for j in range(self.variables_num):
                        if first[j] < self.lower_bound[j]:
                            first[j] = self.lower_bound[j]
                        if second[j] < self.lower_bound[j]:
                            second[j] = self.lower_bound[j]
                        if first[j] > self.upper_bound[j]:
                            first[j] = self.upper_bound[j]
                        if second[j] > self.upper_bound[j]:
                            second[j] = self.upper_bound[j]
                    self.population[first_indices[i]] = first
                    self.population[second_indices[i]] = second




    def mutate(self,mutation_rate = None):
        mutation_rate = self.mutation_rate*(self.mutation_rate_exp**self.global_generations_step)
        if self.mutation_method == mutation_method.simple:
            for i in range(self.population_size):
                if np.random.random() < self.mutation_rate:
                    if self.code_type == code_type.real:
                        choice = np.random.randint(self.variables_num)
                        r = np.random.random()
                        self.population[i][choice] = self.lower_bound[choice] + r*(self.upper_bound[choice]-self.lower_bound[choice])
                    else:
                        choice = np.random.randint(self.variables_num*self.binary_code_length)
                        self.population[i][choice] = 0 if self.population[i][choice] == 1 else 1
        elif self.mutation_method == mutation_method.uniform:
            if self.code_type == code_type.real:
                for i in range(self.population_size):
                    for j in range(self.variables_num):
                        if np.random.random() < mutation_rate:
                            r = np.random.random()
                            self.population[i][j] = self.lower_bound[j] + r*(self.upper_bound[j] - self.lower_bound[j])
            else:
                super(GA,self).mutate(mutation_rate)
        elif self.mutation_method == mutation_method.boundary:
            assert self.code_type == code_type.real,'boundary mutation must use real coding type!'
            for i in range(self.population_size):
                for j in range(self.variables_num):
                    if np.random.random() < mutation_rate:
                        if np.random.random() < 0.5:
                            self.population[i][j] = self.lower_bound[j]
                        else:
                            self.population[i][j] = self.upper_bound[j]
        elif self.mutation_method == mutation_method.none_uniform:
            assert self.code_type == code_type.real, 'none_uniform mutation must use real coding type!'
            for i in range(self.population_size):
                for j in range(self.variables_num):
                    if np.random.random() < mutation_rate:
                        r = np.random.random()
                        if np.random.random() < 0.5:
                            self.population[i][j] += -(self.population[i][j]-self.lower_bound[j])*(1-r**(self.none_uniform_mutation_rate*(1-self.global_generations_step/float(self.generations))))
                        else:
                            self.population[i][j] = (self.upper_bound[j]-self.population[i][j])*(1-r**(self.none_uniform_mutation_rate*(1-self.global_generations_step/float(self.generations))))
        else:
            # gaussian mutation
            assert self.code_type == code_type.real, 'gaussian mutation must use real coding type!'
            for i in range(self.population_size):
                for j in range(self.variables_num):
                    if np.random.random() < mutation_rate:
                        mu = 0.5*(self.lower_bound[j]+self.upper_bound[j])
                        sigma = (self.upper_bound[j]-self.lower_bound[j])/6.
                        gaussian = np.random.randn()*sigma + mu
                        if gaussian < self.lower_bound[j]:
                            gaussian = self.lower_bound[j]
                        if gaussian > self.upper_bound[j]:
                            gaussian = self.upper_bound[j]
                        self.population[i][j] = gaussian


    def run(self):
        self.init_population()
        for i in range(self.generations):
            self.select(complex_constraints=self.complex_constraints,complex_constraints_C=self.complex_constraints_C,M=self.M)
            self.cross()
            self.mutate()
            if self.code_type == code_type.real:
                self.calculate(self.population,self.population,complex_constraints=self.complex_constraints,complex_constraints_C=self.complex_constraints_C)
            else:
                self.calculate(self.population,complex_constraints=self.complex_constraints,complex_constraints_C=self.complex_constraints_C)
            if self.select_method == select_method.keep_best:
                if self.code_type == code_type.real:
                    real_population = self.population
                elif self.code_type == code_type.binary:
                    real_population = self._convert_binary_to_real(self.population,self.cross_code)
                else:
                    population = self._convert_gray_to_binary(self.population,self.cross_code)
                    real_population = self._convert_binary_to_real(population)
                targets_func = np.zeros(self.population_size)
                for i in range(self.population_size):
                    targets_func[i] = self.func(real_population[i])
                    if self.complex_constraints:
                        tmp_plus = self._check_constraints(real_population[i],self.complex_constraints)
                        if self.func_type == basic_config.func_type_min:
                            targets_func[i] += tmp_plus
                        else:
                            targets_func -= tmp_plus
                if self.func_type == basic_config.func_type_min:
                    worst_target_index = np.argmax(targets_func)
                else:
                    worst_target_index = np.argmin(targets_func)
                self.population[worst_target_index] = self.global_best_raw_point
            self.global_generations_step += 1

        if self.func_type == basic_config.func_type_min:
            self.global_best_index = np.array(self.generations_best_targets).argmin()
        else:
            self.global_best_index = np.array(self.generations_best_targets).argmax()



    def show_result(self):
        print("-"*20,"GA config is:","-"*20)
        # iterate all the class attributes
        for k,v in self.__dict__.items():
            if k not in ['population','generations_best_points','global_best_point','generations_best_targets','rank_select_probs',
                         'global_best_index','global_best_target','global_best_raw_point']:
                print("%s:%s" %(k,v))
        print("-"*20,"GA caculation result is:","-"*20)
        print("global best target generation index/total generations:%s/%s" % (self.global_best_index,self.generations))
        print("global best point:%s" % self.global_best_point)
        print("global best target:%s" % self.global_best_target)








