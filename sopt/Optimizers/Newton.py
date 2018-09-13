#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: Newton.py
@time: 2018/09/11 20:29
@description:
newton based optimization method,like:dfp and bfgs
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sopt.util.newton_config import newton_config
from sopt.Optimizers.Gradients import gradients

class DFP:
    def __init__(self,
                 func,
                 variables_num,
                 func_type = newton_config.func_type_min,
                 eps = newton_config.eps,
                 init_variables = newton_config.init_variables,
                 epochs = newton_config.epochs,
                 min_step = newton_config.min_step,
                 max_step = newton_config.max_step,
                 step_size = newton_config.step_size
                 ):
        '''
        newton based optimization method of dfp
        :param func: the target function
        :param variables_num: the number of variables
        :param func_type: 'min' or 'max'
        :param eps: the min stop eps
        :param init_variables:the initial variables
        :param epochs:iteration numbers
        :param min_step:the minimize step
        :param max_step:the maximize step
        :param step_size: the step size
        '''
        self.func = func
        self.variables_num = variables_num
        self.func_type = func_type
        self.eps = eps
        self.init_variables = init_variables
        if self.init_variables is None:
            self.init_variables = np.random.uniform(-1,1,self.variables_num)
        assert type(self.init_variables) == np.ndarray and len(self.init_variables) == self.variables_num,\
            "init_variables should be int or float ndarray of size %d !" % self.variables_num
        self.epochs = epochs
        self.min_step = min_step
        self.max_step = max_step
        self.step_size = step_size
        self.generations_targets = []
        self.generations_points = []
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0

    def find_best_step(self,x,d):
        best_step = self.min_step
        best_res = self.func(x-best_step*d)
        for step in np.arange(self.min_step,self.max_step,self.step_size):
            cur_res = self.func(x-step*d)
            if self.func_type == newton_config.func_type_min:
                if cur_res < best_res:
                    best_step = step
                    best_res = cur_res
            else:
                if cur_res > best_res:
                    best_step = step
                    best_res = cur_res
        return best_step


    def run(self):
        D = np.eye(self.variables_num,self.variables_num)
        x = self.init_variables.reshape(-1,1)
        iteration = 0
        for i in range(self.epochs):
            g1 = gradients(self.func,x.flatten()).reshape((self.variables_num,1))
            d = D.dot(g1) # D is approximately equals to Hessain Matrix
            best_step = self.find_best_step(x,d)
            s = best_step * d
            x -= s
            iteration += 1
            g2 = gradients(self.func,x.flatten()).reshape((self.variables_num,1))
            if np.sqrt((g2 ** 2).sum()) < self.eps:
                break
            y = g2 - g1
            # update D
            D += s.dot(s.T) / ((s.T).dot(y)) - D.dot(y).dot(y.T).dot(D) / ((y.T).dot(D).dot(y))
            self.generations_points.append(x.flatten())
            self.generations_targets.append(self.func(x.flatten()))


        if self.func_type == newton_config.func_type_min:
            self.global_best_target = np.min(np.array(self.generations_targets))
            self.global_best_index = np.argmin(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]
        else:
            self.global_best_target = np.max(np.array(self.generations_targets))
            self.global_best_index = np.argmax(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]



    def save_plot(self, save_name="dfp.png"):
        plt.plot(self.generations_targets, 'r-')
        plt.xlabel("epochs")
        plt.ylabel("target function value")
        plt.plot("newton dfp with %d epochs" % self.epochs)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "newton dfp config is:", "-" * 20)
        for k, v in self.__dict__.items():
            if k not in ['init_variables', 'generations_targets', 'generations_points', 'm',
                         'global_best_target', 'global_best_point', 'global_best_index']:
                print("%s:%s" % (k, v))

        print("-" * 20, "newton dfp caculation result is:", "-" * 20)
        print("global best epoch/total epochs:%s/%s" % (self.global_best_index, self.epochs))
        print("global best point:", self.global_best_point)
        print("global best target:", self.global_best_target)


class BFGS:
    def __init__(self,
                 func,
                 variables_num,
                 func_type=newton_config.func_type_min,
                 eps=newton_config.eps,
                 init_variables=newton_config.init_variables,
                 epochs=newton_config.epochs,
                 min_step=newton_config.min_step,
                 max_step=newton_config.max_step,
                 step_size=newton_config.step_size
                 ):
        '''
        newton based optimization method of bfgs
        :param func: the target function
        :param variables_num: the number of variables
        :param func_type: 'min' or 'max'
        :param eps: the min stop eps
        :param init_variables:the initial variables
        :param epochs:iteration numbers
        :param min_step:the minimize step
        :param max_step:the maximize step
        :param step_size: the step size
        '''
        self.func = func
        self.variables_num = variables_num
        self.func_type = func_type
        self.eps = eps
        self.init_variables = init_variables
        if self.init_variables is None:
            self.init_variables = np.random.uniform(-1, 1, self.variables_num)
        assert type(self.init_variables) == np.ndarray and len(self.init_variables) == self.variables_num, \
            "init_variables should be int or float ndarray of size %d !" % self.variables_num
        self.epochs = epochs
        self.min_step = min_step
        self.max_step = max_step
        self.step_size = step_size
        self.generations_targets = []
        self.generations_points = []
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0

    def find_best_step(self, x, d):
        best_step = self.min_step
        best_res = self.func(x - best_step * d)
        for step in np.arange(self.min_step, self.max_step, self.step_size):
            cur_res = self.func(x - step * d)
            if self.func_type == newton_config.func_type_min:
                if cur_res < best_res:
                    best_step = step
                    best_res = cur_res
            else:
                if cur_res > best_res:
                    best_step = step
                    best_res = cur_res
        return best_step

    def run(self):
        B = np.eye(self.variables_num, self.variables_num)
        x = self.init_variables.reshape(-1, 1)
        iteration = 0
        for i in range(self.epochs):
            g1 = gradients(self.func, x.flatten()).reshape((self.variables_num, 1))
            d = linalg.inv(B).dot(g1)  # D is approximately equals to Hessain Matrix
            best_step = self.find_best_step(x, d)
            s = best_step * d
            x -= s
            iteration += 1
            g2 = gradients(self.func, x.flatten()).reshape((self.variables_num, 1))
            if np.sqrt((g2 ** 2).sum()) < self.eps:
                break
            y = g2 - g1
            # update D
            B += y.dot(y.T) / ((y.T).dot(s)) - (B.dot(s).dot(s.T).dot(B)) / (s.T.dot(B).dot(s))
            self.generations_points.append(x.flatten())
            self.generations_targets.append(self.func(x.flatten()))

        if self.func_type == newton_config.func_type_min:
            self.global_best_target = np.min(np.array(self.generations_targets))
            self.global_best_index = np.argmin(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]
        else:
            self.global_best_target = np.max(np.array(self.generations_targets))
            self.global_best_index = np.argmax(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]

    def save_plot(self, save_name="bfgs.png"):
        plt.plot(self.generations_targets, 'r-')
        plt.xlabel("epochs")
        plt.ylabel("target function value")
        plt.plot("newton bfgs with %d epochs" % self.epochs)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "newton bfgs config is:", "-" * 20)
        for k, v in self.__dict__.items():
            if k not in ['init_variables', 'generations_targets', 'generations_points', 'm',
                         'global_best_target', 'global_best_point', 'global_best_index']:
                print("%s:%s" % (k, v))

        print("-" * 20, "newton bfgs caculation result is:", "-" * 20)
        print("global best epoch/total epochs:%s/%s" % (self.global_best_index, self.epochs))
        print("global best point:", self.global_best_point)
        print("global best target:", self.global_best_target)






