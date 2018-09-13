#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: GD.py
@time: 2018/06/09 20:22
@description:
gradient descent optimizers(including GD,Momentum,AdaGrad,RMSProp,Adam)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sopt.util.functions import *
from sopt.util.gradients_config import gradients_config


def gradients(func,variables):
    grads = np.zeros_like(variables)
    for i in range(len(variables)):
        variables_delta = np.copy(variables)
        variables_delta[i] += gradients_config.delta
        grads[i] = (func(variables_delta)-func(variables))/gradients_config.delta

    return grads

class GradientDescent:
    def __init__(self,
                 func,
                 variables_num,
                 func_type = gradients_config.func_type_min,
                 init_variables = gradients_config.init_variables,
                 lr = gradients_config.lr,
                 epochs = gradients_config.epochs
                 ):
        '''
        basic GradientDescent
        :param func: the target function
        :param variables_num:the numbers of variables
        :param func_type:'min' or 'max'
        :param init_variables: the init variables value
        :param lr: the learning rate
        :param epochs: the iteration epochs
        '''
        self.func = func
        self.variables_num = variables_num
        self.func_type = func_type
        self.init_variables = init_variables
        if self.init_variables is None:
            self.init_variables = np.random.uniform(-1,1,self.variables_num)
        assert len(self.init_variables) == self.variables_num and type(self.init_variables) == np.ndarray,\
            "init_variables should be int or float ndarray of size %d!" %self.variables_num
        self.lr = lr
        self.epochs = epochs
        self.generations_targets = []
        self.generations_points = []
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0



    def run(self):
        variables = self.init_variables
        for i in range(self.epochs):
            grads = gradients(self.func,variables)
            if self.func_type == gradients_config.func_type_min:
                variables -= self.lr*grads
            else:
                variables += self.lr*grads
            self.generations_points.append(variables)
            self.generations_targets.append(self.func(variables))

        if self.func_type == gradients_config.func_type_min:
            self.global_best_target = np.min(np.array(self.generations_targets))
            self.global_best_index = np.argmin(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]
        else:
            self.global_best_target = np.max(np.array(self.generations_targets))
            self.global_best_index = np.argmax(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]



    def save_plot(self,save_name = "GradientDescent.png"):
        plt.plot(self.generations_targets,'r-')
        plt.xlabel("epochs")
        plt.ylabel("target function value")
        plt.plot("Gradient Descent with %d epochs" % self.epochs)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "Gradient Descent config is:", "-" * 20)
        for k,v in self.__dict__.items():
            if k not in ['init_variables','generations_targets','generations_points',
                         'global_best_target','global_best_point','global_best_index']:
                print("%s:%s" %(k,v))

        print("-"*20,"Gradient Descent caculation result is:","-"*20)
        print("global best epoch/total epochs:%s/%s" % (self.global_best_index,self.epochs))
        print("global best point:",self.global_best_point)
        print("global best target:",self.global_best_target)



class Momentum:
    def __init__(self,
                 func,
                 variables_num,
                 func_type = gradients_config.func_type_min,
                 init_variables = gradients_config.init_variables,
                 lr = gradients_config.lr,
                 beta = gradients_config.momentum_beta,
                 epochs = gradients_config.epochs
                 ):
        '''
        Momentum Optimizer
        :param func: the target function
        :param variables_num:the numbers of variables
        :param func_type:'min' or 'max'
        :param init_variables: the init variables value
        :param lr: the learning rate
        :param beta: the Momentum beta parameter
        :param epochs: the iteration epochs
        '''
        self.func = func
        self.variables_num = variables_num
        self.func_type = func_type
        self.init_variables = init_variables
        if self.init_variables is None:
            self.init_variables = np.random.uniform(-1,1,self.variables_num)
        assert len(self.init_variables) == self.variables_num and type(self.init_variables) == np.ndarray,\
            "init_variables should be int or float ndarray of size %d!" %self.variables_num
        self.lr = lr
        self.beta = beta
        self.epochs = epochs
        self.generations_targets = []
        self.generations_points = []
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0


    def run(self):
        variables = self.init_variables
        self.m = np.zeros(self.variables_num)
        for i in range(self.epochs):
            grads = gradients(self.func,variables)
            self.m = self.beta*self.m + self.lr*grads
            if self.func_type == gradients_config.func_type_min:
                variables -= self.m
            else:
                variables += self.m
            self.generations_points.append(variables)
            self.generations_targets.append(self.func(variables))

        if self.func_type == gradients_config.func_type_min:
            self.global_best_target = np.min(np.array(self.generations_targets))
            self.global_best_index = np.argmin(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]
        else:
            self.global_best_target = np.max(np.array(self.generations_targets))
            self.global_best_index = np.argmax(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]



    def save_plot(self,save_name = "Momentum.png"):
        plt.plot(self.generations_targets,'r-')
        plt.xlabel("epochs")
        plt.ylabel("target function value")
        plt.plot("Momentum with %d epochs" % self.epochs)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "Momentum config is:", "-" * 20)
        for k,v in self.__dict__.items():
            if k not in ['init_variables','generations_targets','generations_points','m',
                         'global_best_target','global_best_point','global_best_index']:
                print("%s:%s" %(k,v))

        print("-"*20,"Momentum caculation result is:","-"*20)
        print("global best epoch/total epochs:%s/%s" % (self.global_best_index,self.epochs))
        print("global best point:",self.global_best_point)
        print("global best target:",self.global_best_target)



class AdaGrad:
    def __init__(self,
                 func,
                 variables_num,
                 func_type = gradients_config.func_type_min,
                 init_variables = gradients_config.init_variables,
                 lr = gradients_config.adagrad_lr,
                 eps = gradients_config.eps,
                 epochs = gradients_config.epochs
                 ):
        '''
        Adagrad Optimizer
        :param func: the target function
        :param variables_num:the numbers of variables
        :param func_type:'min' or 'max'
        :param init_variables: the init variables value
        :param lr: the learning rate
        :param eps: the epsilon value
        :param epochs: the iteration epochs
        '''
        self.func = func
        self.variables_num = variables_num
        self.func_type = func_type
        self.init_variables = init_variables
        if self.init_variables is None:
            self.init_variables = np.random.uniform(-1,1,self.variables_num)
        assert len(self.init_variables) == self.variables_num and type(self.init_variables) == np.ndarray,\
            "init_variables should be int or float ndarray of size %d!" %self.variables_num
        self.lr = lr
        self.eps = eps
        self.epochs = epochs
        self.generations_targets = []
        self.generations_points = []
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0


    def run(self):
        variables = self.init_variables
        self.s = np.zeros(self.variables_num)
        for i in range(self.epochs):
            grads = gradients(self.func,variables)
            self.s += np.square(grads)
            if self.func_type == gradients_config.func_type_min:
                variables -= self.lr*grads/(np.sqrt(self.s+self.eps))
            else:
                variables += self.lr*grads/(np.sqrt(self.s+self.eps))
            self.generations_points.append(variables)
            self.generations_targets.append(self.func(variables))

        if self.func_type == gradients_config.func_type_min:
            self.global_best_target = np.min(np.array(self.generations_targets))
            self.global_best_index = np.argmin(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]
        else:
            self.global_best_target = np.max(np.array(self.generations_targets))
            self.global_best_index = np.argmax(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]



    def save_plot(self,save_name = "AdaGrad.png"):
        plt.plot(self.generations_targets,'r-')
        plt.xlabel("epochs")
        plt.ylabel("target function value")
        plt.plot("AdaGrad with %d epochs" % self.epochs)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "AdaGrad config is:", "-" * 20)
        for k,v in self.__dict__.items():
            if k not in ['init_variables','generations_targets','generations_points','s',
                         'global_best_target','global_best_point','global_best_index']:
                print("%s:%s" %(k,v))

        print("-"*20,"AdaGrad caculation result is:","-"*20)
        print("global best epoch/total epochs:%s/%s" % (self.global_best_index,self.epochs))
        print("global best point:",self.global_best_point)
        print("global best target:",self.global_best_target)



class RMSProp:
    def __init__(self,
                 func,
                 variables_num,
                 func_type = gradients_config.func_type_min,
                 init_variables = gradients_config.init_variables,
                 lr = gradients_config.rmsprop_lr,
                 beta = gradients_config.rmsprop_beta,
                 eps = gradients_config.eps,
                 epochs = gradients_config.epochs
                 ):
        '''
        RMSProp Optimizer
        :param func: the target function
        :param variables_num:the numbers of variables
        :param func_type:'min' or 'max'
        :param init_variables: the init variables value
        :param lr: the learning rate
        :param beta: the RMSProp beta parameter
        :param eps: the epsilon value
        :param epochs: the iteration epochs
        '''
        self.func = func
        self.variables_num = variables_num
        self.func_type = func_type
        self.init_variables = init_variables
        if self.init_variables is None:
            self.init_variables = np.random.uniform(-1,1,self.variables_num)
        assert len(self.init_variables) == self.variables_num and type(self.init_variables) == np.ndarray,\
            "init_variables should be int or float ndarray of size %d!" %self.variables_num
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.epochs = epochs
        self.generations_targets = []
        self.generations_points = []
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0


    def run(self):
        variables = self.init_variables
        self.s = np.zeros(self.variables_num)
        for i in range(self.epochs):
            grads = gradients(self.func,variables)
            self.s = self.beta*self.s + (1-self.beta)*np.square(grads)
            if self.func_type == gradients_config.func_type_min:
                variables -= self.lr*grads/(np.sqrt(self.s+self.eps))
            else:
                variables += self.lr*grads/(np.sqrt(self.s+self.eps))
            self.generations_points.append(variables)
            self.generations_targets.append(self.func(variables))

        if self.func_type == gradients_config.func_type_min:
            self.global_best_target = np.min(np.array(self.generations_targets))
            self.global_best_index = np.argmin(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]
        else:
            self.global_best_target = np.max(np.array(self.generations_targets))
            self.global_best_index = np.argmax(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]



    def save_plot(self,save_name = "RMSProp.png"):
        plt.plot(self.generations_targets,'r-')
        plt.xlabel("epochs")
        plt.ylabel("target function value")
        plt.plot("RMSProp with %d epochs" % self.epochs)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "RMSProp config is:", "-" * 20)
        for k,v in self.__dict__.items():
            if k not in ['init_variables','generations_targets','generations_points','s',
                         'global_best_target','global_best_point','global_best_index']:
                print("%s:%s" %(k,v))

        print("-"*20,"RMSProp caculation result is:","-"*20)
        print("global best epoch/total epochs:%s/%s" % (self.global_best_index,self.epochs))
        print("global best point:",self.global_best_point)
        print("global best target:",self.global_best_target)



class Adam:
    def __init__(self,
                 func,
                 variables_num,
                 func_type = gradients_config.func_type_min,
                 init_variables = gradients_config.init_variables,
                 lr = gradients_config.adam_lr,
                 beta1 = gradients_config.adam_beta1,
                 beta2 = gradients_config.adam_beta2,
                 eps = gradients_config.eps,
                 epochs = gradients_config.epochs
                 ):
        '''
        Adam Optimizer
        :param func: the target function
        :param variables_num:the numbers of variables
        :param func_type:'min' or 'max'
        :param init_variables: the init variables value
        :param lr: the Adam learning rate
        :param beta1: the Adam beta1 parameter
        :param beta2: the Adam beta2 parameter
        :param eps: the epsilon value
        :param epochs: the iteration epochs
        '''
        self.func = func
        self.variables_num = variables_num
        self.func_type = func_type
        self.init_variables = init_variables
        if self.init_variables is None:
            self.init_variables = np.random.uniform(-1,1,self.variables_num)
        assert len(self.init_variables) == self.variables_num and type(self.init_variables) == np.ndarray,\
            "init_variables should be int or float ndarray of size %d!" %self.variables_num
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.epochs = epochs
        self.generations_targets = []
        self.generations_points = []
        self.global_best_target = 0
        self.global_best_point = None
        self.global_best_index = 0


    def run(self):
        variables = self.init_variables
        self.s = np.zeros(self.variables_num)
        self.m = np.zeros(self.variables_num)
        for i in range(self.epochs):
            grads = gradients(self.func,variables)
            self.m = self.beta1*self.m +(1-self.beta1)*grads
            self.s = self.beta2*self.s + (1-self.beta2)*np.square(grads)
            self.m /= (1-self.beta1**(i+1))
            self.s /= (1-self.beta2**(i+1))
            if self.func_type == gradients_config.func_type_min:
                variables -= self.lr*self.m/(np.sqrt(self.s+self.eps))
            else:
                variables += self.lr*self.m/(np.sqrt(self.s+self.eps))
            self.generations_points.append(variables)
            self.generations_targets.append(self.func(variables))

        if self.func_type == gradients_config.func_type_min:
            self.global_best_target = np.min(np.array(self.generations_targets))
            self.global_best_index = np.argmin(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]
        else:
            self.global_best_target = np.max(np.array(self.generations_targets))
            self.global_best_index = np.argmax(np.array(self.generations_targets))
            self.global_best_point = self.generations_points[int(self.global_best_index)]



    def save_plot(self,save_name = "Adam.png"):
        plt.plot(self.generations_targets,'r-')
        plt.xlabel("epochs")
        plt.ylabel("target function value")
        plt.plot("Adam with %d epochs" % self.epochs)
        plt.savefig(save_name)

    def show_result(self):
        print("-" * 20, "Adam config is:", "-" * 20)
        for k,v in self.__dict__.items():
            if k not in ['init_variables','generations_targets','generations_points','s','m',
                         'global_best_target','global_best_point','global_best_index']:
                print("%s:%s" %(k,v))

        print("-"*20,"Adam caculation result is:","-"*20)
        print("global best epoch/total epochs:%s/%s" % (self.global_best_index,self.epochs))
        print("global best point:",self.global_best_point)
        print("global best target:",self.global_best_target)