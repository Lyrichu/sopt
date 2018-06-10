#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: functions.py
@time: 2018/06/06 08:01
@description:
useful functions defination for GA
"""
from sopt.util.ga_config import *
import math
from functools import reduce

Rosenbrock_lower_bound = -2.048
Rosenbrock_upper_bound = 2.048
Rosenbrock_variables_num = 2
Rosenbrock_func_type = basic_config.func_type_max

quadratic11_lower_bound = 1
quadratic11_upper_bound = 11
quadratic11_variables_num = 11
quadratic11_func_type = basic_config.func_type_min

quadratic50_lower_bound = 1
quadratic50_upper_bound = 50
quadratic50_variables_num = 50
quadratic50_func_type = basic_config.func_type_min

quadratic100_lower_bound = 1
quadratic100_upper_bound = 100
quadratic100_variables_num = 100
quadratic100_func_type = basic_config.func_type_min

quadratic500_lower_bound = 1
quadratic500_upper_bound = 500
quadratic500_variables_num = 500
quadratic500_func_type = basic_config.func_type_min

quadratic1000_lower_bound = 1
quadratic1000_upper_bound = 1000
quadratic1000_variables_num = 1000
quadratic1000_func_type = basic_config.func_type_min


shubert2_lower_bound = -10
shubert2_upper_bound = 10
shubert2_variables_num = 2
shubert2_func_type = basic_config.func_type_min

def Rosenbrock(x):
    '''
    Rosenbrock function:f(x1,x2) = 100*(x1^2-x2)^2 + (1-x1)^2
    -2.048 <= x1,x2 <= 2.048
    global max point:(-2.048,-2.048)
    :param x:1 D ndarray
    :return:func calculation result
    '''
    x1 = x[0]
    x2 = x[1]
    return 100*(x1**2-x2)**2 + (1-x1)**2

def _quadratic(x,n):
    res = 1
    for i in range(1,n+1):
        res += (x[i-1]-i)**2
    return res

def quadratic11(x):
    return _quadratic(x,11)

def quadratic50(x):
    return _quadratic(x,50)

def quadratic100(x):
    return _quadratic(x,100)

def quadratic500(x):
    return _quadratic(x,500)

def quadratic1000(x):
    return _quadratic(x,1000)

def _shubert(x,n):
    '''
    :param x:
    :param n:
    :return:
    '''
    tmps = []
    for i in range(n):
        tmp = 0
        for j in range(1,6):
            tmp += j*math.cos((j+1)*x[i]+j)
        tmps.append(tmp)
    res = reduce(lambda x,y:x*y,tmps) + 1000
    return res

def shubert2(x):
    '''
    shubert function,which has 760 local minimize points,
    and the global minimize point is:-186.731 (+ 1000 = 813.269)
    -10 <= x1,x2 <= 10,
    we plus 1000 to ensure that the function value is always > 0
    :param x:
    :return:
    '''
    return _shubert(x,2)










