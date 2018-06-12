#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: constraints.py
@time: 2018/06/09 15:36
@description:
some sample constraints functions
"""
def constraints1(x):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + x2**2 -3

def constraints2(x):
    x1 = x[0]
    x2 = x[1]
    return x1+x2

def constraints3(x):
    x1 = x[0]
    x2 = x[1]
    return -2 -x1 -x2
