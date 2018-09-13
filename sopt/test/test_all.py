#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_all.py
@time: 2018/09/12 18:01
@description:
test all in one
"""
import sys
sys.path.append("..")
from sopt.test import test_AdaGrad
from sopt.test import test_Adam
from sopt.test import test_BFGS
from sopt.test import test_DFP
from sopt.test import test_GA
from sopt.test import test_GradientDescent
from sopt.test import test_Momentum
from sopt.test import test_PSO
from sopt.test import test_RandomWalk
from sopt.test import test_RMSProp
from sopt.test import test_SA
from sopt.test import test_SGA


def test_all():
    test_AdaGrad.TestAdaGrad().test()
    test_Adam.TestAdam().test()
    test_BFGS.TestBFGS().test()
    test_DFP.TestDFP().test()
    test_GA.TestGA().test()
    test_GradientDescent.TestGradientDescent().test()
    test_Momentum.TestMomentum().test()
    test_PSO.TestPSO().test()
    test_RandomWalk.TestRandomWalk().test()
    test_RMSProp.TestRMSProp().test()
    test_SA.TestSA().test()
    test_SGA.TestSGA().test()


if __name__ == '__main__':
    test_all()

