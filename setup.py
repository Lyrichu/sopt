#!/usr/bin/env python
# coding=utf-8
from setuptools import setup,find_packages

setup(
    name="sopt",
    version="0.0.6",
    description="sopt:a simple python optimizer library",
    long_description=
    '''
    sopt is a simple python optimizer library.Currentlly,it includes some stochastic optimization
    algorithms,like Genetic Algorithm(GA),Particle Swarm Optimization(PSO),Simulated Anealing
    (SA),Random Walk(and its improvement version),and some gradient based optimizers,like Gradient
    Descent,Momentum,AdaGrad,RMSProp and Adam Optimizers.For the GA optimizers,it includes many
    kinds of different selected methods,mutation methods etc,as well as for PSO and other optimizers,
    so you can try many different kinds of optimizers with different settings,all the stochastic optimization
    also supports the non-linear complex constraints by using penalty methods or dropout-bad-solution methods.
    ''',
    author='lyrichu',
    author_email='919987476@qq.com',
    url = "http://www.github.com/Lyrichu",
    maintainer='lyrichu',
    maintainer_email='919987476@qq.com',
    packages=['sopt','sopt/GA','sopt/SGA','sopt/PSO','sopt/test','sopt/util','sopt/Optimizers'],
    package_dir={'sopt': 'sopt'},
    install_requires=['numpy']
)
