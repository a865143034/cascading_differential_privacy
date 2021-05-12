#coding:utf-8
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import  randrange, betavariate, shuffle
from numpy.random import uniform, choice
import math
from  tqdm import tqdm, trange
from utils import *
from cascade_ucb import *


number_of_rounds =  int(1e5)
p = 0.2
delta  = 0.15
L = 8
K = 2

#weights = [p for i in range(K)] + [np.abs(p-delta) for i in range(L-K)]

def run_epsilon():
    epsilon=0.02
    weights = [0.1, 0.05, 0.3, 0.6, 0.8, 0.6, 0.9, 0.2]
    A=[]
    while epsilon<=2:
        res=0
        for i in range(10):
            shuffle(weights)
            cascade_model = CascadeUCB_LDP_laplace(number_of_rounds, L, K)

            dataset = generate_data(number_of_rounds, weights)
            # 先全部把所有数据sample出来

            # initializing
            cascade_model.initialize(dataset, weights,epsilon)
            # training
            for t in range(1, number_of_rounds):
                cascade_model.one_round(t, dataset)

            reg=cascade_model.regrets
            res+=np.mean(reg)
        res/=10
        A.append(res)
        epsilon+=0.02
        print(A)


run_epsilon()