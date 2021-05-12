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
L = 40
K = 20

weights = [p for i in range(K)] + [np.abs(p-delta) for i in range(L-K)]

#shuffle(weights)
#cascade_model = CascadeUCB(number_of_rounds,L,K)
cascade_model = CascadeUCB_LDP_laplace(number_of_rounds,L,K)
#cascade_model = CascadeUCB_LDP_gaussian(number_of_rounds,L,K)
#cascade_model = CascadeUCB_DP(number_of_rounds,L,K)
#cascade_model = CombiUCB_LDP_gaussian(number_of_rounds,L,K)
#cascade_model = CombiUCB_LDP_laplace(number_of_rounds,L,K)

dataset = generate_data(number_of_rounds, weights)
#先全部把所有数据sample出来

# initializing
cascade_model.initialize(dataset,weights)

# training
for t in range(1,number_of_rounds) :
    cascade_model.one_round(t,dataset)

f=open('cascading_ldp_origin_b','w')
for i in cascade_model.regrets:
    f.write(str(i)+'\n')
f.close()

#print(cascade_model.regrets)
regrets = pd.Series(cascade_model.regrets)
plt.figure(figsize=(12, 6))

plt.plot(regrets)
plt.ylabel("Reget")
plt.xlabel("Rounds")
plt.savefig('convergence')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(regrets))
plt.ylabel("Cumulative Reget")
plt.xlabel("Rounds")
plt.show()



# cascade_model.T = pd.DataFrame(cascade_model.T)
# fig, ax = plt.subplots(figsize=(12,6))
# plt.yscale('log')
# ax.plot(cascade_model.T)
# ax.legend(cascade_model.T)
# plt.title("L = 8 , K= 2 ")
# plt.ylabel("Number of selections of item e ")
# plt.xlabel("Rounds")
# plt.savefig('nbitems')
# plt.show()

assert 1==0
# Hyperparams
list_L =[16,16,16,8]
list_K =[2,4,4,2]
list_delta = [0.15,0.15,0.075,0.075]
n_runs = 5
p=0.2
number_of_rounds =  int(1e2)



#res , n_regret = run_experiment(list_L, list_delta, list_K, number_of_rounds,n_runs,p)


#print(res)