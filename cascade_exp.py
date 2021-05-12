import sys
import os 
import numpy as np

import seaborn as sns
import pandas as pd
from random import  randrange, betavariate, shuffle
from numpy.random import uniform, choice
import math 
#from  tqdm import tqdm, trange
from utils import *
from cascade_ucb import CascadeUCB, CascadeUCB_LDP_gaussian, CascadeUCB_LDP_laplace

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
rcParams['mathtext.default'] = 'regular'

number_of_rounds =  int(1e5)
p = 0.2
d  = 0.15
L = 20
K = 2
#shuffle(weights)

r = np.zeros(number_of_rounds)
r_gaussian = np.zeros(number_of_rounds)
r_laplace = np.zeros(number_of_rounds)
for i in range(10):
    weights = [p for i in range(K)] + [np.abs(p-d) for i in range(L-K)]
    cascade_model = CascadeUCB(number_of_rounds,L,K)
    #cascade_model_gaussian = CascadeUCB_LDP_gaussian(number_of_rounds,L,K)
    cascade_model_laplace = CascadeUCB_LDP_laplace(number_of_rounds,L,K)
    dataset = generate_data(number_of_rounds, weights)

    # initializing
    cascade_model.initialize(dataset, weights)
    #cascade_model_gaussian.initialize(dataset, weights, delta = 0.2, epsilon = 0.15)
    cascade_model_laplace.initialize(dataset, weights, epsilon = 0.2)

    print("-- trial " + str(i) + '--')
    # training
    for t in tqdm(range(1, number_of_rounds), ascii = True):#trange(1,number_of_rounds):
        cascade_model.one_round(t,dataset)
        #cascade_model_gaussian.one_round(t,dataset)
        cascade_model_laplace.one_round(t,dataset)
    
    r += cascade_model.regrets
    #r_gaussian += cascade_model_gaussian.regrets
    r_laplace += cascade_model_laplace.regrets
    
r /= 10
#r_gaussian /=10
r_laplace /= 10


regrets = pd.Series(r)
#regrets_gaussian = pd.Series(r_gaussian)
regrets_laplace = pd.Series(r_laplace)
baseline_f = open('npbaseline.txt', 'w+')
for r in r:
    baseline_f.write(str(r) + "\n")
baseline_f.close()
#gaussian_f = open('gaussian_r.txt', 'w+')
#for r in r_gaussian:
#    gaussian_f.write(str(r) + "\n")
#gaussian_f.close()
laplace_f = open('laplace_r.txt', 'w+')
for r in r_laplace:
    laplace_f.write(str(r) + "\n")
laplace_f.close()

plt.figure(figsize=(7,5))
plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.tick_params(labelsize=10)
plt.plot(regrets, label = "Non-private")
#plt.plot(regrets_gaussian, label = "Gaussian LDP", linestyle = ':')
plt.plot(regrets_laplace, label = "Laplace LDP")
plt.ylabel("Reget")
plt.xlabel("Timesteps")
plt.legend()
plt.savefig('regret')
plt.show()

plt.figure(figsize=(7,5))
plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.tick_params(labelsize=10)
plt.plot(np.cumsum(regrets), label = "Non-private", linestyle = '--')
#plt.plot(np.cumsum(regrets_gaussian), label = "Gaussian LDP", linestyle = ':')
plt.plot(np.cumsum(regrets_laplace), label = "Laplace LDP", linestyle = "-.")
plt.ylabel("Cumulative Reget")
plt.xlabel("Timesteps")
plt.legend()
#plt.xscale('log')
plt.savefig('culregret')
plt.show()

#cascade_model.T = pd.DataFrame(cascade_model.T)
#fig, ax = plt.subplots(figsize=(12,6))
#plt.yscale('log')
#ax.plot(cascade_model.T)
#ax.legend(cascade_model.T)
#plt.title("L = 8 , K= 2 ")
#plt.ylabel("Number of selections of item e ")
#plt.xlabel("Rounds")
#plt.savefig('nbitems')
#plt.show()

# Hyperparams
list_L =[16,16,16,8]
list_K =[2,4,4,2]
list_delta = [0.15,0.15,0.075,0.075]
n_runs = 5
p=0.2
number_of_rounds =  int(1e2)