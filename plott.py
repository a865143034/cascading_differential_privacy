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



A=[]
B=[]
C=[]

f1=open('cascading_ldp_origin', 'r')

for line in f1.readlines():
    line=line.strip()
    line=float(line)
    A.append(line)
A=np.cumsum(A)

f2=open('cascading_ldp_laplace_2', 'r')
for line in f2.readlines():
    line=line.strip()
    B.append(float(line))
B=np.cumsum(B)
f3=open('cascading_ldp_gaussian_2', 'r')
for line in f3.readlines():
    line=line.strip()
    C.append(float(line))
C=np.cumsum(C)

plt.plot(A,'r')
plt.plot(B,'b')
plt.plot(C,'y')
plt.ylabel("Reget")
plt.xlabel("Rounds")
plt.savefig('convergence')
plt.show()