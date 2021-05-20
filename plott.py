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


def plot_3():
    A=[]
    B=[]
    C=[]
    D=[]
    E=[]
    for i in range(99):
        D.append(i*0.02+0.02)
    #print(D)
    #print(len(D))
    f1=open('epsilon_ceshi_origin.txt', 'r')

    for line in f1.readlines():
        line=line.strip()
        line=float(line)
        A.append(line)
    #A=np.cumsum(A)
    #print(len(A))

    f2=open('epsilon_ceshi_laplace.txt', 'r')
    for line in f2.readlines():
        line=line.strip()
        B.append(float(line))
    #B=np.cumsum(B)

    f3=open('epsilon_ceshi_gaussian.txt', 'r')
    for line in f3.readlines():
        line=line.strip()
        C.append(float(line))
    #C=np.cumsum(C)

    # f4=open('dp_l_20', 'r')
    # for line in f4.readlines():
    #     line=line.strip()
    #     D.append(float(line))
    # D=np.cumsum(D)


    figure,ax = plt.subplots()
    #plt.tight_layout()
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(linestyle='--')
    plt.xlim(xmin=0, xmax=2)
    plt.ylim(ymin=0, ymax=0.08)
    plt.plot(D,A,'#054E9F',linestyle='-.',label='Non-private',linewidth=2)
    plt.plot(D,B,color='coral',label='Laplace-LDP',linestyle='--',linewidth=2)
    plt.plot(D,C,color='m',label='Gaussian-LDP',linestyle=':',linewidth=2)
    #plt.plot(D,color='g',linestyle='-.',label=chr(949)+'=2',linewidth=2)
    #plt.plot(A,color='c',label='L=8',linewidth=2)
    plt.ylabel("Average Cumulative Reget",fontsize=14)
    plt.xlabel("epsilon",fontsize=14)
    plt.legend(fontsize=14,loc=2)

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('epsilon_vary_finall.pdf')
    plt.show()


def plot_1():
    A=[]

    f1=open('epsilon_ceshi_laplace.txt', 'r')

    for line in f1.readlines():
        line=line.strip()
        line=float(line)
        A.append(line)

    plt.tight_layout()
    plt.style.use('fivethirtyeight')

    plt.figure(figsize=(5, 5))

    plt.plot(A, 'r')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tick_params(labelsize=13)
    plt.xlabel('Timesteps')  # ,fontsize=18)
    plt.ylabel('Cumulativeregret')  # ,fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=15, handlelength=1.5, framealpha=0)



    plt.show()





def plot_4():
    A = []
    B = []
    C = []
    D=[]
    for i in range(99):
        D.append(i * 0.02 + 0.02)
    # print(D)
    # print(len(D))
    f1 = open('combinatorial_ldp_origin', 'r')

    for line in f1.readlines():
        line = line.strip()
        line = float(line)
        A.append(line)
    A = np.cumsum(A)
    # print(len(A))

    f2 = open('combinatorial_ldp_laplace_2', 'r')
    for line in f2.readlines():
        line = line.strip()
        B.append(float(line))
    B = np.cumsum(B)

    f3 = open('combinatorial_ldp_gaussian_2', 'r')
    for line in f3.readlines():
        line = line.strip()
        C.append(float(line))
    C = np.cumsum(C)


    figure, ax = plt.subplots()
    # plt.tight_layout()
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(linestyle='--')
    plt.xlim(xmin=0, xmax=1e5)
    plt.ylim(ymin=0, ymax=30000)
    plt.plot(A, 'g', label='Non-private', linewidth=2)
    plt.plot(B, color='coral', label='Laplace-LDP', linestyle='--', linewidth=2)
    plt.plot(C, color='#054E9F', label='Gaussian-LDP', linestyle='-.', linewidth=2)
    plt.ylabel("Cumulative Reget", fontsize=14)
    plt.xlabel("Rounds", fontsize=14)
    plt.legend(fontsize=14, loc=2)

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('combi_2.pdf')
    plt.show()



plot_3()



#plt.savefig(figname,bbox_inches="tight")