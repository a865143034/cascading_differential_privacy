import sys
import os
import numpy as np
import pandas as pd

class CascadeUCB():
    def __init__(self, number_of_rounds, L, K):
        super().__init__()
        self.number_of_rounds = number_of_rounds
        self.L = L
        self.K = K
        self.T = np.zeros((number_of_rounds, L))
        self.U = np.zeros((number_of_rounds, L))
        self.w = np.zeros((number_of_rounds, L))
        self.A = np.zeros((number_of_rounds, K), dtype=np.int32)
        self.C = np.zeros(number_of_rounds)
        self.regrets = np.zeros(number_of_rounds)
    
    def initialize(self, dataset, weights):
        self.T[0, :] = 1
        for t in range(self.L - 1):
            d = np.random.permutation(self.L)
            At = np.append([t], d[d != t][:self.K-1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        r = 1
        for k in range(self.K):
            r = r*(1-weights[k])
        self.best_f = 1-r

    def f(self, t):
        r = 1
        for k in range(self.K):
            r = r*(1-self.w[t-1, self.A[t, k]])
        reward = 1-r
        return reward

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        return self.w[t-1, e] + c

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]):
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]])/self.T[t, self.A[t, k]]

            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1)/self.T[t, self.A[t, k]]

    def one_round(self, t, dataset):
        self.U[t] = [self.update_ucb_item(e, t)
                     for e in range(self.L)]
        self.A[t] = np.argsort(self.U[t])[-self.K:][::-1]
        # get reward
        reward = dataset[t][self.A[t]]
        # compute regret
        immediate_regret = self.best_f-self.f(t)
        # self.regrets[t] = self.regrets[t-1] + np.abs(immediate_regret)
        self.regrets[t] = np.abs(immediate_regret)
        # get index of  attractive item
        if np.sum(reward) > 0:
            self.C[t] = np.argmax(reward)
        else:
            self.C[t] = 1e6
        self.update_weights(t)


class CascadeUCB_LDP_gaussian():
    def __init__(self, number_of_rounds, L, K):
        super().__init__()
        self.number_of_rounds = number_of_rounds
        self.L = L
        self.K = K
        self.T = np.zeros((number_of_rounds, L))
        self.U = np.zeros((number_of_rounds, L))
        self.w = np.zeros((number_of_rounds, L))
        self.A = np.zeros((number_of_rounds, K), dtype=np.int32)
        self.C = np.zeros(number_of_rounds)
        self.regrets = np.zeros(number_of_rounds)
        self.sigma = 0
        self.gamma = 0 #pow(number_of_rounds, -3)
        self.best_f=0


    def initialize(self, dataset, weights, epsilon, delta):
        self.T[0, :] = 1
        for t in range(self.L):
            d = np.random.permutation(self.L)
            At = np.append([t], d[d != t][:self.K-1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        r = 1
        for k in range(self.K):
            r = r*(1-weights[k])
        self.best_f = 1-r
        self.sigma = 1/epsilon * np.sqrt(2 * self.K * np.log(1.25/delta))
        #print(1/epsilon, np.sqrt(2 * self.K * np.log(1.25/delta)))
        

    def f(self, t):
        # best permutation reward
        r = 1
        for k in range(self.K):
            r = r*(1-self.w[t-1, self.A[t, k]])
        reward = 1-r
        return reward

    def update_ucb_item(self, e, t):#need to check
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        if self.T[t-1, e] < 1:
            gaussian_term = self.sigma * np.sqrt((2*np.log(2 * pow(t, 3)))/1)
        else:
            gaussian_term = self.sigma * np.sqrt((2*np.log(2 * pow(t, 3)))/self.T[t-1, e])
            #print(np.sqrt((2*np.log(2 * pow(t, 3)))/self.T[t-1, e]))
        return self.w[t-1, e] + c + min(1, gaussian_term)

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]):
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]] + np.random.normal(0, self.sigma))/self.T[t, self.A[t, k]]

            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1 + np.random.normal(0, self.sigma))/self.T[t, self.A[t, k]]

    def one_round(self, t, dataset):
        #update gamma
        self.gamma = pow(t, -3)

        self.U[t] = [self.update_ucb_item(e, t)
                     for e in range(self.L)]
        self.A[t] = np.argsort(self.U[t])[-self.K:][::-1]
        
        # get reward
        reward = dataset[t][self.A[t]]
        # compute regret
        immediate_regret = self.best_f-self.f(t)
        # self.regrets[t] = self.regrets[t-1] + np.abs(immediate_regret)
        self.regrets[t] = np.abs(immediate_regret)
        # get index of  attractive item
        if np.sum(reward) > 0:
            self.C[t] = np.argmax(reward)
        else:
            self.C[t] = 1e6
        self.update_weights(t)





class CascadeUCB_LDP_laplace():
    def __init__(self, number_of_rounds, L, K):
        super().__init__()
        self.number_of_rounds = number_of_rounds
        self.L = L
        self.K = K
        self.T = np.zeros((number_of_rounds, L))#base arm摇的次数
        self.U = np.zeros((number_of_rounds, L))#UCB term
        self.w = np.zeros((number_of_rounds, L),dtype=np.float) #base arm的reward
        self.A = np.zeros((number_of_rounds, K), dtype=np.int32) #每轮的action
        self.C = np.zeros(number_of_rounds) #每轮截止的position
        self.regrets = np.zeros(number_of_rounds)
        self.epsilon = 0
        self.best_f=0


    def initialize(self, dataset, weights, epsilon):#epsilon是lap的参数0.2
        self.T[0,:] = 1
        for t in range(self.L):#保证全部遍历一遍
            d = np.random.permutation(self.L)
            #print(d != t)#A_t是t时刻的动作，第一个坐标表示时间，后面是动作
            At = np.append([t], d[d != t][:self.K-1])
            #print(d.size)
            #print(At)
            #print('****')
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        r = 1
        for k in range(self.K):
            r = r*(1-weights[k])
        self.best_f = 1-r #期望奖励
        self.epsilon = epsilon




    def f(self, t):
        # best permutation reward
        r = 1
        for k in range(self.K):
            r = r*(1-self.w[t-1, self.A[t, k]])
        reward = 1-r
        return reward

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        laplace_term = (self.K/self.epsilon) * np.sqrt((24 * np.log(t))/self.T[t-1, e])
        return self.w[t-1, e] + c + laplace_term

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]): #C[t]截止的position，判断条件是看是否是最后一个
                self.T[t, self.A[t, k]] += 1
                #print(self.K, np.random.laplace(self.epsilon/self.K))
                self.w[t, self.A[t, k]] = float(
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]] + np.random.laplace(0,self.K/self.epsilon))/self.T[t, self.A[t, k]]
            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1 + np.random.laplace(0,self.K/self.epsilon))/self.T[t, self.A[t, k]]



    #这个函数最重要
    def one_round(self, t, dataset):
        self.U[t] = [self.update_ucb_item(e, t)
                     for e in range(self.L)]
        self.A[t] = np.argsort(self.U[t])[-self.K:][::-1]
        #print(self.A[t])
        # get reward
        reward = dataset[t][self.A[t]]
        #print(reward)
        #print(dataset.shape)

        # compute regret
        immediate_regret = self.best_f-self.f(t)
        # self.regrets[t] = self.regrets[t-1] + np.abs(immediate_regret)
        self.regrets[t] = np.abs(immediate_regret)
        # get index of  attractive item


        #reward只用于确定C_t
        if np.sum(reward) > 0:
            self.C[t] = np.argmax(reward) #截止的position，第一个最大值的缩阴
        else:
            self.C[t] = 1e6


        self.update_weights(t)


class CascadeUCB_DP():
    def __init__(self, number_of_rounds, L, K):
        super().__init__()
        self.number_of_rounds = number_of_rounds
        self.L = L
        self.K = K
        self.T = np.zeros((number_of_rounds, L))
        self.U = np.zeros((number_of_rounds, L))
        self.w = np.zeros((number_of_rounds, L))
        self.A = np.zeros((number_of_rounds, K), dtype=np.int32)
        self.C = np.zeros(number_of_rounds)
        self.regrets = np.zeros(number_of_rounds)
        self.best_f = 0
        self.epsilon=0

    def initialize(self, dataset, weights,epsilon):
        self.T[0, :] = 1
        for t in range(self.L - 1):
            d = np.random.permutation(self.L)
            At = np.append([t], d[d != t][:self.K - 1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        r = 1
        for k in range(self.K):
            r = r * (1 - weights[k])
        self.best_f = 1 - r
        self.epsilon=epsilon

    def f(self, t):
        r = 1
        for k in range(self.K):
            r = r * (1 - self.w[t - 1, self.A[t, k]])
        reward = 1 - r
        return reward

    def cal_num(self,t):
        b = bin(t)
        num1 = len(b) - 2
        residue = t - pow(2, num1 - 1)
        if residue == 0: return num1
        num2 = int(np.log2(residue)) + 1
        #print(num1, residue, num2)
        return num1 + num2

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5 * np.log(t)) / self.T[t - 1, e])
        laplace_term = 0.01*3 * self.K * np.log(t) * (np.log(self.T[t - 1, e])) ** (1.5) / (self.T[t - 1, e] * self.epsilon)
        #print(c)
        #print(laplace_term)
        num = self.cal_num(t)
        hat_w = self.w[t - 1, e] + np.random.laplace(0, self.K / self.epsilon) * num/self.T[t - 1, e]
        #print(hat_w)
        #print('***')
        return hat_w + c+laplace_term

    def update_weights(self, t):
        self.T[t] = self.T[t - 1]
        self.w[t] = self.w[t - 1]
        for k in range(min(self.K, int(self.C[t]) + 1)):
            if k < int(self.C[t]):
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                                                  self.T[t - 1, self.A[t, k]] * self.w[t - 1, self.A[t, k]]) / self.T[
                                              t, self.A[t, k]]

            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                                                  self.T[t - 1, self.A[t, k]] * self.w[t - 1, self.A[t, k]] + 1) / \
                                          self.T[t, self.A[t, k]]

    def one_round(self, t, dataset):
        self.U[t] = [self.update_ucb_item(e, t)
                     for e in range(self.L)]
        self.A[t] = np.argsort(self.U[t])[-self.K:][::-1]
        # get reward
        reward = dataset[t][self.A[t]]
        # compute regret
        immediate_regret = self.best_f - self.f(t)


        # self.regrets[t] = self.regrets[t-1] + np.abs(immediate_regret)
        self.regrets[t] = immediate_regret
        # get index of  attractive item
        if np.sum(reward) > 0:
            self.C[t] = np.argmax(reward)
        else:
            self.C[t] = 1e6
        self.update_weights(t)