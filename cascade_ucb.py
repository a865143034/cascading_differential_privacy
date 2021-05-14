import sys
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
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
        self.weights= np.zeros(L)
    
    def initialize(self, dataset, weights):
        w1=sorted(weights,reverse=True)
        self.T[0, :] = 1
        for t in range(self.L - 1):
            d = np.random.permutation(self.L)
            At = np.append([t], d[d != t][:self.K-1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        #print(w1)
        r = 1
        for k in range(self.K):
            r = r*(1-w1[k])
        self.best_f = 1-r
    #f函数有问题
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
        self.epsilon=0
        self.delta=0
        self.weights=np.zeros(L)


    def initialize(self, dataset, weights, epsilon, delta):
        w1=sorted(weights,reverse=True)
        self.T[0, :] = 1
        for t in range(self.L):
            d = np.random.permutation(self.L)
            At = np.append([t], d[d != t][:self.K-1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]

        # Best reward
        r = 1
        for k in range(self.K):
            r = r*(1-w1[k])
        self.best_f = 1-r

        self.weights=weights
        self.epsilon=epsilon
        self.delta=delta
        self.sigma = 1/epsilon * np.sqrt(2 * self.K * np.log(1.25/delta))
        

    def f(self, t):
        r = 1
        for k in range(self.K):
            r = r * (1 - self.w[t - 1, self.A[t, k]])
        reward = 1 - r
        return reward

    def update_ucb_item(self, e, t):#need to check
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        gaussian_term = 0.1*self.sigma * np.sqrt((2*np.log(2 * pow(t, 3)))/self.T[t-1, e])
            #print(np.sqrt((2*np.log(2 * pow(t, 3)))/self.T[t-1, e]))
        #print('*****')
        #print(c)
        #print(gaussian_term)
        return self.w[t-1, e] + c + gaussian_term

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]):
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]] + 0.01*np.random.normal(0, self.sigma))/self.T[t, self.A[t, k]]
                np.random.normal(0, self.sigma)

            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1 + 0.01*np.random.normal(0, self.sigma))/self.T[t, self.A[t, k]]

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
        #print(self.f(t))
        # print(self.f(t))
        # if self.f(t)<0:
        #     print(w[t-1])
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
        self.w = np.zeros((number_of_rounds, L)) #base arm的reward
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
        w1=sorted(weights,reverse=True)
        r = 1
        for k in range(self.K):
            r = r*(1-w1[k])
        self.best_f = 1-r #期望奖励
        self.epsilon = epsilon
        self.weights=weights




    def f(self, t):
        r = 1
        #print(self.w[t-1])
        for k in range(self.K):
            r = r * (1 - self.w[t - 1, self.A[t, k]])
        reward = 1 - r
        return reward

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        laplace_term = 0.1*(self.K/self.epsilon) * np.sqrt((24 * np.log(t))/self.T[t-1, e])
        # print('*****')
        # print(c)
        # print(laplace_term)
        # print('*******')
        # print(c)
        # print(laplace_term)
        return self.w[t-1, e] + c + laplace_term

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]): #C[t]截止的position，判断条件是看是否是最后一个
                self.T[t, self.A[t, k]] += 1
                #print(self.K, np.random.laplace(self.epsilon/self.K))
                self.w[t, self.A[t, k]] = float(
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]] + 0.01*np.random.laplace(0,self.K/self.epsilon))/self.T[t, self.A[t, k]]
            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1 + 0.01*np.random.laplace(0,self.K/self.epsilon))/self.T[t, self.A[t, k]]



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
        # print(self.best_f)
        # print(self.f(t))
        # print(self.regrets[t] )
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
        w1=sorted(weights,reverse=True)
        r = 1
        for k in range(self.K):
            r = r*(1-w1[k])
        self.best_f = 1-r #期望奖励
        self.epsilon=epsilon

    def f(self, t):
        r = 1
        for k in range(self.K):
            r = r * (1 - self.w[t - 1, self.A[t, k]])
        reward = 1 - r
        return reward

    def cal_sum(self,t):
        b = bin(t)
        num1 = len(b) - 2
        sum1=num1*np.random.laplace(0, 2*self.L*num1 /self.epsilon)

        residue = t - pow(2, num1 - 1)
        if residue == 0: return sum1
        num2 = int(np.log2(residue)) + 1
        sum2=num2*np.random.laplace(0, 2*self.L*num2 / self.epsilon)
        #print(num1, residue, num2)
        return sum1+sum2

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5 * np.log(t)) / self.T[t - 1, e])
        laplace_term = 0.1*3 * self.K * np.log(t) * (np.log(self.T[t - 1, e])) ** (1.5) / (self.T[t - 1, e] * self.epsilon)
        #print(c)
        #print(laplace_term)
        sum = self.cal_sum(t)
        hat_w = self.w[t - 1, e] + 0.01*sum/self.T[t - 1, e]
        #print(0.01*sum/self.T[t - 1, e])
        #print(print(laplace_term))
        #print(hat_w)
        # print('***')
        # print(hat_w)
        # print(c)
        # print(laplace_term)
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





class CombiUCB_LDP_gaussian():
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
        self.sigma=0
        self.gamma=0
        self.delta=0


    def initialize(self, dataset, weights, epsilon, delta):#epsilon是lap的参数0.2
        self.T[0,:] = 1
        for t in range(self.L):#保证全部遍历一遍
            d = np.random.permutation(self.L)
            #print(d != t)#A_t是t时刻的动作，第一个坐标表示时间，后面是动作
            At = np.append([t], d[d != t][:self.K-1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        w1=sorted(weights,reverse=True)
        r = 0
        for k in range(self.K):
            r += w1[k]
        self.best_f = r #期望奖励
        self.epsilon = epsilon
        self.delta=delta




    def f(self, t):
        # best permutation reward
        r = 0
        for k in range(self.K):
            r += self.w[t-1, self.A[t, k]]
        reward = r
        return reward

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        gaussian_term = 0.1*(2/self.epsilon) * np.sqrt((2*self.K* np.log(t)*np.log(1.25/self.delta))/self.T[t-1, e])
        # print(c)
        # print(gaussian_term)
        return self.w[t-1, e] + c + gaussian_term

    def update_weights(self, t):
        self.sigma=1/self.epsilon*np.sqrt(2*self.K*np.log(1.25/self.delta))
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]): #C[t]截止的position，判断条件是看是否是最后一个
                self.T[t, self.A[t, k]] += 1
                #print(self.K, np.random.laplace(self.epsilon/self.K))
                self.w[t, self.A[t, k]] = float(
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]] + 0.01*np.random.normal(0,self.sigma))/self.T[t, self.A[t, k]]
            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1 + 0.01*np.random.normal(0,self.sigma))/self.T[t, self.A[t, k]]



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
            self.C[t] = np.argmax(reward) #截止的position，第一个最大值的索引
        else:
            self.C[t] = 1e6


        self.update_weights(t)



class CombiUCB_origin():
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


    def initialize(self, dataset, weights):#epsilon是lap的参数0.2
        self.T[0,:] = 1
        for t in range(self.L):#保证全部遍历一遍
            d = np.random.permutation(self.L)
            #print(d != t)#A_t是t时刻的动作，第一个坐标表示时间，后面是动作
            At = np.append([t], d[d != t][:self.K-1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward

        w1=sorted(weights,reverse=True)
        r = 0
        for k in range(self.K):
            r += w1[k]
        self.best_f = r #期望奖励




    def f(self, t):
        # best permutation reward
        r = 0
        for k in range(self.K):
            r += self.w[t-1, self.A[t, k]]
        reward = r
        return reward

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        return self.w[t-1, e] + c

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]): #C[t]截止的position，判断条件是看是否是最后一个
                self.T[t, self.A[t, k]] += 1
                #print(self.K, np.random.laplace(self.epsilon/self.K))
                self.w[t, self.A[t, k]] = float(
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]] )/self.T[t, self.A[t, k]]
            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1 )/self.T[t, self.A[t, k]]



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
            self.C[t] = np.argmax(reward) #截止的position，第一个最大值的索引
        else:
            self.C[t] = 1e6


        self.update_weights(t)




class CombiUCB_LDP_laplace():


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
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        w1=sorted(weights,reverse=True)
        r = 0
        for k in range(self.K):
            r += w1[k]
        self.best_f = r #期望奖励
        self.epsilon = epsilon


    def f(self, t):
        # best permutation reward
        r = 0
        for k in range(self.K):
            r += self.w[t-1, self.A[t, k]]
        reward = r
        return reward

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        laplace_term = (self.K/self.epsilon) * np.sqrt((24 * np.log(t))/self.T[t-1, e])
        # print('*****')
        # print(c)
        # print(laplace_term)
        return self.w[t-1, e] + c + 0.1*laplace_term

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]): #C[t]截止的position，判断条件是看是否是最后一个
                self.T[t, self.A[t, k]] += 1
                #print(self.K, np.random.laplace(self.epsilon/self.K))
                self.w[t, self.A[t, k]] = float(
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]] + 0.01*np.random.laplace(0,self.K/self.epsilon))/self.T[t, self.A[t, k]]
            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1 + 0.01*np.random.laplace(0,self.K/self.epsilon))/self.T[t, self.A[t, k]]



    #这个函数最重要
    def one_round(self, t, dataset):
        self.U[t] = [self.update_ucb_item(e, t)
                     for e in range(self.L)]
        self.A[t] = np.argsort(self.U[t])[-self.K:][::-1]
        #print(self.A[t])
        # get reward
        reward = dataset[t][self.A[t]]
        # compute regret
        immediate_regret = self.best_f-self.f(t)
        # self.regrets[t] = self.regrets[t-1] + np.abs(immediate_regret)
        self.regrets[t] = np.abs(immediate_regret)
        # get index of  attractive item

        #reward只用于确定C_t
        if np.sum(reward) > 0:
            self.C[t] = np.argmax(reward) #截止的position，第一个最大值的索引
        else:
            self.C[t] = 1e6

        self.update_weights(t)