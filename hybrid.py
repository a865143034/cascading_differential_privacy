#coding:utf-8
import numpy as np
#log 没用
class LogM():
    def __init__(self,epsilon):
        self.epsilon = epsilon
        self.beta = 0
    
    def action(self, t):
        self.beta =0

        # if t is power of 2, every power of 2 has exactly 1 bit set to 1
        if (t & (t-1) == 0) and t != 1:
            self.beta += np.random.laplace(0,1/self.epsilon)
            return  self.beta

        return self.beta

class Binary_M():

    def __init__(self,T, epsilon):
        self.T = T
        self.epsilon = epsilon

    def initialized(self):
        self.sum = np.zeros(3*self.T)


    def action(self, t):
        for t in range(T):
            num=0
            a=bin(t)
            print(a[2:]) #是个string

bina=Binary_M(100,0.2)

bina.initialized()

bina.action(9)



class HybridM():
    def __init__(self,):
        11

    def action(self):
        a=1
