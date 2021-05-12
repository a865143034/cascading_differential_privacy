import numpy as np
import math
# import numpy as np
# x=np.array([1,4,3,-1,6,9])
#
# x=x.argsort()
# #print(x)
#
#
# a=np.random.laplace(0,100)
# b=np.random.laplace(100)
#
# print(a)
# print(b)
#
# c=int(np.log2(8))
# print(c)
from sklearn import preprocessing

def cal_num(t):
    b = bin(t)
    num1 = len(b) - 2
    residue = t - pow(2, num1 - 1)
    if residue==0: return num1
    num2 = int(np.log2(residue))+1
    print(num1,residue,num2)
    return num1 + num2

a=np.random.laplace(0,100,100)

b=np.random.normal(0,267,100)

print(a)
print(b)
#print(b)

a=[1,2,3]
b = np.mean(a)
print(b)


