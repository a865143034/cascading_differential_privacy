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


def cal_num(t):
    b = bin(t)
    num1 = len(b) - 2
    residue = t - pow(2, num1 - 1)
    if residue==0: return num1
    num2 = int(np.log2(residue))+1
    print(num1,residue,num2)
    return num1 + num2


print(cal_num(32))