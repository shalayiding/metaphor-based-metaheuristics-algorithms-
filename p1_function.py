import numpy as np
import sys
from numpy import random
import math


# Rosenbrock function 
def F_08(x): 
    y = 0
    for item in x:
        for i in range(0, len(item)-1):
            y += (item[i]-1)**2 + 100*(item[i+1]-item[i]**2)**2
    return y  


#Schwefel 2.26
def F_09(x):
    alpha = 418.9829*len(x[0])
    y = 0
    for item in x:
        for i in range(0, len(item)):
            y -=item[i]*np.sin(np.sqrt(np.fabs(item[i]))) 
    return alpha+y


#Schwefel 1.2
def F_10(x):
    y= 0
    for item in x:
        sum = 0
        for i in range(0, len(item)):
            s1 = 0
            for j in range(0,i):
                s1 +=item[j]
            sum+=s1
        y+=sum
    return y  

#Schwefel 2.22
def F_11(x):
    y = 0
    for item in x:
        for i in range(0, len(item)):
            y+=abs(item[i])
    y1 = 1
    for item in x:
        for i in range(0, len(item)):
            y1*=abs(item[i])
    return y+y1  

#Schwefel  2.21
def F_12(x):
    y = x[0][0]
    for item in x:
        for i in range(0,len(item)):
            y = max(item[i],y)
    return y

#Sphere
def F_13(x):
    y = 0
    for item in x:
        for i in range(0,len(item)):
            y += item[i]**2
    return y


#Step 
def F_14(x):
    y = 6*len(x[0])
    for item in x:
        for i in range(0,len(item)):
            y  += math.floor(item[i])
    return y
