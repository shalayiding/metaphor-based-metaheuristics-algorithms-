import numpy as np
import sys
from numpy import random
import math

# Ackley function
def F_01(x):
    exp_term2  = 0
    exp_term1 = 0
    for item in x:
        for i in range(0, len(item)-1):
            exp_term1 = exp_term1 + math.pow(item[i], 2)
            exp_term2 = exp_term2 + math.cos(2*math.pi*item[i])
    exp_term1 = math.exp(-0.2 * math.sqrt(exp_term1 * 0.5))
    exp_term2 = math.exp(0.5 * exp_term2)
    return -20.0 * exp_term1 - exp_term2 + math.e + 20


# Fletcher-Powell function
def F_02(x):
    Ai = 0
    for item in x:
        for i in range:
            pass
        pass
    return 0


# Griewank Function
def F_03(x):
    term_1 = 0
    term_2 = 1
    for item in x:
        for i in range(0, len(item)):
            term_1 = term_1 + item[i]**2
            term_2 = term_2 * math.cos(item[i]/math.sqrt(i+1))
            
    term_1 = term_1/4000
    return term_1 - term_2 + 1


# Quartic function with noise 
def F_06(x):
    sum_term = 0
    for item in x:
        for i in range(0, len(item)):
            sum_term = sum_term + (i+1)*math.pow(item[i],4) + np.random.uniform(0, 1)
    return sum_term


# Rastrigin Function
def F_07(x):
    sum_term = 0
    for item in x:
        for i in range(0, len(item)):
            sum_term = sum_term + item[i]**2 - 10*math.cos(2*math.pi*item[i])
    return 10*len(item) + sum_term



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
