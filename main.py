#main.py
import numpy as np
import sys
from numpy import random
import p1_algorithm as algo
import p1_function as func






lower_bound = -500
upper_bound = 500
bounds = [lower_bound,upper_bound]


print(algo.DE_solve(func.F_01,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_01, 100, bounds, 200,0.4,0.8))

# print(algo.DE_solve(func.F_02,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_02, 100, bounds, 200,0.4,0.8))

print(algo.DE_solve(func.F_03,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_03, 100, bounds, 200,0.4,0.8))

print(algo.DE_solve(func.F_06,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_06, 100, bounds, 200,0.4,0.8))


print(algo.DE_solve(func.F_07,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_07, 100, bounds, 200,0.4,0.8))




# def DE_solve(f, population_size, bounds, iteration,vec_scaller_f,Cr):
print(algo.DE_solve(func.F_08,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_08, 100, bounds, 200,0.4,0.8))


print(algo.DE_solve(func.F_09,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_09, 100, bounds, 200,0.4,0.8))

print(algo.DE_solve(func.F_10,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_10, 100, bounds, 200,0.4,0.8))


print(algo.DE_solve(func.F_11,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_11, 100, bounds, 200,0.4,0.8))

print(algo.DE_solve(func.F_12,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_12, 100, bounds, 200,0.4,0.8))


print(algo.DE_solve(func.F_13,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_13, 100, bounds, 200,0.4,0.8))

print(algo.DE_solve(func.F_14,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_14, 100, bounds, 200,0.4,0.8))