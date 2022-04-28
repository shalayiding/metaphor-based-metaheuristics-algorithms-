#main.py
import numpy as np
import sys
from numpy import random
import p1_algorithm as algo
import p1_function as func






lower_bound = -10
upper_bound = 10
bounds = [lower_bound,upper_bound]


# def DE_solve(f, population_size, bounds, iteration,vec_scaller_f,Cr):
print(algo.DE_solve(func.F_08,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_08, 100, bounds, 200,0.4,0.8))
print(algo.KH_solve(func.F_08, 100, 10, 6, 3, 40 ))


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