#main.py
import numpy as np
import sys
from numpy import random
import p1_algorithm as algo
import p1_function as func
from p1_function import OptimaMapping






lower_bound = -10
upper_bound = 10
bounds = [lower_bound,upper_bound]


# def DE_solve(f, population_size, bounds, iteration,vec_scaller_f,Cr):
print("Function 08: Rosenbrock Function")
print("Optima: ", OptimaMapping.F_01)
print(algo.DE_solve(func.F_08,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_08, 100, bounds, 200,0.4,0.8))
print(algo.KH_solve(func.F_08, 100, bounds, 10, 6, 3, 40 ))

print("Function 09: Schwefel 2.26 Function")
print(algo.DE_solve(func.F_09,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_09, 100, bounds, 200,0.4,0.8))

print("Function 10: Schwefel 1.2")
print(algo.DE_solve(func.F_10,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_10, 100, bounds, 200,0.4,0.8))

print("Function 11: Schwefel 2.22")
print(algo.DE_solve(func.F_11,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_11, 100, bounds, 200,0.4,0.8))

print("Function 12: Schwefel 2.21")
print(algo.DE_solve(func.F_12,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_12, 100, bounds, 200,0.4,0.8))

print("Function 13: Sphere")
print(algo.DE_solve(func.F_13,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_13, 100, bounds, 200,0.4,0.8))

print("Function 14: Step")
print(algo.DE_solve(func.F_14,100,bounds,200,0.4,0.8))
print(algo.PSO_solve(func.F_14, 100, bounds, 200,0.4,0.8))