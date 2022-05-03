#main.py
from matplotlib import markers
import numpy as np
import sys
from numpy import random
import p1_algorithm as algo
import p1_function as func
from p1_function import OptimaMapping
import matplotlib.pyplot as plt








# print(algo.DE_solve(func.F_01,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_01, 100, bounds, 200,0.4,0.8))
# print(algo.KH_solve(func.F_01,100,bounds,200,0.4,0.8,1))


# # print(algo.DE_solve(func.F_02,100,bounds,200,0.4,0.8))
# # print(algo.PSO_solve(func.F_02, 100, bounds, 200,0.4,0.8))

# print(algo.DE_solve(func.F_03,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_03, 100, bounds, 200,0.4,0.8))

# print(algo.DE_solve(func.F_06,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_06, 100, bounds, 200,0.4,0.8))


# print(algo.DE_solve(func.F_07,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_07, 100, bounds, 200,0.4,0.8))




# def DE_solve(f, population_size, bounds, iteration,vec_scaller_f,Cr):
# print("Function 08: Rosenbrock Function")
# print("Optima: ", OptimaMapping.F_01)
# print(algo.DE_solve(func.F_08,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_08, 100, bounds, 200,0.4,0.8))
# print(algo.KH_solve(func.F_08, 100, bounds, 200,0.4,0.8,0.8))


# lower_bound = -2
# upper_bound = 2
# bounds = [lower_bound,upper_bound]
# iteration_test = 100
# population = 50
# print("Function 08: Rosenbrock Function")
# x = list(range(0,iteration_test))
# best_solution,best_eva,mat_solution = algo.DE_solve(func.F_08,population,bounds,iteration_test,0.4,0.8)
# plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
# best_solution,best_eva,mat_solution = algo.PSO_solve(func.F_08, population, bounds, iteration_test,0.4,0.8)
# plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
# best_solution,best_eva,mat_solution = algo.KH_solve(func.F_08, population, bounds, iteration_test,0.4,0.8,0.8)
# plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
# plt.title("Function 08: Rosenbrock Function")
# plt.xlabel("Iteration")
# plt.ylabel("Cost function value")
# plt.savefig("Function_08_result.png")



lower_bound = -512
upper_bound = 512
bounds = [lower_bound,upper_bound]
iteration_test = 100
population = 50
print("Function 09: Schwefel 2.26 function")
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.DE_solve(func.F_09,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
best_solution,best_eva,mat_solution = algo.PSO_solve(func.F_09, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(func.F_09, population, bounds, iteration_test,0.4,0.8,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
plt.title("Function 09: Schwefel 2.26 function")
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
plt.savefig("Function_09_result.png")



# print("Function 10: Schwefel 1.2")
# print(algo.DE_solve(func.F_10,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_10, 100, bounds, 200,0.4,0.8))

# print("Function 11: Schwefel 2.22")
# print(algo.DE_solve(func.F_11,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_11, 100, bounds, 200,0.4,0.8))

# print("Function 12: Schwefel 2.21")
# print(algo.DE_solve(func.F_12,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_12, 100, bounds, 200,0.4,0.8))

# print("Function 13: Sphere")
# print(algo.DE_solve(func.F_13,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_13, 100, bounds, 200,0.4,0.8))

# print("Function 14: Step")
# print(algo.DE_solve(func.F_14,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(func.F_14, 100, bounds, 200,0.4,0.8))