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
# print(algo.DE_solve(cost_function,100,bounds,200,0.4,0.8))
# print(algo.PSO_solve(cost_function, 100, bounds, 200,0.4,0.8))
# print(algo.KH_solve(cost_function, 100, bounds, 200,0.4,0.8,0.8))


 # ==========================================function 08 ---------------------------------------------------
lower_bound = -2
upper_bound = 2
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 10
cost_function = func.F_08
Function_name = "Function 08: Rosenbrock Function"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()



 # ==========================================function 08 ---------------------------------------------------
lower_bound = -512
upper_bound = 512
bounds = [lower_bound,upper_bound]
iteration_test = 200
population = 20
cost_function = func.F_09
Function_name = "Function 09: Schwefel 2.26"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()



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