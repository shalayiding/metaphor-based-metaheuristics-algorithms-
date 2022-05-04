#main.py
from matplotlib import markers
import numpy as np
import sys
from numpy import random
import p1_algorithm as algo
import p1_function as func
from p1_function import OptimaMapping
import matplotlib.pyplot as plt





 # ==========================================function 01 ---------------------------------------------------
lower_bound = -10
upper_bound = 10
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 10
cost_function = func.F_01
Function_name = "Function 01: Ackley function"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()



 # ==========================================function 03 ---------------------------------------------------
lower_bound = -100
upper_bound = 100
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 10
cost_function = func.F_03
Function_name = "Function 03: Griewank function"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()



 # ==========================================function 06 ---------------------------------------------------
lower_bound = -100
upper_bound = 100
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 10
cost_function = func.F_06
Function_name = "Function 06: Quartic function"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()






 # ==========================================function 07 ---------------------------------------------------
lower_bound = -5.12
upper_bound = 5.12
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 10
cost_function = func.F_07
Function_name = "Function 07: Rastrigin Function"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()



 # ==========================================function 08 ---------------------------------------------------
lower_bound = -2
upper_bound = 2
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 4
cost_function = func.F_08
Function_name = "Function 08: Rosenbrock Function"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()


 # ==========================================function 09 ---------------------------------------------------
lower_bound = -512
upper_bound = 512
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 4
cost_function = func.F_09
Function_name = "Function 09: Schwefel 2.26"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()




 # ==========================================function 10 ---------------------------------------------------
lower_bound = -100
upper_bound = 100
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 4
cost_function = func.F_10
Function_name = "Function 10: Schwefel 1.2"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()




 # ==========================================function 11 ---------------------------------------------------
lower_bound = -100
upper_bound = 100
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 4
cost_function = func.F_11
Function_name = "Function 11: Schwefel 2.22"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()


 # ==========================================function 12 ---------------------------------------------------
lower_bound = -1000
upper_bound = 1000
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 4
cost_function = func.F_12
Function_name = "Function 12: Schwefel  2.21"


orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()



 # ==========================================function 13 ---------------------------------------------------
lower_bound = -1000
upper_bound = 1000
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 4
cost_function = func.F_13
Function_name = "Function 13: Sphere"
orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()


 # ==========================================function 14 ---------------------------------------------------
lower_bound = -1000
upper_bound = 1000
bounds = [lower_bound,upper_bound]
iteration_test = 80
population = 4
cost_function = func.F_14
Function_name = "Function 14: Step"
orginal_population = algo.generate_random_population(cost_function,population,bounds)
x = list(range(0,iteration_test))
best_solution,best_eva,mat_solution = algo.PSO_solve(orginal_population,cost_function, population, bounds, iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='r',marker= "o",label='PSO')
best_solution,best_eva,mat_solution = algo.KH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08)
plt.plot(x,[a[1] for a in mat_solution],c='k',marker= "x",label='KH')
best_solution,best_eva,mat_solution = algo.MKH_solve(orginal_population,cost_function, population, bounds, iteration_test,0.02,0.08,0.08,0.5,0.4)
plt.plot(x,[a[1] for a in mat_solution],c='g',marker= "*",label='BKH')
best_solution,best_eva,mat_solution = algo.DE_solve(orginal_population,cost_function,population,bounds,iteration_test,0.4,0.8)
plt.plot(x,[a[1] for a in mat_solution],c='b',marker= "v",label='DE')
plt.legend(loc="upper right")
plt.title(Function_name)
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
print(Function_name + "DONE")
plt.savefig(Function_name+".png")
plt.clf()

