import numpy as np
import sys
from numpy import random



#----------------------------------DE------------------------------
D = 2
# return new vector accecpt scaller F and 3 different vector
def mutation(three_vector, F): # mutation process 
    v1,v2,v3 = three_vector  # 3 vector expect current is pass to function 
    return np.array([np.array(v1[0]) +F*(np.array(v2[0])-np.array(v3[0]))]) # use the formula to calcualte the new mutated vector

# clip the number to bounds
def clip_bound(vec,bounds): 
    low_bound, up_bound = bounds
    for i in range(len(vec[0])):
        if  vec[0][i]> up_bound:
            vec[0][i] = up_bound
        elif vec[0][i] < low_bound:
            vec[0][i] = low_bound
        else :
            continue

    return 0

# crossover take 2 vector and rate cr 
def crossover(mutated_vec,current_vec,Cr):
    r = random.rand(D) # D random number
    result = []
    for i in range(len(mutated_vec[0])):
        if r[i] < Cr:
            result.append(mutated_vec[0][i])
        else :
            result.append(current_vec[0][i])
    return [result]
# this fucntion is regular de return best candidate and it is evaluation score
# it takes cost function, population size , bounds, iteration , scaller f, and Cr
def DE_solve(f, population_size, bounds, iteration,vec_scaller_f,Cr):
    population = []
    population_eva = []
    for i in range(population_size):    #generate number of population and evaluate them 
        population.append(
            np.array([bounds[0] + random.rand(len(bounds)) * (bounds[1] - bounds[0])]))
        population_eva.append(f(population[i]))
    best_j = np.argmin(population_eva) # find minimum 
    best_candidate = population[best_j] 
    best_candidate_eva = f(best_candidate)

    for i in range(iteration): # until givin iteration 
        for j in range(population_size):
            # select 3 vector other than j 
            indexs_without_j = [indexs_without_j for indexs_without_j in range(
                population_size) if indexs_without_j != j]
            # print(population[random.choice(indexs_without_j, 3, replace=False)])
            v1, v2, v3 = random.choice(indexs_without_j, 3, replace=False)
            Three_vector = [population[v1],population[v2],population[v3]] 
            mutated_candidate = mutation(Three_vector,vec_scaller_f) # generate new vector 
            clip_bound(mutated_candidate,bounds) # clip it if it is out of bounds
            cross_vec = crossover(mutated_candidate,population[j],Cr)
            if f(cross_vec) <f(population[j]):
                population[j] = cross_vec
                population_eva[j] = f(cross_vec)
            if  best_candidate_eva > f(population[j]):
                best_candidate_eva = f(population[j])
                best_candidate =population[j]
       
    return best_candidate, best_candidate_eva
#----------------------------------END DE-------------------------------







#-----------------------------------PSO -------------------------------------
 #use formula to generate v1 
def Generate_new_velocity(curr_velocity,alpha,beta,curr_pos,global_pos,curr_best):
    r1=np.array(random.rand(1,2))
    r2=np.array(random.rand(1,2))
    new_velocity = curr_velocity + alpha*r1*(np.array(global_pos) - np.array(curr_pos)) + beta*r2*(np.array(curr_best) -np.array(curr_pos))
    return new_velocity
# this function is regular particle swarm optimization 
# takes paricles_size, bounds, ietration, alpha and beta value 
def PSO_solve(f, particles_size, bounds, iteration,alpha,beta):
    
    particles = []
    velocity = []
    particles_eva = []
    for i in range(particles_size):    #generate number of population and evaluate them 
        pos = np.array([bounds[0] + random.rand(len(bounds)) * (bounds[1] - bounds[0])])
        clip_bound(pos,bounds)
        particles.append([pos,pos])
        velocity.append(0)
        particles_eva.append(f(pos))
    best_j = np.argmin(particles_eva) # find minimum 
    best_candidate = particles[best_j][0]
    best_candidate_eva = particles_eva[best_j]
    for i in range(iteration):
        for j in range(particles_size):     
            velocity[j] = Generate_new_velocity(velocity[j],alpha,beta,particles[j][0],best_candidate,particles[j][1]) # new v1
            particles[j][0] = np.array(particles[j][0]) + np.array(velocity[j]) # get new position 
            clip_bound(particles[j][0],bounds)
            particles_eva[j] = f(particles[j][0])
            if f(particles[j][0]) < f(particles[j][1]): # check if the new pos is better 
                particles[j][1] = particles[j][0]
            if  best_candidate_eva > f(particles[j][1]):
                best_candidate_eva = f(particles[j][1])
                best_candidate =particles[j][1] 
    return best_candidate,best_candidate_eva

#-----------------------------------END PSO--------------------------------------









# Static function take cost function and iterate over 30 trails 
# with given set of step_size and Tzero array to output the min, max and means,standard deviation value 
def static(cost_function,solver,test_population_size,bounds,test_p1,test_p2,itration_time):
    for i in range(len(test_population_size)):
        for j in range(len(test_p1)):
            best_set = []
            best_eva_set = [] 
            for t in range(0,30):
                best,best_eva = solver(cost_function,test_population_size[i],bounds,itration_time,test_p1[j],test_p2[j])
                best_set.append(best)
                best_eva_set.append(best_eva)
            print("population size is :",test_population_size[i])
            print("parameter 1 size is :",test_p1[j])
            print("parameter 2 size is :",test_p2[j])
            print("Mean=", f"{np.sum(best_eva_set)/len(best_eva_set):.9f}")
            print("Std=",f"{np.std(best_eva_set):.9f}")
            print("Min=",f"{np.min(best_eva_set):.9f}")




