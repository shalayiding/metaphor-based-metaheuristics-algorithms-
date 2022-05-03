import numpy as np
import sys
from numpy import random
from sklearn.decomposition import randomized_svd


def generate_random_population(f, population_size, bounds):
    population = []
    population_eva = []
    for i in range(population_size):  # generate number of population and evaluate them
        population.append(
            np.array([bounds[0] + random.randn(len(bounds)) * (bounds[1] - bounds[0])]))
        population_eva.append(f(population[i]))
    best_j = np.argmin(population_eva)  # find minimum
    return population, population_eva, best_j


# ----------------------------------DE------------------------------
D = 2
# return new vector accecpt scaller F and 3 different vector


def mutation(three_vector, F):  # mutation process
    v1, v2, v3 = three_vector  # 3 vector expect current is pass to function
    # use the formula to calcualte the new mutated vector
    return np.array([np.array(v1[0]) + F*(np.array(v2[0])-np.array(v3[0]))])

# clip the number to bounds


def clip_bound(vec, bounds):
    low_bound, up_bound = bounds
    for i in range(len(vec[0])):
        if vec[0][i] > up_bound:
            vec[0][i] = up_bound
        elif vec[0][i] < low_bound:
            vec[0][i] = low_bound
        else:
            continue

    return 0

# crossover take 2 vector and rate cr


def crossover(mutated_vec, current_vec, Cr):
    r = random.rand(D)  # D random number
    result = []
    for i in range(len(mutated_vec[0])):
        if r[i] < Cr:
            result.append(mutated_vec[0][i])
        else:
            result.append(current_vec[0][i])
    return [result]
# this fucntion is regular de return best candidate and it is evaluation score
# it takes cost function, population size , bounds, iteration , scaller f, and Cr


def DE_solve(f, population_size, bounds, iteration, vec_scaller_f, Cr):
    population, population_eva, best_idx = generate_random_population(
        f, population_size, bounds)
    best_candidate = population[best_idx]
    best_candidate_eva = f(best_candidate)
    mat_solutions = []
    for i in range(iteration):  # until givin iteration
        
        for j in range(population_size):
            # select 3 vector other than j
            indexs_without_j = [indexs_without_j for indexs_without_j in range(
                population_size) if indexs_without_j != j]
            # print(population[random.choice(indexs_without_j, 3, replace=False)])
            v1, v2, v3 = random.choice(indexs_without_j, 3, replace=False)
            Three_vector = [population[v1], population[v2], population[v3]]
            mutated_candidate = mutation(
                Three_vector, vec_scaller_f)  # generate new vector
            # clip it if it is out of bounds
            clip_bound(mutated_candidate, bounds)
            cross_vec = crossover(mutated_candidate, population[j], Cr)
            if f(cross_vec) < f(population[j]):
                population[j] = cross_vec
                population_eva[j] = f(cross_vec)
            if best_candidate_eva > f(population[j]):
                best_candidate_eva = f(population[j])
                best_candidate = population[j]
        mat_solutions.append([best_candidate, best_candidate_eva])
    return best_candidate, best_candidate_eva,mat_solutions
# ----------------------------------END DE-------------------------------


# -----------------------------------PSO -------------------------------------
 # use formula to generate v1
def Generate_new_velocity(curr_velocity, alpha, beta, curr_pos, global_pos, curr_best):
    r1 = np.array(random.rand(1, 2))
    r2 = np.array(random.rand(1, 2))
    new_velocity = curr_velocity + alpha*r1*(np.array(global_pos) - np.array(
        curr_pos)) + beta*r2*(np.array(curr_best) - np.array(curr_pos))
    return new_velocity
# this function is regular particle swarm optimization
# takes paricles_size, bounds, ietration, alpha and beta value


def PSO_solve(f, particles_size, bounds, iteration, alpha, beta):

    particles = []
    velocity = []
    particles_eva = []
    for i in range(particles_size):  # generate number of population and evaluate them
        pos = np.array([bounds[0] + random.rand(len(bounds))
                       * (bounds[1] - bounds[0])])
        clip_bound(pos, bounds)
        particles.append([pos, pos])
        velocity.append(0)
        particles_eva.append(f(pos))
    best_j = np.argmin(particles_eva)  # find minimum
    best_candidate = particles[best_j][0]
    best_candidate_eva = particles_eva[best_j]

    mat_solutions = []
    for i in range(iteration):
        
        for j in range(particles_size):
            velocity[j] = Generate_new_velocity(
                velocity[j], alpha, beta, particles[j][0], best_candidate, particles[j][1])  # new v1
            particles[j][0] = np.array(
                particles[j][0]) + np.array(velocity[j])  # get new position
            clip_bound(particles[j][0], bounds)
            particles_eva[j] = f(particles[j][0])
            if f(particles[j][0]) < f(particles[j][1]):  # check if the new pos is better
                particles[j][1] = particles[j][0]
            if best_candidate_eva > f(particles[j][1]):
                best_candidate_eva = f(particles[j][1])
                best_candidate = particles[j][1]
        mat_solutions.append([best_candidate, best_candidate_eva])
    return best_candidate, best_candidate_eva,mat_solutions

# -----------------------------------END PSO--------------------------------------


# -----------------------------------Krill Herd -------------------------------------

def KH_solve(f, population_size, bounds, generation_counter, Vf, Dmax, Nmax):
    # randomly generate population, initialize location of krill
    population, population_eva, best_idx = generate_random_population(
        f, population_size, bounds)
    best_candidate = population[best_idx]
    best_candidate_eva = f(best_candidate)
    Ni = 1
    Fi = 1
    weight = 0.4  # in range of 0 to 1
    rand_num = 0.3
    Ct = 0.8
    mat_solutions = []
    for i in range(0, generation_counter):
        population.sort(key=f)
        population_eva = [f(population[e]) for e in range(len(population))]
        best_candidate = population[0]
        best_candidate_eva = population_eva[0]
        mat_solutions.append([best_candidate, best_candidate_eva])
        for j in range(0, len(population)):
            #   Motion induced by other krill individuals ----------------------------
            Xj = []
            Kj = []
            Alocal = 0
            for c in range(0, len(population)):
                if j != c:
                    Xjtmp = (population[c] - population[j]) / \
                        (abs(population[c] - population[j])+random.rand(1))
                    Xj.append(Xjtmp)
                    Kjtmp = (population_eva[j] - population_eva[c]) / \
                        (min(population_eva) - max(population_eva))
                    Kj.append(Kjtmp) 
            Alocal = sum([Xj[e] * Kj[e] for e in range(len(Xj))])  # equation 4
            Cbest = 2*(random.rand(1) + i/generation_counter)
            Kibest = min(Kj)
            Xibest = Xj[Kj.index(Kibest)]
            Atarget = Cbest*Kibest*Xibest  
            Ai = Alocal + Atarget     # equation 3
            Ni = Nmax*Ai + weight*Ni    #equation 2
            
            #  Foraging motion ----------------------------
            Xfood = sum([population[e]/population_eva[e] for e in range(len(population_eva))])/\
                sum([1/population_eva[e] for e in range(len(population_eva))]) # equeation 12
            Cfood = 2*(1-i/generation_counter) # equeation 14
            
            Kifood = (Xfood - population_eva[j])/(min(population_eva) - max(population_eva))
            Xifood = (Xfood - population[j])/(abs(population[j] - Xfood)+random.rand(1))
            Bibest = Kibest*Xibest # equeation 15
            Bifood = Cfood*Kifood*Xifood # equeation 13

            Bi = Bifood + Bibest #  equeation 11
            Fi = Vf*Bi + weight*Fi # equation 10



            #physical diffusion ----------------------------
            Di = Dmax* (1-i/generation_counter)*random.uniform(-1,1,2)
            dXi_dt = Ni + Fi + Di # equation 1
            deltat = Ct*(bounds[1] - bounds[0])  # equation 19
            Newposition = population[j] + deltat*dXi_dt # equation 18
            # cross over 
            Cr = 0.2*Kibest # equation 21
            Newposition = crossover(Newposition, population[j], Cr) # equation 20

            
            #mutation around the global best 
            Mu = 0.05/Kibest# equation 23
            if random.rand(1) < Mu:
                indexs_without_j = [indexs_without_j for indexs_without_j in range(
                population_size) if indexs_without_j != j]
                v1,v2,v3 = random.choice(indexs_without_j, 3, replace=False)
                Three_vector = [best_candidate, population[j], population[v3]]
                Newposition = mutation(
                    Three_vector, random.rand(1))  # generate new vector
            

            
            
            clip_bound(Newposition, bounds)
            if f(Newposition) < f(population[j]):
                population[j] = np.array(Newposition)
                population_eva[j] = f(population[j])
                # print(population_eva[j])
                # print(population)
                # print(best_candidate_eva)

            # updat the positon of the krill
    # calculate fitness of each krill
    # while max generation # isn't hit
        # sort the population by fitness
        # for each krill
            # calculate: 1) motion induced by other individuals
            # ...2) foraging motion
            # ...3) physical diffusion
            # implement genetic operations
            # update locations of krill
            # calculate fitness for each krill in new positions
        # increment generation counter
    return best_candidate, best_candidate_eva,mat_solutions


# -----------------------------------END Krill Herd--------------------------------------


# Static function take cost function and iterate over 30 trails
# with given set of step_size and Tzero array to output the min, max and means,standard deviation value
def static(cost_function, solver, test_population_size, bounds, test_p1, test_p2, itration_time):
    for i in range(len(test_population_size)):
        for j in range(len(test_p1)):
            best_set = []
            best_eva_set = []
            for t in range(0, 30):
                best, best_eva = solver(
                    cost_function, test_population_size[i], bounds, itration_time, test_p1[j], test_p2[j])
                best_set.append(best)
                best_eva_set.append(best_eva)
            print("population size is :", test_population_size[i])
            print("parameter 1 size is :", test_p1[j])
            print("parameter 2 size is :", test_p2[j])
            print("Mean=", f"{np.sum(best_eva_set)/len(best_eva_set):.9f}")
            print("Std=", f"{np.std(best_eva_set):.9f}")
            print("Min=", f"{np.min(best_eva_set):.9f}")
