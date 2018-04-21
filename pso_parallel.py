"""
Particle Swarm Optimisation with improvements - Jie Jenny Yan, January 2018
- Framework: DEAP, Scoop
- Fitness function: Optimisation Test Functions
- Particle Attributes: velocity, best_known, curr_pos
- Constants: upper/lower bounds (b_u, b_l: generate function, random.uniform),
             inertia weighting (w: swarm evolution function -> velocity equation),
             accel coefficients (phi_p, phi_g: swarm evolution function -> velocity equation),
             diversify search (r_p, r_g: swarm evolution function, velocity equation),

PSO Algorithm:

1) swarm initialisation
    - for each particle in the swarm, initialise the position from a uniform
      distribution with b_l and b_u (tbc)
    - find global best position while initialising the swarm
    - sample the velocity per particle from a uniform distribution

2) swarm evolution
    - for each particle in the swarm:
        - sample the r_p, r_g values from uniform distribution(0,1)
        - with w = phi_p = phi_g = 0.5, calculate the new velocity with:

          v = w * v + phi_p * r_p * (best_known - curr_pos) + phi_g * r_g * (glob_best - curr_pos)

          *NOTE: consider tuning these scaling values

        - update position
        - if fitness new position better than fitness of best position,
            - update best position
            - if best pos better than global best,
                - update global best
                - check <termination conditions>
    - update generation
    - update prev best

3) termination conditions
    - exceeded max generations

4) return global best

Here, we talk about the global best as being the particle that has the best position
or in this case, the best output value. Best known refers to the best particle position
that has been seen by a particular particle.

NOTE: there were some issues with using numpy.sum potentially problems with using
reduce

"""
import numpy as np
import time

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

from scoop import futures

from evaluation_functions import * 
from deep import keras_capos_input_pei

GMAX = 5               # Max number of generations
DELTA = 1e-7           # Smallest position increment allowed
EPSILON = 1e-7         # Smallest fitness value increment allowed
DIM = 2                # No. of Dimensions in the problem
POPULATION = 10        # Size of the particle Swarm

# --------------------------Swarm operations ---------------------------------
# Creates a fitness object that maximises its fitness value
creator.create("Fitness", base.Fitness, weights=(1.0,))

# Creates a particle with initial declaration of its contained attributes
creator.create("Particle", np.ndarray, fitness=creator.Fitness, velocity=np.ndarray(DIM), best_known=None)

# generates and returns a particle based on the dim (size) of the problem
def generate(bound_l, bound_u):
    particle = creator.Particle(np.random.uniform(bound_l,bound_u) for _ in range(DIM))
    bound = bound_u - bound_l
    particle.velocity = np.array([np.random.uniform(-abs(bound), abs(bound)) for _ in range(DIM)])
    particle.best_known = creator.Particle(particle)
    return particle

# updating the velocity and position of the particle
def updateParticle(particle, best, generator, w, phi_p, phi_g):

    r_p = np.array([generator.uniform(0,1) for _ in particle])
    r_g = np.array([generator.uniform(0,1) for _ in particle])
    #NOTE: random seed is the same in each process, so in parallel, the performance is poor

    p = np.subtract(particle.best_known, particle)
    g = np.subtract(best, particle)

    v_p = phi_p * np.multiply(p, r_p)
    v_g = phi_g * np.multiply(g, r_g)

    v_w = w * particle.velocity
    particle.velocity[:] = np.add(v_w, np.add(v_p, v_g))
    particle[:] = np.add(particle, particle.velocity)

# initialise the swarm with fitness values.
def initialiseSwarm(particle):

    # assigning the fitness values and initialising best known position
    particle.fitness.values = toolbox.evaluate(particle)
    particle.best_known.fitness.values = particle.fitness.values

    return particle.fitness.values

# updates a particle with new values
def updateSwarm(particle, best, generator):

    # move the particles with the update function and eval new fitness
    toolbox.update(particle, best, generator)
    particle.fitness.values = toolbox.evaluate(particle)

    # if the particle output is better than the best known output, update the
    # best known for this particle
    if particle.fitness.values > particle.best_known.fitness.values:

        particle.best_known = creator.Particle(particle)
        particle.best_known.fitness.values = particle.fitness.values

        # if the particle best known happens to be better than the global best
        # then also update the global best value 
        if particle.best_known.fitness.values > best.fitness.values:

            best = creator.Particle(particle.best_known)
            best.fitness.values = particle.best_known.fitness.values

    time.sleep(5)    # simulating long computation time
    return best

def createBest(best):
    particle = creator.Particle(best)
    particle.fitness.values = best.fitness.values
    return particle
# ---------------------- toolbox -------------------------------------
# registering all the functions to the toolbox for more convenient access
# by assignment default parameter values. To use these functions call
# toolbox.<functionname>
toolbox = base.Toolbox()

# deep evaluations function has default arguments defined in the deep.py file 
# The arguments that can be changed are: 
#      num_classes,
#      num_rows,
#      num_cols,
#      num_channels,
#      num_epochs,
#      num_conv_filters,
#      kernel_size,
#      pooling_size,
#      dropout_rate,
#      num_dense_out
toolbox.register("evaluate", keras_capos_input_pei)
toolbox.register("particle", generate, bound_l=-5, bound_u=5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi_p=0.8, phi_g=0.8, w=0.8)
toolbox.register("map", futures.map)

# ---------------------------------------------------------------------
def main():

    # initialise the swarm of particles with their fitness values
    pop = toolbox.population(n=POPULATION)
    fitness = list(map(initialiseSwarm, pop))

    # initialise the global best particle in the swarm
    best = creator.Particle(pop[fitness.index(max(fitness))])
    best.fitness.values = pop[fitness.index(max(fitness))].fitness.values

    # create a list of global best particles so that we can map each particle
    # to a global best value when updating the particle positions. At the same time,
    # create a list of random generators so that each process runs with a different seed
    # as this also impacts the performance of the algorithm
    global_best = []
    random_generators = []
    for _ in range(POPULATION):
        global_best.append(createBest(best))
        random_generators.append(np.random.RandomState())

    g = 1
    while g <= GMAX:

        # update the particles in the swarm and return a list of the new global
        # best particles
        global_best = list(toolbox.map(updateSwarm, pop, global_best, random_generators))

        # calculate the best global best from the list and create a new global
        # best list
        best = creator.Particle(global_best[0])
        best.fitness.values = global_best[0].fitness.values
        for n in range(1, POPULATION):
            if global_best[n].fitness.values > best.fitness.values:
                best = creator.Particle(global_best[n])
                best.fitness.values = global_best[n].fitness.values

        global_best = []
        for _ in range(POPULATION):
            global_best.append(createBest(best))

        # iterate the generation
        g = g + 1

    return pop, best

if __name__ == "__main__":
    print(main())
