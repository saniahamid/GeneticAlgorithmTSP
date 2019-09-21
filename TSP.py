# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:30:29 2018

@author: sania
"""

import array
import random
import numpy
import pandas as pd
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


df = pd.read_csv('eil51.CSV')

data = df.as_matrix()


tsp_data = dict()

for x in range(0,51):

    tsp_data[str(x)] = (data[x][1],data[x][2])
        
IND_SIZE = 51

#creating individuals
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
stats = tools.Statistics(key = lambda ind: ind.fitness.values)

    
def evaluateTSP(individual):
    summation = 0
    start = individual[0]
    for i in range (1,len(individual)):
        end = individual[i]
        a = tsp_data[str(start)]
        b = tsp_data[str(end)]

        summation += math.sqrt(((b[0]-a[0]))**2+((b[1]-a[1]))**2)
        start = end
    #print([summation])
    return [summation]

#def printBestTour(individual):
    #print(hof[0])

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes,indpb = 0.05)
toolbox.register("select",tools.selTournament,tournsize = 3)
toolbox.register("evaluate",evaluateTSP)
stats.register("AvgTourLength",numpy.mean)
stats.register("BestTourLength",numpy.min)
#stats.register("Best Tour",printBestTour)

def main():
    random.seed(169)
    
    pop = toolbox.population(n = 3000)
    
    hof = tools.HallOfFame(1)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 350, stats = stats, halloffame=hof, verbose = True, )
    #algorithms.eaSimple(pop, toolbox, 0.7, 0.03, 3000)
    #eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen[, stats, halloffame, verbose])
    #algorithms.eaMuCommaLambda(pop, toolbox, 2000, 3000, 0.8, 0.02,1000,halloffame=hof) #523
    #algorithms.eaMuCommaLambda(pop, toolbox, 2000, 3000, 0.7, 0.2,100,halloffame=hof)
    #algorithms.eaMuCommaLambda(pop, toolbox, 2000, 3000, 0.7, 0.01,1000,halloffame=hof)
    print("The best tour is:")
    print(hof)
    return pop, hof

#if __name__ == "__main__":
   # main()
    
pop, hof = main()