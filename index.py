import numpy as np
import sys
import time
from math import e

def target(X):
    firstOptimum = e**(-((X - 2)**2))
    secondOptimum = 2*e**(-((X - 4.2)**2))
    return firstOptimum + secondOptimum
#     return epsilon**(X-1)+2*epsilon**(X-2)

def ES1plus1(X, sigma):
    return X + sigma*np.random.normal(0, 1, size=X.shape)

def roulette(probability):
    sum = 0
    targetValue = np.random.rand()
    for index, prob in enumerate(probability):
        sum += prob
        if targetValue < sum:
            return index

def proportionalSelection(population):
    result = np.zeros(shape=population.shape)
    quality = target(population)
    sum = quality.sum()
    probability = quality/sum
    for i in range(population.shape[0]):
        selectedParentIndex = roulette(probability)
        result[i] = population[selectedParentIndex]
    return result

populationSize = 10
genotypes = 1
lowestValue = 0
highestValue = 0.3

population = (np.random.rand(populationSize, genotypes)*(highestValue - lowestValue))+lowestValue
sigma = 0.1
targetValue1 = target(2)
targetValue2 = target(4.2)
while True:
    max = target(population).max()
    val1 = "{:.2f}".format(targetValue1 - max)
    val2 = "{:.2f}".format(targetValue2 - max)
    print(val1, val2)
    time.sleep(0.01)
    parents = proportionalSelection(population)
    population = ES1plus1(parents, sigma)
