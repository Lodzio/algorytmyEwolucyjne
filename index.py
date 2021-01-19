import numpy as np
import sys
import time
from math import e
from random import choice
import functools
import operator
import matplotlib.pyplot as plt

def createOptimum(Xopt, K=1):
    return  lambda X : K*e**functools.reduce(operator.add, [-((X[:,i] - x)**2) for i, x in enumerate(Xopt)])

def createTarget():
    firstOptimum = createOptimum([2, 4])
    secondOptimum= createOptimum([4.2, 2], 2)
    return lambda X: firstOptimum(X) + secondOptimum(X)

def ES1plus1(X, sigma):
    return X + sigma*np.random.normal(0, 1, size=X.shape)

def BEST(X, F):
    children = np.zeros(shape=X.shape)
    bestIndex = target(X).argmax()
    best = X[bestIndex]
    for index, _ in enumerate(X):
        if index == bestIndex:
            children[index] = best
            continue

        i1 = choice([i for i in range(0,9) if i not in [bestIndex, index]])
        i2 = choice([i for i in range(0,9) if i not in [bestIndex, i1, index]])
        children[index] = best + F*(X[i1] - X[i2])
    return children

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
genotypes = 2
lowestValue = 0
highestValue = 0.3
sigma = 0.4

population = (np.random.rand(populationSize, genotypes)*(highestValue - lowestValue))+lowestValue
F = 0.4

target = createTarget()
targetValue1, targetValue2 = target(np.array([[2, 4], [4.2, 2]]))

while True:
    max = target(population).max()
    val1 = "{:.2f}".format(targetValue1 - max)
    val2 = "{:.2f}".format(targetValue2 - max)
    print(val1, val2)
    time.sleep(0.01)
    population = proportionalSelection(population)
    population = ES1plus1(population, sigma)
    population = BEST(population, F)
#     plt.scatter(population, target(population), marker='.')
#     plt.pause(0.1)

