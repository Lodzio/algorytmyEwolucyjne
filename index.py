import numpy as np
import sys
import time
from math import e
from random import choice
import matplotlib.pyplot as plt

def target(X):
    firstOptimum = e**(-((X - 2)**2))
    secondOptimum = 2*e**(-((X - 4.2)**2))
    return firstOptimum + secondOptimum

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
genotypes = 1
lowestValue = 0
highestValue = 0.3

population = (np.random.rand(populationSize, genotypes)*(highestValue - lowestValue))+lowestValue
sigma = 0.4
F = 0.4
targetValue1 = target(2)
targetValue2 = target(4.2)

while True:
    max = target(population).max()
    val1 = "{:.2f}".format(targetValue1 - max)
    val2 = "{:.2f}".format(targetValue2 - max)
    print(val1, val2)
    time.sleep(0.1)
    population = proportionalSelection(population)
    population = ES1plus1(population, sigma)
    population = BEST(population, F)
#     plt.scatter(population, target(population), marker='.')
#     plt.pause(0.1)

