import numpy as np
import sys
import time
from math import e
from random import choice
import functools
import operator
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

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

def run(q, populationSize, genotypes, startPoint, randRange, F, sigma):
    population = startPoint + ((np.random.rand(populationSize, genotypes)*(2*randRange))-randRange)
    steps = 0
    while True:
        max = target(population).max()
        val1 = "{:.2f}".format(targetValue1 - max)
        val2 = "{:.2f}".format(targetValue2 - max)
        if targetValue2 - max < 0.01:
            return q.put(steps)
        elif steps >= 200:
            return q.put(steps)
        steps += 1
#         print(val1, val2)
        population = proportionalSelection(population)
        population = ES1plus1(population, sigma)
        population = BEST(population, F)

populationSize = 10
genotypes = 2
startPoint = [0, 0]
randRange = 0.3
sigma = 0.2
F = 0.4

target = createTarget()
targetValue1, targetValue2 = target(np.array([[2, 4], [4.2, 2]]))

attempts = 1000
sigmas = np.arange(0.1, 0.7, 0.05).tolist()
times = []

# print(run(populationSize, genotypes, startPoint, randRange, F, sigma))

for xindex, sigma in enumerate(sigmas):
    sum = 0
    processes = []
    q = Queue()
    for i in range(attempts):
        p = Process(target=run, args=(q, populationSize, genotypes, startPoint, randRange, F, sigma))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        sum+=q.get()
    print(xindex)
    times.append(sum/attempts)
#     plt.scatter(population, target(population), marker='.')
#     plt.pause(0.1)

print(times)