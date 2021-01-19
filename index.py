import numpy as np
import sys
import time
from math import e
from random import choice
import functools
import operator
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, current_process

def createOptimum(Xopt, K=1):
    return  lambda X : K*e**functools.reduce(operator.add, [-((X[:,i] - x)**2) for i, x in enumerate(Xopt)])

def createTarget():
    firstOptimum = createOptimum([2, 2])
    secondOptimum= createOptimum([4.2, 2], 2)
    return lambda X: firstOptimum(X) + secondOptimum(X)

def ES1plus1(X, sigma):
    return X + sigma*np.random.normal(0, 1, size=X.shape)

def BEST(X, F, target):
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

def rouletteSelection(population, target):
    result = np.zeros(shape=population.shape)
    quality = target(population)
    sum = quality.sum()
    probability = quality/sum
    for i in range(population.shape[0]):
        selectedParentIndex = roulette(probability)
        result[i] = population[selectedParentIndex]
    return result

def f(d, alpha, delta):
    if d<delta:
        return 1-((d/delta)**alpha)
    else:
         return 0

def fitnessSharing(population, target, alpha, delta):
    F = target(population)
    return F/[np.sum([f(np.linalg.norm(Y-X), alpha, delta) for Y in population]) for X in population]

def fitnessSharingForTests(x, population, target, alpha, delta):
    F = target(x)
    return F/[np.sum([f(np.linalg.norm(Y-X), alpha, delta) for Y in population]) for X in x]

def run(q, nativeTarget, populationSize, genotypes, startPoint, randRange, F, sigma):
    np.random.seed(current_process().pid)
    fig, ax = plt.subplots()
    target = lambda pop: fitnessSharing(pop, nativeTarget, 0.5, 3)
#     target = nativeTarget
    population = startPoint + ((np.random.rand(populationSize, genotypes)*(2*randRange))-randRange)
    fitnesses = []
    steps = 0
    while True:
        fitness = nativeTarget(population)
        fitnesses.append(fitness.mean())
        max = nativeTarget(population).max()
        val1 = "{:.2f}".format(targetValue1 - max)
        val2 = "{:.2f}".format(targetValue2 - max)
        if max > targetValue1*0.1:
            return q.put(steps)
        if steps >= 400:
            return q.put(steps)
        steps += 1
        population = BEST(population, F, target)
        population = rouletteSelection(population, target)
        population = ES1plus1(population, sigma)
#         ax.clear()
#         xTest = np.array([[i, 2] for i in np.arange(population[:, 0].min() - 2, population[:, 0].max() + 2, 2**-4)])
#         x = np.array([[i, 2] for i in np.arange(0, 6, 2**-4)])
#         changeFitness = lambda x: fitnessSharingForTests(x, population, nativeTarget, 0.5, 3)
#         ax.plot(x[:, 0], nativeTarget(x))
#         ax.plot(xTest[:, 0], changeFitness(xTest))
#         ax.scatter(population[:, 0], nativeTarget(population), marker='.')
#         plt.pause(1)
#     plt.plot(fitnesses)
#     plt.show()

populationSize = 10
genotypes = 2
startPoint = [0, 0]
randRange = 0.3

sigma = 0.5
F = 0.4

attempts = 100
sigmas = np.arange(0.1, 1, 0.05).tolist()
times = []

customTarget = createTarget()
targetValue1, targetValue2 = customTarget(np.array([[2, 4], [4.2, 2]]))

# q = Queue()
# run(q, customTarget, populationSize, genotypes, startPoint, randRange, F, sigma)
for xindex, sigma in enumerate(sigmas):
    sum = 0
    processes = []
    q = Queue()
    for i in range(attempts):
        p = Process(target=run, args=(q, customTarget, populationSize, genotypes, startPoint, randRange, F, sigma))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        sum+=q.get()
    print(xindex)
    times.append(sum/attempts)

# plt.plot(sigmas, times)
# plt.xlabel('sigma')
# plt.ylabel('steps')
# plt.show()
print(times)