#!/usr/bin/env python
#encoding:UTF-8
import random

#种群初始化函数
def population_init(population, N, V, minRealVal, maxRealVal):
    del population[:]

    rangeRealVal=[maxRealVal[i]-minRealVal[i] for i in range(V)]

    for i in range(N):
        tempIndividual=[]
        for j in range(V):
            temp=random.uniform(0, 1)*rangeRealVal[j]+minRealVal[j]
            tempIndividual.append(temp)
        population.append(tempIndividual)



