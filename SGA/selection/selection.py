#!/usr/bin/env python
#encoding:UTF-8
import copy
import random
#轮盘赌选择法
def selection(population, current_fitness):
    s=sum(current_fitness)
    temp=[k*1.0/s for k in current_fitness]
    temp2=[]

    s2=0
    for k in temp:
        s2=s2+k
        temp2.append(s2)

    temp3=[]
    for _ in range(len(population)):
        r=random.random()
        for i in range(len(temp2)):
            if r<=temp2[i]:
                temp3.append(i)
                break

    temp4=[]
    temp5=[]
    for i in temp3:
        temp4.append(copy.deepcopy(population[i]))
        temp5.append(current_fitness[i])
    population[:]=temp4
    current_fitness[:]=temp5

