#!/usr/bin/env python
#encoding:UTF-8
import random
from population_init.population_init import population_init
from function.eval_fun import eval_fun
from selection.selection import selection
from crossover.crossover import crossover
from mutation.mutation import mutation

N=4
V=3
minRealVal=(-1, -1, -1)
maxRealVal=(1, 1, 1)
population=[]
pcross_real=0.7
pmut_real=0.1
eta_c=1
eta_m=1

#目标函数值列表
xReal=[]

def per_run():
    population_init(population, N, V, minRealVal, maxRealVal)
    print('population=', population)

    for i in range(2):
        eval_fun(population, xReal)
        print('1_eval_population=', population)
        print('1_xReal=', xReal)

        selection(population, xReal)
        print('selected_population=', population)
        print('selected_xReal=', xReal)
        crossover(population, pcross_real, V, minRealVal, maxRealVal, eta_c)
        print('crossover_population=', population)
        mutation(population, pmut_real, V, minRealVal, maxRealVal, eta_m)
        print('mutation_population=', population)

    eval_fun(population, xReal)
    print('2_eval_population=', population)
    print('2_xReal=', xReal)
    return 100.0-max(xReal)

if __name__=="__main__":
    score_list=[]
    for i in range(2):
        temp=per_run()
        score_list.append(temp)
        print(i, " : ", temp)
    print(sum(score_list)/len(score_list))
