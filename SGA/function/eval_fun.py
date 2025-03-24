#!/usr/bin/env python
#encoding:UTF-8
from .object_fun import object_fun

def eval_fun(population, xReal):
    del xReal[:]
    for i in population:
        # print('i=', i)
        xReal.append(object_fun(i))

