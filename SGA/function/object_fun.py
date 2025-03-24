#!/usr/bin/env python
#encoding:UTF-8

#对偶问题， 转化为求最大值
#二维　Rastrigin测试函数
def object_fun(p):
    import math
    x=p[0]
    y=p[1]
    # z=p[3]
    object_value=20.0+x**2+y**2-10.0*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y))
    return 100.0-object_value

