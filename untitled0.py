#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:57:35 2018

@author: chendongcai
"""
import numpy as np

grid=np.zeros([10,8])

m = 2+1 # 行方格数+1为行点数
n = 3+1# 列方格数+1为列点数
    
def funr(dotstack, curlines):
    curdot = dotstack[-1]
    avalines = [x for x in curlines if x[0] == curdot or x[1] == curdot]
    if len(avalines) == 0:
        return 0
    
    res = 0
    for eachline in avalines:
        nextdot = eachline[0] if eachline[1] == curdot else eachline[1]
        curlines.remove(eachline)
        dotstack.append(nextdot)
        if nextdot == m * n - 1:
            print(dotstack)
            res += 1
        else:
            res += funr(dotstack, curlines)
        dotstack.pop()
        curlines.append(eachline)
    return res

if __name__ == '__main__':
    dots = range(m * n)
    lines = []
    for x in range(m):
        for y in range(n):
            if x < m-1:
                lines.append((y * m + x, y * m + x + 1))
            if y < n-1:
                lines.append((y * m + x, (y+1) * m + x))
    
    cnt = funr([0], lines)
    print('route count:', cnt)

