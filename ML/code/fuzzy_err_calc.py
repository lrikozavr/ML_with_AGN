#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

delta = 1e-4

def M(data,n):
    sum = 0
    for i in range(n):
        sum += data[i]
    return sum/float(n)

def Normali(data,max):
    count = len(data)
    s = []
    for i in range(count):
        s.append(1 - data[i] / (max + delta))
    return s

def fuzzy_dist(data):
    columns = data.columns.values
    count = len(data)
    rc = pd.DataFrame()

    for col in columns:
        rc[col] = M(data[col],count)

    r = []
    max = -1
    for i in range(count):
        ev_sum = 0
        for col in columns:
            ev_sum += (rc[col] - data[col].iloc[i])**2
        r.append(math.sqrt(ev_sum))
        if(r > max):
            max = r[i]
    
    return r, max

def fuzzy_err(data):
    columns = data.columns.values
    count = len(data)
    print(count,columns)

    summ = []
    max = -1

    for i in range(count):
        sum = 0
        for col in columns:
            sum += data[col].iloc[i]
        summ.append(sum)
        if(sum > max):
            max=sum
    return summ,max