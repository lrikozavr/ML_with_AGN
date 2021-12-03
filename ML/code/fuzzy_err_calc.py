#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

delta = 1e-4

def M(data,n):
    sum = 0
    data = data.reset_index(drop=True)
    #print(data)
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
    rc = pd.DataFrame(np.array(['fuf']), columns=['word'])
    #
    for col in columns:
        rc[col] = M(data[col],count)
        #print(rc[col])
        #print("M        ", M(data[col],count))
    #print("rc           ",rc)
    r = []
    max = -1
    for i in range(count):
        ev_sum = 0
        for col in columns:
            ev_sum += (rc[col].iloc[0] - data[col].iloc[i])**2
        #print(ev_sum)    
        r.append(math.sqrt(ev_sum))
        if(r[i] > max):
            max = r[i]
    print("fuzzy_dist complite")
    return r, max

def fuzzy_err(data):
    columns = data.columns.values
    count = len(data)
    #print(count,columns)

    summ = []
    max = np.zeros(len(columns))
    index=0
    for col in columns:
        for i in range(count):
            if(data[col].iloc[i] > max[index]):
                max[index]=data[col].iloc[i]
        index+=1
    np.zeros((count,))

    for i in range(count):
        sum = 0
        index = 0
        for col in columns:
            sum += (1 - data[col].iloc[i]/(max[index]+delta))**2
            index += 1
        summ.append(math.sqrt(sum/float(index)))
    print("fuzzy_err complite")
    return summ

def colors(data):
    print(data)
    list_name = data.columns.values
    count = data.shape[0]
    mags = int(data.shape[1]/2)
    num_colours = sum(i for i in range(mags))
    colours = np.zeros((count,num_colours))
    colours_error = np.zeros((count,num_colours))
    index = 0
    colours_name, colours_error_name = [], []
    data=np.array(data)
    for j in range(mags):
        for i in range(j, mags):
            if(i!=j):
                colours_name.append(f"{list_name[j*2]}&{list_name[i*2]}")
                colours_error_name.append(f"{list_name[j*2+1]}&{list_name[i*2+1]}")
                colours[:,index] = data[:,j*2] - data[:,i*2]
                colours_error[:,index] = np.sqrt(data[:,j*2+1]**2 + data[:,i*2+1]**2)
                index += 1
    print(colours_name)
    print(colours_error_name)
    colours = pd.DataFrame(colours, columns=colours_name)
    colours_error = pd.DataFrame(colours_error, columns=colours_error_name)
    return colours, colours_error