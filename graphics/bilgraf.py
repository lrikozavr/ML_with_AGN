# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:33:58 2020

@author: кирил
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#from pandas.tools.plotting import scatter_matrix
data=pd.read_csv("E:/Download/ex_all_PS1_nd.csv")
#print(data.shape)
#print(data.head())
#print(data.tail())
#print(data['RA'])
#print(data.describe())
'''
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print(categorical_columns)
print(numerical_columns)
'''
#print(data.columns)
#data1=data.columns
num_col = [c for c in data.columns if (c != 'RAJ2000' and c != 'DEJ2000' and c != 'RAJ2000.1' and c != 'DEJ2000.1'   )]
data_m=data[num_col]
#print(data_m.corr())
num_col1 = [c for c in data_m.columns if (c != 'RA' and c != 'DEC')]
data_nc=data_m[num_col1]
#print(data_nc.corr())
def Graf(ix,iy,jx,jy):
    x=data_nc[num_col1[ix]]-data_nc[num_col1[iy]]
    y=data_nc[num_col1[jx]]-data_nc[num_col1[jy]]
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=[np.min(x), np.max(x)],
           ylim=[np.min(y), np.max(y)]
           #title='Основы анатомии matplotlib',
           #xlabel=num_col1[ix]+"-"+num_col1[iy],
           
           #ylabel=num_col1[jx]+"-"+num_col1[jy])
           )
    ax.set_xlabel(num_col1[ix]+"-"+num_col1[iy],fontsize=40)
    ax.set_ylabel(num_col1[jx]+"-"+num_col1[jy],fontsize=40)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    fig.set_size_inches(20,20)
    #ax.xaxis.label.set_size(20)
    ax.scatter(x,y)
    fig.savefig('E:\GRAFICS\w1-y\plot'+str(ix)+str(iy)+str(jx)+str(jy)+'.png')
#plt.show()
def swap(i,j):
    return j,i
'''
for i in range(8,11,1):
    for j in range(8,11,1):
        if(not i==j):
            #q,w=swap(i,i+1)
            #if(not q==j and not w==j+1):
            Graf(i,i+1,j,j+1)
'''
'''
for i1 in range(7):
    for i2 in range(i1,7,1):
        for j1 in range(7):
            for j2 in range(j1,7,1):
                if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                    q,w=swap(i1,i2)
                    Graf(i1,i2,j1,j2)
'''
for i1 in range(12):
    for i2 in range(i1,12,1):
        for j1 in range(12):
            for j2 in range(j1,12,1):
                if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                    q,w=swap(i1,i2)
                    Graf(i1,i2,j1,j2)
'''
plt.xlabel(num_col1[0]+"-"+num_col1[1])
plt.ylabel(num_col1[1]+"-"+num_col1[2])
plt.scatter(data_nc[num_col1[0]]-data_nc[num_col1[1]],data_nc[num_col1[1]]-data_nc[num_col1[2]],marker=".")
plt.savefig('E:\GRAFICS\plot.png')


#fig(num=None,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
fig.savefig()
'''


#print(data[num_col])
#print(data1)
#print(data.corr())

#scatter_matrix(data, alpha=0.05, figsize=(10, 10));