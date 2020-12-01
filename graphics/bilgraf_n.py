# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:33:58 2020
@author: кирил
"""
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#from pandas.tools.plotting import scatter_matrix
data=pd.read_csv("/media/kiril/j_08/AGN/excerpt/catalogue/EX_Z_NAME_ALL_PS1.csv")
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
def swap(i,j):
    return j,i
    
#print(data.columns)
#data1=data.columns
num_col = [c for c in data.columns if (c != 'RAJ2000' and c != 'DEJ2000' and c != 'RAJ2000.1' and c != 'DEJ2000.1'   )]
data_m=data[num_col]
#print(data_m.corr())
num_col1 = [c for c in data_m.columns if (c != 'RA' and c != 'DEC')]
data_nc=data_m[num_col1]
#print(data_nc.corr())
s=data_nc['Name_1'].unique()
#print(s)
s[6],s[5]=swap(s[6],s[5])
#print(s)
groupped=data_nc.groupby('Name_1')
g=[[[]]]
for i in s:
    g.append(groupped.get_group(i))

#print(g[1])
c=['yellow','blue','black','red','pink','magenta','green']
def Graf(ix,iy,jx,jy):
    y,x=[],[]
    for i in range(1,8,1):
        x.append(g[i][num_col1[ix]]-g[i][num_col1[iy]])
        y.append(g[i][num_col1[jx]]-g[i][num_col1[jy]])
        
    fig=plt.figure()
    ax = fig.add_subplot(111)
    '''
    ax.set(xlim=[np.min(x), np.max(x)],
           ylim=[np.min(y), np.max(y)]
           #title='Основы анатомии matplotlib',
           #xlabel=num_col1[ix]+"-"+num_col1[iy],
           yt gjrfpsdfnm uhfabr 
           #ylabel=num_col1[jx]+"-"+num_col1[jy])
           )
    '''
    ax.set_xlabel(num_col1[ix]+"-"+num_col1[iy],fontsize=40)
    ax.set_ylabel(num_col1[jx]+"-"+num_col1[jy],fontsize=40)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    fig.set_size_inches(20,20)
    #ax.xaxis.label.set_size(20)
    for i in range(6,-1,-1):
        ax.scatter(x[i],y[i],s=20,c=c[i],label=s[i])
    ax.legend(loc=1,prop={'size': 20})
    fig.savefig('/media/kiril/j_08/AGN/excerpt/catalogue/grizy(7)/plot'+str(ix)+str(iy)+str(jx)+str(jy)+'.png')
    plt.close(fig)
#plt.show()
'''
for i1 in range(7,12,1):
    for i2 in range(i1,12,1):
        for j1 in range(7,12,1):
            for j2 in range(j1,12,1):
                if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                    Graf(i1+2,i2+2,j1+2,j2+2)
'''
            
################
'''
for i1 in range(7):
    for i2 in range(i1,7,1):
        for j1 in range(7):
            for j2 in range(j1,7,1):
                if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                    #q,w=swap(i1,i2)
                    Graf(i1+2,i2+2,j1+2,j2+2)
'''
#################
#Graf(11,10,11,10)
'''
for i1 in range(12):
    for i2 in range(i1,12,1):
        for j1 in range(12):
            for j2 in range(j1,12,1):
                if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                    #q,w=swap(i1,i2)
                    Graf(i1,i2,j1,j2)
'''
'''
plt.xlabel(num_col1[0]+"-"+num_col1[1])
plt.ylabel(num_col1[1]+"-"+num_col1[2])
plt.scatter(data_nc[num_col1[0]]-data_nc[num_col1[1]],data_nc[num_col1[1]]-data_nc[num_col1[2]],marker=".")
plt.savefig('E:\GRAFICS\plot.png')
#fig(num=None,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
fig.savefig()
'''
#Graf(10,11,9,10)

#print(data[num_col])
#print(data1)
#print(data.corr())

#scatter_matrix(data, alpha=0.05, figsize=(10, 10));

def Hist1(x,name):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    '''
    volmin=math.trunc(np.min(x))
    volmax=math.trunc(np.max(x))
   # print(volmin,volmax)
    gis=[0]*(volmax-volmin)
   # print(gis)
    n=np.size(x)
    x_=x.isnull()
    for i in range(n):
        if (not x_[i]):     
            #print(math.trunc(x[i]-volmin),x_[i])
            gis[math.trunc(x[i]-volmin-1)]+=1
    bins=[i for i in range(volmin,volmax,1)]
    #print(bins,gis,np.size(bins),np.size(gis))
    '''
    x1=[]
    x_=x.isnull()
    for i in range(np.size(x)):
        if (not x_[i]):     
            x1.append(x[i])
    fig.suptitle(name, fontsize=50)       
    ax.set_xlabel("mag",fontsize=40)
    ax.set_ylabel("count",fontsize=40)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    #ax.set_title(name,fontsize = 50)
    fig.set_size_inches(30,20)
    ax.hist(x1,bins=200)
    fig.savefig('/media/kiril/j_08/AGN/excerpt/catalogue/Hist(7)/hist_'+name+'.png')
    plt.close(fig)

def Hist2(n1,n2):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    '''
    volmin=math.trunc(np.min(x))
    volmax=math.trunc(np.max(x))
   # print(volmin,volmax)
    gis=[0]*(volmax-volmin)
   # print(gis)
    n=np.size(x)
    x_=x.isnull()
    for i in range(n):
        if (not x_[i]):     
            #print(math.trunc(x[i]-volmin),x_[i])
            gis[math.trunc(x[i]-volmin-1)]+=1
    bins=[i for i in range(volmin,volmax,1)]
    #print(bins,gis,np.size(bins),np.size(gis))
    '''
    x=[]
    x1=data_nc[n1]-data_nc[n2]
    x_=x1.isnull()
    for i in range(np.size(x1)):
        if (not x_[i]):     
            x.append(x1[i])
    fig.suptitle(n1+"-"+n2, fontsize=50)       
    ax.set_xlabel("mag",fontsize=40)
    ax.set_ylabel("count",fontsize=40)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    #ax.set_title(name,fontsize = 50)
    fig.set_size_inches(30,20)
    #x.sort()
    ax.hist(x,bins=200)
    fig.savefig('/media/kiril/j_08/AGN/excerpt/catalogue/Hist_m(7)/hist_'+n1+"_"+n2+'.png')
    plt.close(fig)


#print(data_nc['gmag'])
col=[]
for i in data_nc.columns:
    if (not i=='Name_1' and not i=='z_1'):
        col.append(i)
        Hist1(data_nc[i],i)
#print(col[4])
'''
for i1 in range(7):
    for i2 in range(i1,7,1):
        if(not i1==i2 ):
            Hist2(col[i1],col[i2])            
'''
'''
for i1 in range(7,12,1):
    for i2 in range(i1,12,1):
        if(not i1==i2 ):
            Hist2(col[i1],col[i2])            
'''            
