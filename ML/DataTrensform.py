#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#Разница между всеми
def Diff(data,flag_color):
	stars = data.shape[0]
	mags = data.shape[1]
	num_colours = sum(i for i in range(mags))
	colours = np.zeros((stars,num_colours))
	index = 0
	#
	for j in range(mags):
		for i in range(j, mags):
			if(i!=j):
				colours[:,index] = data[:,j] - data[:,i]
				index += 1
	if(not flag_color):
		Result = np.append(data,colours, axis=1)
	else:
		Result = colours
	print("Different all Data")
	print(Result.shape)
	return Result

#Выравнивание
def Rou(data):
	features = data.shape[1]
	#print(features)
	#print(data)
	means = np.zeros(features)
	stds = np.zeros(features)
	Result = np.array(data)
	for i in range(features):
		means[i] = np.mean(data[:,i])
		stds[i] = np.std(data[:,i])
		Result[:,i] = (data[:,i] - means[i])/stds[i]
	#print(Result)
	print("Normalisation Data")
	return Result
	
def DataP(data,flag_color):
	data.fillna(0)
	data = np.array(data)
	return  data #Diff(data,flag_color) #Rou(Diff(data))
