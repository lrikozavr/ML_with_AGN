#!/home/kiril/python_env_iron_ment/my_project/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:01:55 2018

@author: vlad
"""
#from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K

def DeepDenseNN(features):
    input_img = Input(shape=(features,))
    #print(input_img)
    #input_img = Dense(128, activation='relu', kernel_initializer='he_uniform', input_shape=(features,))
    layer_1 = Dense(64, activation='linear', kernel_initializer='he_uniform' )(input_img)
    layer_2 = Dense(32, activation='tanh', kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(16, activation='relu', kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(8, activation='tanh', kernel_initializer='he_uniform' )(layer_3)
    layer_5 = Dense(4, activation='elu', kernel_initializer='he_uniform' )(layer_4)
    Label = Dense(1, activation='sigmoid', kernel_initializer='he_uniform')(layer_5)    
    model = Model(input_img, Label)
    return model

def SaveModel(model, path_model, path_weights):
    model_json = model.to_json()
    with open(path_model, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path_weights)
    print("Model is saved to disk\n")

def LoadModel(path_model, path_weights, optimizer, loss):
    from keras.models import model_from_json
    json_file = open(path_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    print("Model is loaded from disk\n")
    loaded_model.compile(optimizer=optimizer, loss=loss)
    return loaded_model

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


def NN(train,label,test_size,validation_split,batch_size,num_ep,optimizer,loss,output_path_predict,output_path_mod,output_path_weight):
	X_train, X_test, y_train, y_test = train_test_split(train, label, 
														test_size = test_size, random_state = 56) #0.4
	#print(X_train, X_test, y_train, y_test)
	#batch_size = 1024
	#num_ep = 15
	features = train.shape[1]
	print(features)
	model = DeepDenseNN(features)

	model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

	model.fit(X_train, y_train,
			epochs=num_ep,
			batch_size=batch_size,
			validation_split=validation_split) #0.25

	model.summary()
	Class = model.predict(X_test, batch_size)
	
	Class = np.array(Class)
	
	g=open(output_path_predict,'w')
	Class.tofile(g,"\n")
	g.close()
	
	count = 0
	Tq, Fq, Ts, Fs = 0,0,0,0
	for i in range(y_test.shape[0]):
		#print(Tq,Fq,Ts,Fs)
		if(Class[i]<0.5):
			Class[i] = 0
		if(Class[i]>=0.5):
			Class[i] = 1
		if(Class[i]==y_test[i]):
			count+=1
		if(Class[i]==1):
			if(Class[i]==y_test[i]):
				Tq += 1
			else:
				Fq += 1
		if(Class[i]==0):
			if(Class[i]==y_test[i]):
				Ts += 1
			else:
				Fs += 1
	print("Accuracy:",              count/y_test.shape[0])
	print("AGN precision:",     Tq/(Tq+Fq))
	print("nonAGN precision:",    Ts/(Ts+Fs))
	print("AGN completness:",       Tq/(Tq+Fs))
	print("nonAGN completness:",     Ts/(Ts+Fq))
	print("AGN_F:",     2*(Tq/(Tq+Fq)*Tq/(Tq+Fs))/(Tq/(Tq+Fq)+Tq/(Tq+Fs)) )
	print("non_AGN_F:",     2*(Ts/(Ts+Fs)*Ts/(Ts+Fq))/(Ts/(Ts+Fq)+Ts/(Ts+Fs)) )
	SaveModel(model,output_path_mod,output_path_weight)
	#return model

from graf import Many_Graf_diff,Many_Graf,Many_Graf_many

batch_size = 1024
optimizer = 'adam'
loss = 'binary_crossentropy'
'''
output_path_predict_0 = "/home/kiril/github/ML_with_AGN/ML/predict/main.csv"
output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/"

output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod1"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight1"

input_path_data_test = "/home/kiril/github/ML_with_AGN/ML/test/"
filename = ['2dfFGRS_all.csv',
'DEEP2_all.csv',
'Tic_all.csv',
'VIMOS_DR1_p_w1_all.csv',
'VIMOS_DR1_p_w4_all.csv',
'VIMOS_DR1_s_w1_all.csv',
'VIMOS_DR1_s_w4_all.csv',
'file_ex_all.csv',
'star_shuf_all.csv']

input_path_data = "/home/kiril/github/ML_with_AGN/ML/train/sample.csv"
input_path_data_train = "/home/kiril/github/ML_with_AGN/ML/train/file_ex_all.csv"
input_path_data_trash = "/home/kiril/github/ML_with_AGN/ML/train/star_shuf_all.csv"
'''
output_path_predict_0 = "/home/kiril/github/ML_with_AGN/ML/predict/sample_news.csv"
output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/"

col_test_start_s = 2
col_test_end_s = 5

output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod_news"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight_news"

input_path_data_test = "/media/kiril/j_08/AGN/excerpt/exerpt_folder/news/"
filename = ['comp_ex.csv','news_shuf_ex.csv','news_ex.csv']

col_test_start_p = 2
col_test_end_p = 5

#col_label = 5
#input_path_data = "/home/kiril/github/ML_with_AGN/ML/train/sample_news.csv"
col_label = 4
input_path_data = "/home/kiril/github/ML_with_AGN/ML/train/sample.csv"

flag_color = 1 # 1 - only color; 0 - with main data
#######################################################################################################################################################
##### ###       ## ### ##   # ### ##   #  ####
  #   #  #     # #  #  # #  #  #  # #  # #    #
  #   ###     ####  #  #  # #  #  #  # # # ###
  #   #  #   #   #  #  #   ##  #  #   ## #    #
  #   #   # #    # ### #    # ### #    #  ####
#######################################################################################################################################################
'''
data_test = pd.read_csv(input_path_data_train, header=None, sep=',',dtype=np.float)
label = [1 for i in range(data_test.shape[0])]


#dada_test = pd.read_csv("/media/kiril/j_08/AGN/excerpt/catalogue/star_n_inf_nonzero.csv", header=None, sep=',',dtype=np.float)
dada_test = pd.read_csv(input_path_data_trash, header=None, sep=',', dtype=np.float)
babel = [0 for i in range(dada_test.shape[0])]

label = np.array(label)
babel = np.array(babel)

data_test = data_test.append(dada_test, ignore_index=True)
label = np.append(label,babel,axis=0)
'''
data_test = pd.read_csv(input_path_data, header=None, sep=',',dtype=np.float)



#data_test = data_test.iloc[:, 4:8]
label = data_test[col_label]

agn_sample,news_sample=[],[]
agn_sample,news_sample=data_test[label == 1],data_test[label == 0]
data_test = data_test.drop(col_label, axis=1)
#print(data_test)
#print(label)
#exit()
c=0
for i in range(label.size):
	if(label[i]==1):
		c+=1
print("Data test shape:	",data_test.shape)
print("Data val size:	",np.size(label))
print("%",c/label.size *100)

data_name=['AGN','Other']
name=['1','2','3','4','5']
save_pic_path='/home/kiril/github/ML_with_AGN/ML/pic'
save_pic_path_agn='/home/kiril/github/ML_with_AGN/ML/pic/pic_sample_agn'
save_pic_path_news='/home/kiril/github/ML_with_AGN/ML/pic/pic_sample_news'


#Many_Graf(data_test,name,save_pic_path,5)
Many_Graf_many(agn_sample,news_sample,data_name,name,save_pic_path,col_label)
#Many_Graf(agn_sample,name,save_pic_path_agn,5)
#Many_Graf(news_sample,name,save_pic_path_news,5)
exit()
train=DataP(data_test,flag_color) 									############################flag_color

print("Data train shape:	",train.shape)

num_ep = 25
batch_size = 1024

NN(train,np.array(label),0.25,0.25,batch_size,num_ep,optimizer,loss,output_path_predict_0,output_path_mod,output_path_weight)

#######################################################################################################################################################
##### ###       ## ### ##   # ### ##   #  ####
  #   #  #     # #  #  # #  #  #  # #  # #    #
  #   ###     ####  #  #  # #  #  #  # # # ###
  #   #  #   #   #  #  #   ##  #  #   ## #    #
  #   #   # #    # ### #    # ### #    #  ####
#######################################################################################################################################################
##########
#data_test = pd.read_csv("/media/kiril/j_08/AGN/excerpt/catalogue/LQAC3_WISE_w1_w4/LQAC3_WISE.csv", header=0, sep=',',dtype=np.float)
#data_test = data_test.drop(['RA','DEC'], axis=1)

#######################################################################################################################################################
##### ####  ### #####
  #   #    #      #
  #   ###   ##    #
  #   #       #   #
  #   #### ###    #
#######################################################################################################################################################

def Test(batch_size,output_path_mod,output_path_weight,optimizer,loss,input_path_data_test,output_path_predict,name,flag_color):
	data_test = pd.read_csv(input_path_data_test+"/"+name, header=None, sep=',', dtype=np.float)
	data_test.fillna(0)
	train=DataP(data_test,flag_color) 									############################flag_color
	
	model1 = LoadModel(output_path_mod,output_path_weight,optimizer,loss)
	Class = model1.predict(train, batch_size)


	Class = np.array(Class)

	g=open(output_path_predict+"/"+name,'w')
	Class.tofile(g,"\n")
	g.close()

	j=0
	for i in range(np.size(Class)):
		#if(Class[i]<0.5):
			#Class[i] = 0
		if(Class[i]>=0.5):
			#Class[i] = 1
			j+=1
	print(name+":	",j /np.size(Class) *100,"%")

for i in filename:
	Test(batch_size,output_path_mod,output_path_weight,optimizer,loss,input_path_data_test,output_path_predict,i,flag_color)

#######################################################################################################################################################
##### ####  ### #####
  #   #    #      #
  #   ###   ##    #
  #   #       #   #
  #   #### ###    #
#######################################################################################################################################################
