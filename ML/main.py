#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:01:55 2018

@author: vlad
"""


import numpy as np
import pandas as pd

from ml import NN
from absorption import dust_SFD
from graf import Many_Graf_diff,Many_Graf,Many_Graf_many,Many_Graf_diff_many

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

#input_path_data_test = "/media/kiril/j_08/AGN/excerpt/exerpt_folder/news/"
#filename = ['comp_ex.csv','news_shuf_ex.csv','news_ex.csv']


input_path_data_test = "/home/kiril/github/ML_with_AGN/ML/test/"
filename = ['news_phot.csv']

col_test_start_p = 2
col_test_end_p = 5

#col_label = 5
#input_path_data = "/home/kiril/github/ML_with_AGN/ML/train/sample_news.csv"
#col_label = 4
#input_path_data = "/home/kiril/github/ML_with_AGN/ML/train/sample.csv"
#col_label = 9
#input_path_data = "/home/kiril/github/ML_with_AGN/ML/train/sample_news_phot.csv"
col_label = 14
input_path_data = "/home/kiril/github/ML_with_AGN/ML/train/sample_news_all.csv"


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
#data_test = pd.read_csv(input_path_data, header=None, sep=',',dtype=np.float)
data_test = pd.read_csv(input_path_data, header=0, sep=',',dtype=np.float)
#print(data_test)
data_test['E(B-V)'] = dust_SFD(data_test['RA'],data_test['DEC'])

data_test['g'] -= 3.172*data_test['E(B-V)']
data_test['r'] -= 2.271*data_test['E(B-V)']
data_test['i'] -= 1.682*data_test['E(B-V)']
data_test['z'] -= 1.322*data_test['E(B-V)']
data_test['y'] -= 1.087*data_test['E(B-V)']

'''
data_test['g'] -= [3.172*x if 3.172*x < 1 else 0.6 + 0.2 * (1 - np.tanh((3.172*x - 0.15) / 0.3)) for x in data_test['abs_q']]
data_test['r'] -= [2.271*x if 2.271*x < 1 else 0.6 + 0.2 * (1 - np.tanh((2.271*x - 0.15) / 0.3)) for x in data_test['abs_q']]
data_test['i'] -= [1.682*x if 1.682*x < 1 else 0.6 + 0.2 * (1 - np.tanh((1.682*x - 0.15) / 0.3)) for x in data_test['abs_q']]
data_test['z'] -= [1.322*x if 1.322*x < 1 else 0.6 + 0.2 * (1 - np.tanh((1.322*x - 0.15) / 0.3)) for x in data_test['abs_q']]
data_test['y'] -= [1.087*x if 1.087*x < 1 else 0.6 + 0.2 * (1 - np.tanh((1.087*x - 0.15) / 0.3)) for x in data_test['abs_q']]
'''

#label = data_test[col_label]
label = data_test['label']

agn_sample,news_sample=[],[]
agn_sample,news_sample=data_test[label == 1],data_test[label == 0]
#data_test = data_test.drop(col_label, axis=1)
data_test = data_test.drop(['label','RA','DEC','abs_q','E(B-V)'],axis=1)
#data_test = data_test.drop(['f1','f2','f3','f4','f5'],axis=1)
#data_test = data_test.drop(['w1','w2','w3','w4'],axis=1)
#data_test = data_test.drop(['g','r','i','z','y'],axis=1)
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
print(data_test.columns.values)
#exit()
data_name=['AGN','Other']
#name=['1','2','3','4','5','6','7','8','9']
name=np.array(data_test.columns.values)
#print(name[0])
save_pic_path='/home/kiril/github/ML_with_AGN/ML/pic'
save_pic_path_agn='/home/kiril/github/ML_with_AGN/ML/pic/pic_sample_agn'
save_pic_path_news='/home/kiril/github/ML_with_AGN/ML/pic/pic_sample_news'


#####
#Many_Graf_many(agn_sample,news_sample,data_name,name,save_pic_path,col_label)
#exit()
#####


#Many_Graf_diff_many(agn_sample,news_sample,data_name,name,save_pic_path,col_label)

#Many_Graf(agn_sample,name,save_pic_path_agn,5)
#Many_Graf(news_sample,name,save_pic_path_news,5)


train=DataP(data_test,flag_color) 									############################flag_color

print("Data train shape:	",train.shape)

num_ep = 50
batch_size = 1024

NN(train,np.array(label),0.25,0.25,batch_size,num_ep,optimizer,loss,output_path_predict_0,output_path_mod,output_path_weight)
exit()
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
