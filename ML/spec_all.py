#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ml import NN,LoadModel
from graf import Many_Graf_pd,Many_Graf_pd_diff
from DataTrensform import DataP
import os
import time

#save_pic_path='/home/kiril/github/ML_with_AGN/ML/pic/P_nonerr'
save_pic_path="/home/kiril/github/ML_data/image_phot"

output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod_relu_"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight_relu_"

output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/P"

#input_path_data_agn = "/home/kiril/github/ML_with_AGN/ML/train_/sample_z_allwise_ps1_gaiadr3.csv"
input_path_data_agn = "/home/kiril/github/ML_data/data/agn_end.csv"
input_path_data_star = "/home/kiril/github/ML_data/data/star_end.csv"
input_path_data_qso = "//home/kiril/github/ML_data/data/qso_end.csv"
input_path_data_gal = "/home/kiril/github/ML_data/data/gal_end.csv"

data_agn = pd.read_csv(input_path_data_agn, header=0, sep=',',dtype=np.float)
data_star = pd.read_csv(input_path_data_star, header=0, sep=',',dtype=np.float)
data_qso = pd.read_csv(input_path_data_qso, header=0, sep=',',dtype=np.float)
data_gal = pd.read_csv(input_path_data_gal, header=0, sep=',',dtype=np.float)

data_agn['name'] = "AGN"
data_star['name'] = "STAR"
data_qso['name'] = "QSO"
data_gal['name'] = "GALAXY"

data_agn['label']=[1 for i in data_agn['RA']]
data_star['label']=[0 for i in data_star['RA']]
data_qso['label']=[0 for i in data_qso['RA']]
data_gal['label']=[0 for i in data_gal['RA']]

data_agn_star = data_agn.append(data_star, ignore_index=True)
data_agn_qso = data_agn.append(data_qso, ignore_index=True)
data_agn_gal = data_agn.append(data_gal, ignore_index=True)
data_agn_star_qso = data_agn_star.append(data_qso, ignore_index=True)
data_agn_star_gal = data_agn_star.append(data_gal, ignore_index=True)
data_agn_qso_gal = data_agn_qso.append(data_gal, ignore_index=True)
data_agn_star_qso_gal = data_agn_star_qso.append(data_gal, ignore_index=True)



optimizer = 'adam'
#loss = 'categorical_crossentropy'
loss = 'binary_crossentropy'
num_ep = 50
batch_size = 1024

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return dir_name

def test(data):
    data = data.sort_values(by=['DEC'], ascending=True, ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)

    label = data['label']
    #
    
    #
    c=0
    for i in range(label.size):
        if(label[i]==1):
            c+=1
    print("Data test shape:	",data.shape)
    print("Data val size:	",np.size(label))
    print("%",c/label.size *100)
    print(data.columns.values)
    #
    #data = data.drop(['e_W1mag','e_W2mag','e_W3mag','e_W4mag','e_Jmag','e_Hmag','e_Kmag',
    #                'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
    #                'parallax','parallax_error','pm','pmra','pmra_error','pmdec','pmdec_error','phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error'], axis=1)
    data = data.drop(['z','e_W1mag','e_W2mag','e_W3mag','e_W4mag','e_Jmag','e_Hmag','e_Kmag','Jmag','Hmag','Kmag',
                    'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
                    'parallax_error','pm','pmra_error','pmdec_error','phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error','bp_rp','phot_g_mean_mag'], axis=1)
    data = data.drop(['parallax','pmra','pmdec','W4mag','W3mag'], axis=1)
#'gmag','rmag','imag','zmag','ymag','phot_bp_mean_mag','phot_rp_mean_mag'
    #
    #data = data.drop(['z'], axis=1)
    #
    #data = data.drop(['W3mag','W4mag','Jmag','Hmag','Kmag','bp_rp'], axis=1)
    #
    #agn_sample,other_sample=data[label == [1],data[label == 0]
    
    data = data.drop(['label','RA','DEC'],axis=1)
    name_list = data['name'].unique()
    local_output_path_mod = output_path_mod
    local_output_path_weight = output_path_weight
    local_output_path_predict = output_path_predict
    local_save_pic_path = save_pic_path + "/P"
    nname="P"
    for name_ in name_list:
        local_output_path_mod = local_output_path_mod + "_" + name_
        local_output_path_weight = local_output_path_weight + "_" + name_
        local_output_path_predict = local_output_path_predict + "_" + name_
        
        local_save_pic_path = local_save_pic_path + "_" + name_
        nname += "_" + name_
    print(dir(save_pic_path,nname))    
    local_output_path_predict += ".csv"
    
    #os.mkdir(local_save_pic_path)
    #Many_Graf_pd(data,local_save_pic_path)
    #Many_Graf_pd(data,local_save_pic_path)
    
    data = data.drop(['name'],axis=1)
    
    #cols = ['phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','gmag','rmag','imag','zmag','ymag','W1mag','W2mag']
    #data = data[cols]
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = data.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.savefig('2.jpg')
    #exit()
    print(data.columns.values)
    train = DataP(data,0)
    print(train)
    print("Data train shape:	",train.shape)
    print(label)
    NN(train,np.array(label),0.25,0.25,batch_size,num_ep,optimizer,loss,local_output_path_predict,local_output_path_mod,local_output_path_weight)
    #agn_sample=agn_sample.drop(['name','label','RA','DEC'],axis=1)
    #agn_sample=agn_sample[cols]
    #agn_sample=DataP(agn_sample,0) 									############################flag_color
    model1 = LoadModel(local_output_path_mod,local_output_path_weight,optimizer,loss)
    #Class = model1.predict(agn_sample, batch_size)
    Class = model1.predict(train, batch_size)
    
    Class = np.array(Class)

    #g=open(output_path_predict+"/"+name,'w') 
    #Class.tofile(g,"\n")
    #g.close()

    agn,gal,qso,star=0,0,0,0
    for i in range(len(Class)):
        #if(Class[i]<0.5):
            #Class[i] = 0
        if(Class[i][0]>0.5):
            agn+=1
        else:
            if(Class[i][1]>0.5):
                gal+=1
            else:
                if(Class[i][2]>0.5):
                    qso+=1
                else:
                    if(Class[i][3]>0.5):
                        star+=1
        
    print("AGN:	",agn /np.size(Class) *100,"%")
    print("GAL:	",gal /np.size(Class) *100,"%")
    print("QSO:	",qso /np.size(Class) *100,"%")
    print("STAR:	",star /np.size(Class) *100,"%")
    
#test(data_agn_star)
#test(data_agn_qso)
#test(data_agn_gal)
test(data_agn_star_qso_gal)
