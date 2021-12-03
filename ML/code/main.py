#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

from ml_decide import et,rf,dm,logreg,sv,xg
from sklearn.utils.fixes import loguniform
import training_utils
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append('/home/kiril/github/ML_with_AGN/ML')
from ml import NN

training_data = pd.read_csv("training_sample.csv", header=0, sep=',')
general_data = pd.read_csv("main_sample.csv", header=0, sep=',')

info_columns = ['name','type']

#fuzzy_options = ["normal", "fuzzy_dist", "fuzzy_err"]
fuzzy_options = ["fuzzy_dist", "fuzzy_err"]
'''
features = ['W1mag','W2mag','W3mag',
            #'e_W1mag','e_W2mag','e_W3mag',
            #'gmag','rmag','imag','zmag','ymag',
            #'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
            'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag'
            #'phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error'
            ]
'''
features = ['W1mag&W2mag', 'W1mag&W3mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
'W2mag&W3mag', 'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag', 
'W3mag&phot_g_mean_mag', 'W3mag&phot_bp_mean_mag', 'W3mag&phot_rp_mean_mag', 
'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
'phot_bp_mean_mag&phot_rp_mean_mag']


fuzzy_dist_column = ["fuzzy_dist"]
fuzzy_err_column = ["fuzzy_err"]
output_path = "./results"

def cor():
    import seaborn as sns
    import matplotlib.pyplot as plt
    print(general_data)
    corr = general_data[features].corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.savefig('1.jpg')
    exit()

#NN(training_data[features].values,training_data["Y"].values,0.4,0.4,1024,100,'adam','binary_crossentropy',
#'/home/kiril/github/AGN_article_final_data/inform/predict','/home/kiril/github/AGN_article_final_data/inform/mod','/home/kiril/github/AGN_article_final_data/inform/weight','/home/kiril/github/AGN_article_final_data/inform')

#NN(general_data[features].values,general_data["Y"].values,0.4,0.4,1024,100,'adam','binary_crossentropy',
#'/home/kiril/github/AGN_article_final_data/inform/predict','/home/kiril/github/AGN_article_final_data/inform/mod','/home/kiril/github/AGN_article_final_data/inform/weight','/home/kiril/github/AGN_article_final_data/inform')

NN(general_data[features],general_data["Y"].values,0.4,0.4,1024,100,'adam','binary_crossentropy',
'/home/kiril/github/AGN_article_final_data/inform/predict','/home/kiril/github/AGN_article_final_data/inform/mod','/home/kiril/github/AGN_article_final_data/inform/weight','/home/kiril/github/AGN_article_final_data/inform')

 
def multi_classifire(general_data,training_data,fuzzy_dist_column,fuzzy_err_column,output_path,info_columns,features):


    for fuzzy_option in fuzzy_options:
        
        print(fuzzy_option)
        
        train_X = training_data[features].values
        general_X = general_data[features].values

        '''
        dm(fuzzy_option, 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "dummy", info_columns, features)
        '''
        et(fuzzy_option, 'balanced', 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "et_b", info_columns, features)
        et(fuzzy_option, None, 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "et_not_b", info_columns, features)
        
        rf(fuzzy_option,'balanced', 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "rf_b", info_columns, features)
        rf(fuzzy_option,None, 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "rf_not_b", info_columns, features)
        
    # scale features of the data:
        #train_X, general_X = training_utils.scale_X_of_the_data(training_data[features], general_data[features])
        #train_X, general_X = training_utils.scale_X_of_the_data(data_t, data_g)
        '''
        params = {"penalty": ["l1", "l2"],#, "elasticnet"],
                'C': loguniform(1e0, 1e3)}
        
        logreg(fuzzy_option,'balanced',params, 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "logreg_b", info_columns, features)
        logreg(fuzzy_option,None,params, 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "logreg_not_b", info_columns, features)
        
        
        
        params = {'C': loguniform(1e0, 1e3),
                'gamma': loguniform(1e-4, 1e-2)}
        sv(fuzzy_option,'balanced',params, 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "svm_b", info_columns, features)
        sv(fuzzy_option,None,params, 
            general_X, general_data, training_data, train_X, 
            fuzzy_dist_column, fuzzy_err_column, 
            output_path, "svm_not_b", info_columns, features)
        '''
    '''
	# scale features of the data:
    #training_part, test_part = train_test_split(training_data, test_size=0.2, random_state=42)
    training_part, test_part = train_test_split(training_data, test_size=0.2, random_state=42)

    #train_X_all, general_X = training_utils.scale_X_of_the_data(training_data[features], general_data[features])
    #train_X, test_X = training_utils.scale_X_of_the_data(training_part[features], test_part[features])

    train_X_all, general_X = training_utils.scale_X_of_the_data(data_t, data_g)
    data_trp = Diff(training_part, flag_color)
    data_tep = Diff(test_part, flag_color)
    train_X, test_X = training_utils.scale_X_of_the_data(data_trp, data_tep)
    
    params = {'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
	           'min_child_weight': [1, 5, 10],
	           'gamma': [0.5, 1, 1.5, 2, 5],
	           'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
	           'colsample_bytree': [0.8, 1.0],
	           'max_depth': [2, 3, 4, 5, 6, 10],
	           'lambda': [1, 2, 4],
	           'alpha': [0, 1, 2]}
	#class_weight= 7 (balanced), 1 (not)
    xg(fuzzy_option,7,params, 
        train_X, training_part, test_X, test_part,
        general_X, general_data, training_data,  
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, "xgb_b", info_columns, features)
    xg(fuzzy_option,1,params, 
        train_X, training_part, test_X, test_part,
        general_X, general_data, training_data,  
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, "xgb_not_b", info_columns, features)
    '''
multi_classifire(general_data,training_data,fuzzy_dist_column,fuzzy_err_column,output_path,info_columns,features)
    