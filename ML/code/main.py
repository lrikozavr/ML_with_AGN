#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from ml_decide import et,rf,dm,logreg,svm,xgb
from sklearn.utils.fixes import loguniform
import training_utils
from sklearn.model_selection import train_test_split

training_data = pd.read_csv("training_sample.csv", header=0, sep=',', dtype=np.float)
general_data = pd.read_csv("main_sample.csv", header=0, sep=',', dtype=np.float)

info_columns = ['name','type']

fuzzy_options = ["normal", "fuzzy_dist", "fuzzy_err"]

features = ['W1mag','W2mag','W3mag','W4mag',
            'e_W1mag','e_W2mag','e_W3mag','e_W4mag',
            'gmag','rmag','imag','zmag','ymag',
            'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
            'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
            'phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error']
fuzzy_dist_column = ["fuzzy_dist"]
fuzzy_err_column = ["fuzzy_err"]
output_path = "./results"

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)

    train_X = training_data[features].values
    general_X = general_data[features].values

    dm(fuzzy_option, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, "dummy", info_columns, features)
    
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
    train_X, general_X = training_utils.scale_X_of_the_data(training_data[features], general_data[features])

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
    svm(fuzzy_option,'balanced',params, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, "svm_b", info_columns, features)
    svm(fuzzy_option,None,params, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, "svm_not_b", info_columns, features)

	# scale features of the data:
    training_part, test_part = train_test_split(training_data, test_size=0.2, random_state=42)

    train_X_all, general_X = training_utils.scale_X_of_the_data(training_data[features], general_data[features])
    train_X, test_X = training_utils.scale_X_of_the_data(training_part[features], test_part[features])

    params = {'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
	           'min_child_weight': [1, 5, 10],
	           'gamma': [0.5, 1, 1.5, 2, 5],
	           'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
	           'colsample_bytree': [0.8, 1.0],
	           'max_depth': [2, 3, 4, 5, 6, 10],
	           'lambda': [1, 2, 4],
	           'alpha': [0, 1, 2]}
	#class_weight= 7 (balanced), 1 (not)
    xgb(fuzzy_option,7,params, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, "xgb_b", info_columns, features)
    xgb(fuzzy_option,1,params, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, "xgb_not_b", info_columns, features)