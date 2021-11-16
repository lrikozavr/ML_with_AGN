#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

from sklearn import preprocessing

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

import xgboost as xgb
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import training_utils

experiment_name = "dummy"
print(experiment_name)
experiment_name = "et_balanced"
print(experiment_name)
experiment_name = "et_not_balanced"
print(experiment_name)
experiment_name = "logistic_balanced"
print(experiment_name)
experiment_name = "logistic_not_balanced"
print(experiment_name)
experiment_name = "rf_balanced"
print(experiment_name)
experiment_name = "rf_not_balanced"
print(experiment_name)
experiment_name = "stacked_classifier"
print(experiment_name)
experiment_name = "svm_balanced"
print(experiment_name)
experiment_name = "svm_not_balanced"
print(experiment_name)
experiment_name = "xgb_balanced"
print(experiment_name)
experiment_name = "xgb_not_balanced"
print(experiment_name)
#------------------------------------ TRAINING: --------------------------------------

def fuzzy_1(clf,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column):
    if fuzzy_option == "normal":
        clf.fit(X=train_X, y=training_data["Y"])
    elif fuzzy_option == "fuzzy_dist":
        clf.fit(X=train_X, y=training_data["Y"],
                   sample_weight=training_data[fuzzy_dist_column].values.T[0])
    elif fuzzy_option == "fuzzy_err":
         clf.fit(X=train_X, y=training_data["Y"],
                    sample_weight=training_data[fuzzy_err_column].values.T[0])
    else:
        print("wrong fuzzy option")	
        exit()
    return clf

def fuzzy_2(clf_gs,train_X,training_part,test_X,test_part,fuzzy_option,fuzzy_dist_column,fuzzy_err_column):
    if fuzzy_option == "normal":
        
        clf_gs.fit(X=train_X, y=training_part["Y"], 
                   early_stopping_rounds=20, eval_metric="logloss", 
                   eval_set=[(test_X, test_part["Y"])], verbose=False)
        
    elif fuzzy_option == "fuzzy_dist":
        
        clf_gs.fit(X=train_X, y=training_part["Y"], 
                   sample_weight=training_part[fuzzy_dist_column].values.T[0],
                   early_stopping_rounds=20, eval_metric="logloss", 
                   eval_set=[(test_X, test_part["Y"])], verbose=False)
                          
    elif fuzzy_option == "fuzzy_err":
         clf_gs.fit(X=train_X, y=training_part["Y"], 
                   sample_weight=training_part[fuzzy_err_column].values.T[0],
                   early_stopping_rounds=20, eval_metric="logloss", 
                   eval_set=[(test_X, test_part["Y"])], verbose=False)                   
    else:
        print("wrong fuzzy option")
        exit()
    return clf_gs

def ml_volume(general_X, general_data, training_data, train_X, clf, fuzzy_option, fuzzy_dist_column, fuzzy_err_column, output_path, experiment_name, info_columns, features):
	metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    clf = fuzzy_1(clf,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)
        
    # generalization:
    general_data["y_pred"] = clf.predict(general_X)
    general_data["y_prob_positive_class"] = clf.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, pd.DataFrame(), pd.DataFrame(),
                 info_columns, features)
    print("done.")


train_X = training_data[features].values
general_X = general_data[features].values

##############################
#▓▓▓▓░░▓░░░▓░▓░░░▓░▓░░░▓░▓░░░▓
#▓░░░▓░▓░░░▓░▓▓░▓▓░▓▓░▓▓░░▓░▓░
#▓░░░▓░▓░░░▓░▓░▓░▓░▓░▓░▓░░░▓░░
#▓░░░▓░▓░░░▓░▓░░░▓░▓░░░▓░░░▓░░
#▓▓▓▓░░░▓▓▓░░▓░░░▓░▓░░░▓░░░▓░░
##############################

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = DummyClassifier(strategy="stratified")
    
    ml_volume(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)
    
############################################################
#▓▓▓▓▓░▓▓▓▓▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░░░░░░▓░░░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓░░░░░▓░░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#▓░░░░░░░▓░░░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓▓▓░░░▓░░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
############################################################
for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = ExtraTreesClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight='balanced',
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    
    ml_volume(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

##############################################################################
#▓▓▓▓▓░▓▓▓▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░░░░░░▓░░░▓▓░░▓░▓░░░▓░░░▓░░░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓░░░░░▓░░░▓░▓░▓░▓░░░▓░░░▓░░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#▓░░░░░░░▓░░░▓░░▓▓░▓░░░▓░░░▓░░░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓▓▓░░░▓░░░▓░░░▓░░▓▓▓░░░░▓░░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
##############################################################################

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = ExtraTreesClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight=None,
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    
    ml_volume(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

#####################################################################################
#▓░░░░░▓▓▓░░░▓▓▓▓░░▓▓▓░░░▓▓▓▓▓░░▓▓▓▓░░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░░░▓░░░▓░▓░░░░▓░▓░░▓░░▓░░░░░▓░░░░▓░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#▓░░░░▓░░░▓░▓░▓▓▓░░▓▓▓░░░▓▓▓░░░▓░▓▓▓░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#▓░░░░▓░░░▓░▓░▓░░▓░▓░░▓░░▓░░░░░▓░▓░░▓░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓▓░░▓▓▓░░░▓▓▓▓░░▓░░░▓░▓▓▓▓▓░░▓▓▓▓░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#####################################################################################
# scale features of the data:
train_X, general_X = training_utils.scale_X_of_the_data(training_data[features], general_data[features])

params = {"penalty": ["l1", "l2"],#, "elasticnet"],
          'C': loguniform(1e0, 1e3)}

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = LogisticRegression(class_weight='balanced',
                             solver='saga',
                             random_state=476,
                             max_iter=10000,
                             n_jobs=-1)         #ZMIEŃ
                             
    
    clf_for_eval = LogisticRegression(class_weight='balanced',
                             solver='saga',
                             random_state=476,
                             max_iter=10000,
                             n_jobs=-1) 
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1000, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=100, test_size=0.2), 
                                refit=True, verbose=0)    #ZMIEŃ
   
    # fit to the data:
    clf_gs = fuzzy_1(clf_gs,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    
    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
print("done.")

#######################################################################################################
#▓░░░░░▓▓▓░░░▓▓▓▓░░▓▓▓░░░▓▓▓▓▓░░▓▓▓▓░░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░░░▓░░░▓░▓░░░░▓░▓░░▓░░▓░░░░░▓░░░░▓░▓▓░░▓░▓░░░▓░░░▓░░░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#▓░░░░▓░░░▓░▓░▓▓▓░░▓▓▓░░░▓▓▓░░░▓░▓▓▓░░▓░▓░▓░▓░░░▓░░░▓░░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#▓░░░░▓░░░▓░▓░▓░░▓░▓░░▓░░▓░░░░░▓░▓░░▓░▓░░▓▓░▓░░░▓░░░▓░░░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓▓░░▓▓▓░░░▓▓▓▓░░▓░░░▓░▓▓▓▓▓░░▓▓▓▓░░▓░░░▓░░▓▓▓░░░░▓░░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#######################################################################################################

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = LogisticRegression(class_weight=None,
                             solver='saga',
                             random_state=476,
                             max_iter=10000,
                             n_jobs=-1)         #ZMIEŃ
                             
    
    clf_for_eval = LogisticRegression(class_weight=None,
                             solver='saga',
                             random_state=476,
                             max_iter=10000,
                             n_jobs=-1) 
    
#     clf = LogisticRegression(class_weight='balanced',
#                              solver='saga',
#                              random_state=476,
#                              max_iter=5000,
#                              n_jobs=-1, l1_ratio=0.5)         #ZMIEŃ
                             
    
#     clf_for_eval = LogisticRegression(class_weight='balanced',
#                              solver='saga',
#                              random_state=476,
#                              max_iter=5000,
#                              n_jobs=-1, l1_ratio=0.5)         #ZMIEŃ
                             
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1000, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=100, test_size=0.2),  
                                refit=True, verbose=0)    #ZMIEŃ
   
    # fit to the data:
    clf_gs = fuzzy_1(clf_gs,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    
    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
print("done.")

############################################################
#▓▓▓░░░▓▓▓▓▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░▓░░▓░░░░░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓░░░▓▓▓░░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#▓░░▓░░▓░░░░░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓░░░▓░▓░░░░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
############################################################

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = RandomForestClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight='balanced',
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    
    ml_volume(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

##############################################################################
#▓▓▓░░░▓▓▓▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░▓░░▓░░░░░▓▓░░▓░▓░░░▓░░░▓░░░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓░░░▓▓▓░░░▓░▓░▓░▓░░░▓░░░▓░░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#▓░░▓░░▓░░░░░▓░░▓▓░▓░░░▓░░░▓░░░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓░░░▓░▓░░░░░▓░░░▓░░▓▓▓░░░░▓░░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
##############################################################################

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = RandomForestClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight=None,
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    
    ml_volume(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

######################################################################################################################################
#░▓▓▓▓░▓▓▓▓▓░░░░░▓▓░░▓▓▓░░▓░░▓░▓▓▓▓▓░▓▓▓▓░░░▓▓▓░░▓░░░░░░░░▓▓░░▓▓▓▓░░▓▓▓▓░▓▓▓░▓▓▓▓▓░▓▓▓░▓▓▓▓▓░▓▓▓░░░▓░░░░░▓▓▓░░░▓▓▓▓░░▓▓▓░░░▓▓▓▓▓░▓▓▓▓░
#▓░░░░░░░▓░░░░░░▓░▓░▓░░░▓░▓░▓░░▓░░░░░▓░░░▓░▓░░░▓░▓░░░░░░░▓░▓░▓░░░░░▓░░░░░░▓░░▓░░░░░░▓░░▓░░░░░▓░░▓░░▓░░░░▓░░░▓░▓░░░░▓░▓░░▓░░▓░░░░░▓░░░▓
#░▓▓▓░░░░▓░░░░░▓▓▓▓░▓░░░░░▓▓░░░▓▓▓░░░▓░░░▓░▓░░░░░▓░░░░░░▓▓▓▓░░▓▓▓░░░▓▓▓░░░▓░░▓▓▓░░░░▓░░▓▓▓░░░▓▓▓░░░▓░░░░▓░░░▓░▓░▓▓▓░░▓▓▓░░░▓▓▓░░░▓░░░▓
#░░░░▓░░░▓░░░░▓░░░▓░▓░░░▓░▓░▓░░▓░░░░░▓░░░▓░▓░░░▓░▓░░░░░▓░░░▓░░░░░▓░░░░░▓░░▓░░▓░░░░░░▓░░▓░░░░░▓░░▓░░▓░░░░▓░░░▓░▓░▓░░▓░▓░░▓░░▓░░░░░▓░░░▓
#▓▓▓▓░░░░▓░░░▓░░░░▓░░▓▓▓░░▓░░▓░▓▓▓▓▓░▓▓▓▓░░░▓▓▓░░▓▓▓▓░▓░░░░▓░▓▓▓▓░░▓▓▓▓░░▓▓▓░▓░░░░░▓▓▓░▓▓▓▓▓░▓░░░▓░▓▓▓▓░░▓▓▓░░░▓▓▓▓░░▓░░░▓░▓▓▓▓▓░▓▓▓▓░
######################################################################################################################################


for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = LogisticRegression(class_weight=None,
                             solver='saga',
                             random_state=476,
                             max_iter=10000,
                             n_jobs=-1)         #ZMIEŃ
                             
    
    clf_for_eval = LogisticRegression(class_weight=None,
                             solver='saga',
                             random_state=476,
                             max_iter=10000,
                             n_jobs=-1) 
                             
    
    # create grid search instance: 1000, 100
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1000, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=100, test_size=0.2),  
                                refit=True, verbose=0)    #ZMIEŃ
   
    # fit to the data:
    clf_gs = fuzzy_1(clf_gs,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
print("done.")


##################################################################
#░▓▓▓▓░▓░░░▓░▓░░░▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░░░░▓░░░▓░▓▓░▓▓░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#░▓▓▓░░░▓░▓░░▓░▓░▓░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#░░░░▓░░▓░▓░░▓░░░▓░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓▓░░░░▓░░░▓░░░▓░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
##################################################################

params = {'C': loguniform(1e0, 1e3),
          'gamma': loguniform(1e-4, 1e-2)}

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = svm.SVC(gamma='scale',
                  kernel='rbf',
                  probability=True,
                  class_weight='balanced',
                  cache_size=5000,
                  random_state=476)
    
    clf_for_eval = svm.SVC(gamma='scale',
                  kernel='rbf',
                  probability=True,
                  class_weight='balanced',
                  cache_size=5000,
                  random_state=476)
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1000, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=100, test_size=0.2),   
                                refit=True, verbose=0)    #ZMIEŃ
   
    # fit to the data:
    clf_gs = fuzzy_1(clf_gs,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    
    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
print("done.")


####################################################################################
#░▓▓▓▓░▓░░░▓░▓░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#▓░░░░░▓░░░▓░▓▓░▓▓░▓▓░░▓░▓░░░▓░░░▓░░░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#░▓▓▓░░░▓░▓░░▓░▓░▓░▓░▓░▓░▓░░░▓░░░▓░░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#░░░░▓░░▓░▓░░▓░░░▓░▓░░▓▓░▓░░░▓░░░▓░░░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓▓▓▓░░░░▓░░░▓░░░▓░▓░░░▓░░▓▓▓░░░░▓░░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
####################################################################################


for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = svm.SVC(gamma='scale',
                  kernel='rbf',
                  probability=True,
                  class_weight=None,
                  cache_size=5000,
                  random_state=476)
    
    clf_for_eval = svm.SVC(gamma='scale',
                  kernel='rbf',
                  probability=True,
                  class_weight=None,
                  cache_size=5000,
                  random_state=476)
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1000, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=100, test_size=0.2),   
                                refit=True, verbose=0)    #ZMIEŃ
   
    # fit to the data:
    clf_gs = fuzzy_1(clf_gs,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    
    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
print("done.")

##################################################################
#▓░░░▓░░▓▓▓▓░░▓▓▓░░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#░▓░▓░░▓░░░░▓░▓░░▓░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#░░▓░░░▓░▓▓▓░░▓▓▓░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#░▓░▓░░▓░▓░░▓░▓░░▓░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓░░░▓░░▓▓▓▓░░▓▓▓░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
##################################################################

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

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = xgb.XGBClassifier(n_estimators=500, 
                            objective='binary:logistic',
                            verbosity=0,
                            scale_pos_weight=7) # jesli balanced to 7. Jeśli nie, to usunąć ten element (default 1).
    
    clf_for_eval = xgb.XGBClassifier(n_estimators=500, 
                            objective='binary:logistic',
                            verbosity=0,
                            scale_pos_weight=7) # jesli balanced to 7. Jeśli nie, to usunąć ten element (default 1). 
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1000, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=100, test_size=0.2),  
                                refit=True, verbose=0)    #ZMIEŃ
   

    # fit to the data:
    clf_gs = fuzzy_2(clf_gs,train_X,training_part,test_X,test_part,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X_all, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
                                                 xgb_flag=True)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X_all, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
                                                         xgb_flag=True)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
print("done.")


####################################################################################
#▓░░░▓░░▓▓▓▓░░▓▓▓░░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓░░░░░░▓▓░▓░░░░░░░░▓▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
#░▓░▓░░▓░░░░▓░▓░░▓░▓▓░░▓░▓░░░▓░░░▓░░░▓░░▓░░░░▓░▓░▓░░░░░░░▓░▓░▓▓░░▓░▓░░░▓░▓░░░░░▓░░░▓
#░░▓░░░▓░▓▓▓░░▓▓▓░░▓░▓░▓░▓░░░▓░░░▓░░░▓▓▓░░░░▓▓▓▓░▓░░░░░░▓▓▓▓░▓░▓░▓░▓░░░░░▓▓▓░░░▓░░░▓
#░▓░▓░░▓░▓░░▓░▓░░▓░▓░░▓▓░▓░░░▓░░░▓░░░▓░░▓░░▓░░░▓░▓░░░░░▓░░░▓░▓░░▓▓░▓░░░▓░▓░░░░░▓░░░▓
#▓░░░▓░░▓▓▓▓░░▓▓▓░░▓░░░▓░░▓▓▓░░░░▓░░░▓▓▓░░▓░░░░▓░▓▓▓▓░▓░░░░▓░▓░░░▓░░▓▓▓░░▓▓▓▓▓░▓▓▓▓░
####################################################################################


for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = xgb.XGBClassifier(n_estimators=500, 
                            objective='binary:logistic',
                            verbosity=0,
                            scale_pos_weight=1) # jesli balanced to 7. Jeśli nie, to usunąć ten element (default 1).
    
    clf_for_eval = xgb.XGBClassifier(n_estimators=500, 
                            objective='binary:logistic',
                            verbosity=0,
                            scale_pos_weight=1) # jesli balanced to 7. Jeśli nie, to usunąć ten element (default 1). 
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1000, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=100, test_size=0.2),  
                                refit=True, verbose=0)    #ZMIEŃ
   

    # fit to the data:
    clf_gs = fuzzy_2(clf_gs,train_X,training_part,test_X,test_part,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X_all, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
                                                 xgb_flag=True)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X_all, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
                                                         xgb_flag=True)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
print("done.")