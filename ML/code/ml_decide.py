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

def ml_volume_1(general_X, general_data, training_data, train_X, clf, 
                fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
                output_path, experiment_name, info_columns, features):
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    clf = fuzzy_1(clf,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)
        
    # generalization:
    general_data["y_pred"] = clf.predict(general_X)
    general_data["y_prob_positive_class"] = clf.predict_proba(general_X)[:, 1] 
    
    #print(clf.get_params)
    #best_param_df = pd.DataFrame(np.array(clf.get_params), index=[0])

    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, pd.DataFrame(), pd.DataFrame(),
                 info_columns, features)
    print("done.")
    return clf

def ml_volume_2(general_X, general_data, training_data, train_X, clf_for_eval, clf_gs, 
                fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
                output_path, experiment_name, info_columns, features, xgb_flag=False):
    # grid search results data frame:
    gs_results_df = training_utils.get_gs_results(clf_gs)
    
    # best parameters from grid search:
    best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])
    
    # evaluation:
    clf_for_eval.set_params(**clf_gs.best_params_)
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf_for_eval, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column, xgb_flag=xgb_flag)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf_for_eval,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column, xgb_flag=xgb_flag)
    
    # best model from grid search:
    clf_best = clf_gs.best_estimator_
        
    # generalization:
    general_data["y_pred"] = clf_best.predict(general_X)
    general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features)
    
    print("done.")

def dm(fuzzy_option, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features):
    clf = DummyClassifier(strategy="stratified")
    
    ml_volume_1(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

def et(fuzzy_option, class_weight, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features):
    clf = ExtraTreesClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight=class_weight,
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    
    return ml_volume_1(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

def logreg(fuzzy_option,class_weight,params,
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features):
    clf = LogisticRegression(class_weight=class_weight,
                             solver='saga',
                             random_state=476,
                             max_iter=500,
                             n_jobs=-1)         #ZMIEŃ
                             
    
    clf_for_eval = LogisticRegression(class_weight=class_weight,
                             solver='saga',
                             random_state=476,
                             max_iter=500,
                             n_jobs=-1)
                             
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=1, test_size=0.2),  
                                refit=True, verbose=0)    #ZMIEŃ
   
    # fit to the data:
    clf_gs = fuzzy_1(clf_gs,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    ml_volume_2(general_X, general_data, training_data, train_X, clf_for_eval, clf_gs, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

def rf(fuzzy_option,class_weight, 
        general_X, general_data, training_data, train_X, 
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features):    
    clf = RandomForestClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight=class_weight,
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    
    return ml_volume_1(general_X, general_data, training_data, train_X, clf, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

def sv(fuzzy_option,class_weight,params, 
        general_X, general_data, training_data, train_X, 
        fuzzy_dist_column, fuzzy_err_column, 
        output_path, experiment_name, info_columns, features):
    clf = svm.SVC(gamma='scale',
                  #kernel='rbf',
                  kernel='linear',
                  probability=True,
                  class_weight='balanced',
                  cache_size=5000,
                  random_state=476)
    
    clf_for_eval = svm.SVC(gamma='scale',
                  #kernel='rbf',
                  kernel='linear',
                  probability=True,
                  class_weight=class_weight,
                  cache_size=5000,
                  random_state=476)
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=1, test_size=0.2),   
                                refit=True, verbose=0)    #ZMIEŃ
   
    # fit to the data:
    clf_gs = fuzzy_1(clf_gs,train_X,training_data,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    ml_volume_2(general_X, general_data, training_data, train_X, clf_for_eval, clf_gs, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features)

def xg(fuzzy_option,class_weight,params, 
        train_X, training_part, test_X, test_part,
        general_X, general_data, training_data,  
    	fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features):
    clf = xgb.XGBClassifier(n_estimators=1, 
                            objective='binary:logistic',
                            verbosity=0,
                            scale_pos_weight=class_weight) # jesli balanced to 7. Jeśli nie, to usunąć ten element (default 1).
    
    clf_for_eval = xgb.XGBClassifier(n_estimators=1, 
                            objective='binary:logistic',
                            verbosity=0,
                            scale_pos_weight=class_weight) # jesli balanced to 7. Jeśli nie, to usunąć ten element (default 1). 
    
    # create grid search instance:
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=1, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=1, test_size=0.2),  
                                refit=True, verbose=0)    #ZMIEŃ
   

    # fit to the data:
    clf_gs = fuzzy_2(clf_gs,train_X,training_part,test_X,test_part,fuzzy_option,fuzzy_dist_column,fuzzy_err_column)

    ml_volume_2(general_X, general_data, training_data, train_X, clf_for_eval, clf_gs, 
    	fuzzy_option, fuzzy_dist_column, fuzzy_err_column, 
    	output_path, experiment_name, info_columns, features, xgb_flag=True)