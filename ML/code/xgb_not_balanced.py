
import pandas as pd
import numpy as np
import os

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
# import sklearn.metrics as skmetrics

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import training_utils

experiment_name = "xgb_not_balanced"
print(experiment_name)

#----------------- Data: --------------------
train_path = "training_sample.csv"
general_path = "mcd_general_sample_akari_nep_wide.csv"

training_data = pd.read_csv(train_path, index_col=0)
general_data = pd.read_csv(general_path, index_col=0)

training_data = training_data.sample(frac=1)
#----------------- Columns: -----------------
# columns to preserve in the output file:
info_columns = ['HSC-ID', 'AKR_ID', 'specz', 'clss']

# fuzzy options to test:
fuzzy_options = ["normal", "fuzzy_dist", "fuzzy_err"]

# Features:
features = ["N2-N4", "N3-N4", "N2-N3", "Y-N4", "Z-N4", "G-I", "G-R", "I-N4"]
fuzzy_dist_column = ["fuzzy_dist"]
fuzzy_err_column = ["fuzzy_err"]
output_path = "./results"


#------------------------------------ TRAINING: --------------------------------------

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