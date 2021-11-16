import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import training_utils

experiment_name = "logistic_balanced"
print(experiment_name)

#----------------- Data: --------------------
train_path = "training_sample.csv"
general_path = "mcd_general_sample_akari_nep_wide.csv"

training_data = pd.read_csv(train_path, index_col=0)
general_data = pd.read_csv(general_path, index_col=0)

training_data = training_data.sample(frac=1)#.reset_index(drop=True)
#----------------- Columns: -----------------
# columns to preserve in the output file:
info_columns = ['HSC-ID', 'AKR_ID', 'specz', 'clss']

# fuzzy options to test:
fuzzy_options = ["normal", "fuzzy_dist", "fuzzy_err"]

# Features:
features = ["N2-N4", "N3-N4", "N2-N3", "Y-N4", "Z-N4", 
                        "G-I", "G-R", "I-N4"]
fuzzy_dist_column = ["fuzzy_dist"]
fuzzy_err_column = ["fuzzy_err"]
output_path = "./results"


#------------------------------------ TRAINING: --------------------------------------

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
    if fuzzy_option == "normal":
        clf_gs.fit(X=train_X, y=training_data["Y"])
    elif fuzzy_option == "fuzzy_dist":
        clf_gs.fit(X=train_X, y=training_data["Y"],
                   sample_weight=training_data[fuzzy_dist_column].values.T[0])
    elif fuzzy_option == "fuzzy_err":
         clf_gs.fit(X=train_X, y=training_data["Y"],
                    sample_weight=training_data[fuzzy_err_column].values.T[0])
    else:
        print("wrong fuzzy option")
    
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