import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
# import sklearn.metrics as skmetrics

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import training_utils

experiment_name = "rf_balanced"
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
features = ["N2-N4", "N3-N4", "N2-N3", "Y-N4", "Z-N4", "G-I", "G-R", "I-N4"]
fuzzy_dist_column = ["fuzzy_dist"]
fuzzy_err_column = ["fuzzy_err"]
output_path = "./results"


#------------------------------------ TRAINING: --------------------------------------
train_X = training_data[features].values
general_X = general_data[features].values

for fuzzy_option in fuzzy_options:
    
    print(fuzzy_option)
    
    clf = RandomForestClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight='balanced',
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    
    metrics, std = training_utils.evaluate_on_cv(training_data, train_X, clf, 
                                                 fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    pr_curve = training_utils.predict_and_pr_curve_on_cv(training_data, train_X, clf,
                                                        fuzzy_option, fuzzy_dist_column, fuzzy_err_column)
    
    # fit to the data:
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
        
    # generalization:
    general_data["y_pred"] = clf.predict(general_X)
    general_data["y_prob_positive_class"] = clf.predict_proba(general_X)[:, 1] 
    
    training_utils.save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, pd.DataFrame(), pd.DataFrame(),
                 info_columns, features)
    
print("done.")