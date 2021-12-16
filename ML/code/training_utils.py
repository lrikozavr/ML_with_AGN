import pandas as pd
import numpy as np
import os

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.utils.fixes import loguniform
from sklearn import preprocessing
import sklearn.metrics as skmetrics

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#------------------ Functions: -------------------------

def scale_X_of_the_data(training_X, test_X):
    
    scaler = preprocessing.StandardScaler()
    training_X_transformed = scaler.fit_transform(training_X)
    test_X_transformed = scaler.transform(test_X)
    
    return training_X_transformed, test_X_transformed

#Создает датафрейм, в котором каждая колонка отдельный показатель точности
def evaluate(y_true, y_pred, y_prob_positive_class):
    
    metrics = {"f1": skmetrics.f1_score(y_true=y_true, y_pred=y_pred),
           "precision": skmetrics.precision_score(y_true=y_true, y_pred=y_pred),
           "recall": skmetrics.recall_score(y_true=y_true, y_pred=y_pred),
#            "matthews_corrcoef": skmetrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred),
           "balanced_accuracy": skmetrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)}#,
#            "average_precision": skmetrics.average_precision_score(y_true=y_true, y_score=y_prob_positive_class)}
    
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = \
                                skmetrics.confusion_matrix(y_true, y_pred, normalize=None).ravel()
    
    precision, recall, thresholds = skmetrics.precision_recall_curve(y_true, y_prob_positive_class)
    pr_curve_df = pd.DataFrame({"precision": precision, "recall": recall, 
                                "thresholds": np.append(thresholds, 9999)})
    
    metrics["pr_auc"] = skmetrics.auc(pr_curve_df["recall"], pr_curve_df["precision"])
    metrics_df = pd.DataFrame(metrics, index=[0])
    
    return metrics_df, pr_curve_df


def evaluate_on_cv(training_data, train_X, clf_for_eval, fuzzy_option, fuzzy_dist_column, fuzzy_err_column, xgb_flag=False):
    
    # performance evaluation on cross validation
    
    all_cv_metrics = []
    #Случайно формирует валидационную выборку
    shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2)#100
    
    #100 раз случайно формирует тестовую и тренеровочную выбороки с коэфом разделения 0.2
    for train_index, test_index in shuffle_split.split(train_X):
    
        k_train_X = train_X[train_index]
        k_train_Y = training_data["Y"].values[train_index]
    
        k_test_X = train_X[test_index]
        k_test_Y = training_data["Y"].values[test_index]
    
        if xgb_flag is False:
            
            if fuzzy_option == "normal":
                clf_for_eval.fit(X=k_train_X, y=k_train_Y)
            elif fuzzy_option == "fuzzy_dist":
                k_train_fuzzy_weight = training_data[fuzzy_dist_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight)
            elif fuzzy_option == "fuzzy_err":
                k_train_fuzzy_weight = training_data[fuzzy_err_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight)
            else:
                print("wrong fuzzy option")
        
        else:
            
            if fuzzy_option == "normal":
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, 
                          early_stopping_rounds=20, eval_metric="logloss", 
                          eval_set=[(k_test_X, k_test_Y)], verbose=False)
            elif fuzzy_option == "fuzzy_dist":
                k_train_fuzzy_weight = training_data[fuzzy_dist_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight,
                                 early_stopping_rounds=20, eval_metric="logloss", 
                                 eval_set=[(k_test_X, k_test_Y)], verbose=False)
                          
            elif fuzzy_option == "fuzzy_err":
                k_train_fuzzy_weight = training_data[fuzzy_err_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight,
                                 early_stopping_rounds=20, eval_metric="logloss", 
                                 eval_set=[(k_test_X, k_test_Y)], verbose=False)                   
            else:
                print("wrong fuzzy option")
    
        k_test_Y_pred = clf_for_eval.predict(k_test_X)
        k_test_Y_prob_positive_class = clf_for_eval.predict_proba(k_test_X)[:, 1] 
    
        metrics, pr_curve = evaluate(y_true=k_test_Y, y_pred=k_test_Y_pred, 
                                     y_prob_positive_class=k_test_Y_prob_positive_class)
    
        all_cv_metrics.append(metrics)


    all_cv_metrics = pd.concat(all_cv_metrics)


    metrics_mean = pd.DataFrame(all_cv_metrics.mean()).T
    metrics_std = pd.DataFrame(all_cv_metrics.std()).T
    
    return metrics_mean, metrics_std

def pred__save(output_path_predict,data,test,clf):
    
    data['AGN_probability'] = np.array(clf.predict_proba(test)[:, 1])
    data.to_csv(output_path_predict, index=False)

def predict_and_pr_curve_on_cv(training_data, train_X, clf_for_eval, 
                               fuzzy_option, fuzzy_dist_column, fuzzy_err_column, xgb_flag=False):
    
    # prediction and precision-recall curve using cross validation
    
    kfold = KFold(n_splits=5, shuffle=False)
    training_data.reset_index(drop=True, inplace=True)
    training_data["y_pred"] = 999
    training_data["y_prob_positive_class"] = 999

    i = 0
    for train_index, test_index in kfold.split(train_X):

        #print(train_index,len(train_index),len(test_index),len(train_X))
        k_train_X = train_X[train_index]
        k_train_Y = training_data["Y"].values[train_index]
    
        k_test_X = train_X[test_index]
        k_test_Y = training_data["Y"].values[test_index]
    
        if xgb_flag is False:
            
            if fuzzy_option == "normal":
                clf_for_eval.fit(X=k_train_X, y=k_train_Y)
            elif fuzzy_option == "fuzzy_dist":
                k_train_fuzzy_weight = training_data[fuzzy_dist_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight)
#                        sample_weight=training_data[fuzzy_dist_column].to_numpy().T[0])
            elif fuzzy_option == "fuzzy_err":
                k_train_fuzzy_weight = training_data[fuzzy_err_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight)
            else:
                print("wrong fuzzy option")
        
        else:
            
            if fuzzy_option == "normal":
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, 
                          early_stopping_rounds=20, eval_metric="logloss", 
                          eval_set=[(k_test_X, k_test_Y)], verbose=False)
            elif fuzzy_option == "fuzzy_dist":
                k_train_fuzzy_weight = training_data[fuzzy_dist_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight,
                                 early_stopping_rounds=20, eval_metric="logloss", 
                                 eval_set=[(k_test_X, k_test_Y)], verbose=False)
                          
            elif fuzzy_option == "fuzzy_err":
                k_train_fuzzy_weight = training_data[fuzzy_err_column].values.T[0][train_index]
                clf_for_eval.fit(X=k_train_X, y=k_train_Y, sample_weight=k_train_fuzzy_weight,
                                 early_stopping_rounds=20, eval_metric="logloss", 
                                 eval_set=[(k_test_X, k_test_Y)], verbose=False)                   
            else:
                print("wrong fuzzy option")
    
        y_pred = clf_for_eval.predict(k_test_X)
        y_prob = clf_for_eval.predict_proba(k_test_X)[:, 1] 
    
        training_data.loc[test_index, "y_pred"] = y_pred 
        training_data.loc[test_index, "y_prob_positive_class"] = y_prob 
    
        if i==0:
            #pr curve
            precision, recall, thresholds = skmetrics.precision_recall_curve(training_data["Y"].loc[test_index], 
                                                                             y_prob)
            pr_curve_df = pd.DataFrame({"precision": precision, "recall": recall, 
                                        "thresholds": np.append(thresholds, 9999)})
            i+=1
    
    return pr_curve_df
        
    
def save_results(output_path, experiment_name, fuzzy_option,
                 training_data, general_data, metrics, std, pr_curve, best_param_df, gs_results_df,
                 info_columns, features):
    
    # ----------  Create output paths:
    if not os.path.exists(os.path.join(output_path, experiment_name)):
        os.makedirs(os.path.join(output_path, experiment_name))
        
    experiment_path = os.path.join(output_path, experiment_name)
    fuzzy_option_path = os.path.join(experiment_path, fuzzy_option)

    if not os.path.exists(fuzzy_option_path):
        os.makedirs(fuzzy_option_path)
        
    metric_path = os.path.join(fuzzy_option_path, fuzzy_option+"_metrics.csv")
    std_path = os.path.join(fuzzy_option_path, fuzzy_option+"_std.csv")
    pr_curve_path = os.path.join(fuzzy_option_path, fuzzy_option+"_pr_curve.csv")
    training_predictions_path = os.path.join(fuzzy_option_path, fuzzy_option+"_training_prediction.csv")
    generalization_path = os.path.join(fuzzy_option_path, fuzzy_option+"_generalization.csv")
    
    best_param_path = os.path.join(fuzzy_option_path, fuzzy_option+"_best_params.csv")
    gs_results_path = os.path.join(fuzzy_option_path, fuzzy_option+"_gs_results.csv")
    
    # ---------- Save results:
    training_data[info_columns + features + ["Y", "y_pred", "y_prob_positive_class"]].to_csv(training_predictions_path)
    general_data[info_columns + features + ["y_pred", "y_prob_positive_class"]].to_csv(generalization_path)
    
    metrics.to_csv(metric_path)
    std.to_csv(std_path)
    pr_curve.to_csv(pr_curve_path)
    
    best_param_df.to_csv(best_param_path)
    gs_results_df.to_csv(gs_results_path)
    
def get_gs_results(clf_gs):
    #get grid search results
    
    gs_results = clf_gs.cv_results_
    keys_to_extract = ['mean_test_score', 'std_test_score', 'rank_test_score']
    gs_results_subset = {key: gs_results[key] for key in keys_to_extract}
    gs_results_df = pd.DataFrame(gs_results_subset)
    
    return gs_results_df