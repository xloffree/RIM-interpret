import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.inspection import permutation_importance
import lime
from lime import lime_tabular
import shap






def get_pfi(fit, X_test, y_test):
    '''
    get_pfi: calculates permutation feature importance (PFI) values for each predictor in the model
    fit: fitted model
    X_test: dataframe containing all predictors in the testing set
    y_test: dataframe containing all target variable values in the testing set
    returns: dataframe of mean PFI values for each predictor across 10 repetitions
    '''
    pfi = permutation_importance(fit, 
                                 X_test.values, 
                                 y_test.values,
                                 n_repeats=10,
                                 random_state=0,
                                 scoring='r2')
    
    #Store pfi values in dataframe
    pfi_df = pd.DataFrame(pfi.importances_mean)
    pfi_df = pfi_df.rename(columns={0: "PFI"})
    pfi_df = pfi_df.set_index(X_test.columns)
    return pfi_df




def get_shap(fit, X_test, model_type):
    '''
    get_shap: calculates the mean SHAP value across all instances for each predictor
    fit: fitted model
    X_test: dataframe containing all predictors in the testing set
    model_type: type of model ("Linear" or "Tree")
    returns: dataframe of mean SHAP values across all instances for each predictor
    '''
    if model_type == "Tree":
        shap_explainer = shap.TreeExplainer(fit, X_test)
        shap_vals = shap_explainer.shap_values(X_test, check_additivity=False)
    elif model_type == "Linear":
        shap_explainer = shap.LinearExplainer(fit, X_test)
        shap_vals = shap_explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_vals)
    shap_df_mean = shap_df.mean()
    shap_mean_df = shap_df_mean.to_frame()
    shap_mean_df = shap_mean_df.set_index([X_test.columns])
    shap_mean_df = shap_mean_df.rename(columns={0: "SHAP"})
    
    return shap_mean_df




def get_lime(fit, X_train, X_test):
    '''
    get_lime: calculates the mean LIME value across all instances for each predictor
    fit: fitted model
    X_train: dataframe containing all predictors in the training set
    X_test: dataframe containing all predictors in the testing set
    returns: dataframe of mean LIME values across all instances for each predictor
    '''

    explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(X_train), 
                                             mode = "regression",
                                             feature_names = X_train.columns,
                                             discretize_continuous = True,
                                                 random_state = 0)
    
    
    lime_df = pd.DataFrame(columns = X_test.columns)

    for i in range(len(X_test)):
    
        exp = explainer.explain_instance(data_row = X_test.iloc[i],
                            predict_fn = fit.predict,
                            num_features = len(X_test.columns))
    
        weight_list_sorted = sorted(exp.as_map()[1], key=lambda x: x[0])
    
        weight_list = []
        for row in weight_list_sorted:
            weight = row[1]
            weight_list.append(weight)
        
        
        lime_df.loc[len(lime_df.index)] = weight_list
    
    lime_avg_df = lime_df.mean(axis=0)
    
    lime_avg_df = lime_avg_df.to_frame()
    lime_avg_df = lime_avg_df.rename(columns={0: "LIME"})
    
    return lime_avg_df
        
        

def get_inter(fit, X_train, X_test, y_test, model_type):
    '''
    get_inter: calculates the mean PFI, SHAP, LIME, and RIM values for each predictor
    fit: fitted model
    X_train: dataframe containing all predictors in the training set
    X_test: dataframe containing all predictors in the testing set
    y_test: y_test: dataframe containing all target variable values in the testing set
    returns: dataframe of PFI, SHAP, LIME, and RIM values for each predictor
    '''
    
    
    
    lime_df = get_lime(fit, X_train, X_test)
    pfi_df = get_pfi(fit, X_test, y_test)
    shap_df = get_shap(fit, X_test, model_type)
    
    inter_df = lime_df.merge(pfi_df, left_index = True, right_index = True)
    inter_df = inter_df.merge(shap_df, left_index=True, right_index=True)
    
    
    inter_df = inter_df.drop("LIME",axis = 1).join(inter_df["LIME"])
    inter_df = inter_df.drop("PFI",axis = 1).join(inter_df["PFI"])
    inter_df = inter_df.drop("SHAP",axis = 1).join(inter_df["SHAP"])

    inter_avg_list = []
    for i in range(len(inter_df.index)):
        avg = ((abs(inter_df["LIME"][i] - inter_df["LIME"].mean())/inter_df["LIME"].std()) + 
               (inter_df["PFI"][i] - inter_df["PFI"].mean())/inter_df["PFI"].std() + 
               (abs(inter_df["SHAP"][i] - inter_df["SHAP"].mean())/inter_df["SHAP"].std()))/3
        inter_avg_list.append(avg)
    
    inter_avg_df = pd.DataFrame(inter_avg_list)
    inter_avg_df[0] = (inter_avg_df[0]-mean(inter_avg_df[0]))/np.std(inter_avg_df[0])
    inter_avg_df = inter_avg_df.set_index(inter_df.index)
    
    inter_df = inter_df.join(inter_avg_df)
    inter_df = inter_df.rename(columns={0: "RIM"})
    
    
    inter_df = inter_df.sort_values("RIM", ascending=False)
    
    return inter_df
