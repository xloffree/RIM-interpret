import sklearn
import pandas as pd
import sys
sys.path.append("..")
from RIM_interpret.RIM_interpret import get_pfi, get_shap, get_lime, get_inter
from sklearn import datasets
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestRegressor







#Import example dataset and convert to pandas df
data = datasets.load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

X=df.drop("target", axis=1)
y=df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


#Train elastic net regression model
en_model = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
fit_en = en_model.fit(X_train, y_train)
#Create dataframes to test
pfi_df_en = get_pfi(fit=fit_en, X_test=X_test, y_test=y_test)
shap_df_en = get_shap(fit=fit_en, X_test=X_test, model_type="Linear")
lime_df_en = get_lime(fit=fit_en, X_train=X_train, X_test=X_test)
inter_df_en = get_inter(fit=fit_en, X_train=X_train, X_test=X_test, y_test=y_test, model_type="Linear")

#Train random forest model
rf_mdl = RandomForestRegressor(n_estimators=100,
                                  random_state=0)
fit_rf = rf_mdl.fit(X_train, y_train)
#Create dataframes to test
pfi_df_rf = get_pfi(fit=fit_rf, X_test=X_test, y_test=y_test)
shap_df_rf = get_shap(fit=fit_rf, X_test=X_test, model_type="Tree")
lime_df_rf = get_lime(fit=fit_rf, X_train=X_train, X_test=X_test)
inter_df_rf = get_inter(fit=fit_rf, X_train=X_train, X_test=X_test, y_test=y_test, model_type="Tree")



def test_pfi_cols_num_en():
    assert len(pfi_df_en.columns) == 1

def test_shap_cols_num_en():
    assert len(shap_df_en.columns) == 1

def test_lime_cols_num_en():
    assert len(lime_df_en.columns) == 1

def test_inter_cols_num_en():
    assert len(inter_df_en.columns) == 4

def test_inter_rim_mean_en():
    assert mean(inter_df_en['RIM']) <= 0.0001

def test_inter_rim_std_en():
    assert abs(inter_df_en['RIM'].std() - 1) <= 0.1

def test_pfi_inter_match_en():
    pfi_df_sort = pfi_df_en.sort_index()
    inter_df_sort = inter_df_en.sort_index()
    assert pfi_df_sort['PFI'].equals(inter_df_sort['PFI'])

def test_shap_inter_match_en():
    shap_df_sort = shap_df_en.sort_index()
    inter_df_sort = inter_df_en.sort_index()
    assert shap_df_sort['SHAP'].equals(inter_df_sort['SHAP'])

def test_lime_inter_match_en():
    lime_df_sort = lime_df_en.sort_index()
    inter_df_sort = inter_df_en.sort_index()
    assert lime_df_sort['LIME'].equals(inter_df_sort['LIME'])


def test_pfi_cols_num_rf():
    assert len(pfi_df_rf.columns) == 1

def test_shap_cols_num_rf():
    assert len(shap_df_rf.columns) == 1

def test_lime_cols_num_rf():
    assert len(lime_df_rf.columns) == 1

def test_inter_cols_num_rf():
    assert len(inter_df_rf.columns) == 4

def test_inter_rim_mean_rf():
    assert mean(inter_df_rf['RIM']) <= 0.0001

def test_inter_rim_std_rf():
    assert abs(inter_df_rf['RIM'].std() - 1) <= 0.1

def test_pfi_inter_match_rf():
    pfi_df_sort = pfi_df_rf.sort_index()
    inter_df_sort = inter_df_rf.sort_index()
    assert pfi_df_sort['PFI'].equals(inter_df_sort['PFI'])

def test_shap_inter_match_rf():
    shap_df_sort = shap_df_rf.sort_index()
    inter_df_sort = inter_df_rf.sort_index()
    assert shap_df_sort['SHAP'].equals(inter_df_sort['SHAP'])

def test_lime_inter_match_rf():
    lime_df_sort = lime_df_rf.sort_index()
    inter_df_sort = inter_df_rf.sort_index()
    assert lime_df_sort['LIME'].equals(inter_df_sort['LIME'])


#print(pfi_df.index)
#print(pfi_df)
#print(pfi_df.loc["worst perimeter", "PFI"])