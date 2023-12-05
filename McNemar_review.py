########用外部测试集做
col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

col_cat = ['diabetes', 'pneum', 'heart', 'sex',
       'smoke', 'drink','age_cat']
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import pickle
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
#导入测试集
#导入测试集
# Import test set
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv("y_test.csv")

# Separate continuous and categorical variables
X_test_cont = X_test[col_xcg]
X_test_cat = X_test[col_cat]

# Load the saved scaler
with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Standardize only continuous variables
X_test_cont = scaler_loaded.transform(X_test_cont)

# Transform the array back to dataframe and assign the column names
X_test_cont = pd.DataFrame(X_test_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_test = pd.concat([X_test_cont, X_test_cat], axis=1)

##导入xgboost的xcg+cat变量选择模型
Pkl_Filename = "save/SR1_xgb_full_model.pkl"  
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(xgboost_final_xcg, file)
#导入模型
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)

pred_full_xg = xgboost_final_xcg.predict(X_test)


###############单一xcg输入的xgboost
import pickle
Pkl_Filename = "save/SR1_xgb_xcg_model.pkl"  
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(xgboost_final_xcg, file)
#导入模型
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)
xgboost_final_xcg.get_params()
#导入测试集
X_test=pd.read_csv('X_test.csv')[col_xcg]

y_test = pd.read_csv('y_test.csv')
X_test = scaler_loaded.transform(X_test)
X_test = pd.DataFrame(X_test, columns=col_xcg).reset_index(drop=True)
pred_xcg_xg = xgboost_final_xcg.predict(X_test)

#####全变量xgboost
import pickle
Pkl_Filename = "save/SR1_xgb_full_all_model.pkl"  
#导入模型
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)
col_xcg = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC',
       'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT',
       'MPV', 'PDW', 'P_LCR', 'IG', 'IG_p', 'NEUT', 'NEUT_p', 'NLR', 'LMR',
       'PWR', 'PNR', 'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p',
       'EOS_p', 'LYM_p']
col_cat = ['diabetes', 'pneum', 'heart', 'sex', 'smoke', 'drink', 'smo_dri','age_cat']

#导入测试集
# Import test set
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv("y_test.csv")

# Separate continuous and categorical variables
X_test_cont = X_test[col_xcg]
X_test_cat = X_test[col_cat]

# Load the saved scaler
with open('robust_scaler_all.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Standardize only continuous variables
X_test_cont = scaler_loaded.transform(X_test_cont)

# Transform the array back to dataframe and assign the column names
X_test_cont = pd.DataFrame(X_test_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_test = pd.concat([X_test_cont, X_test_cat], axis=1)

pred_all_xg = xgboost_final_xcg.predict(X_test)







######################
# 获取预测结果
y_test = pd.read_csv("y_test.csv")
Y = y_test  
Y['model_all_xg'] = pred_all_xg
Y['model_full_xg'] = pred_full_xg 
Y['model_xcg_xg'] = pred_xcg_xg

import itertools
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from statsmodels.stats.multitest import multipletests

def compute_mcnemar_test(pred1, pred2, alpha=0.05):
    ct = DataFrame(confusion_matrix(pred1, pred2))
    p_value = mcnemar(ct, exact=True).pvalue
    statistic = mcnemar(ct).statistic

    return p_value, statistic

models = {'model_all_xg': pred_all_xg, 'model_full_xg': pred_full_xg, 'model_xcg_xg': pred_xcg_xg}
model_comparisons = list(itertools.combinations(models.keys(), 2))
p_values = []
statistics = []

for model1, model2 in model_comparisons:
    p_value, statistic = compute_mcnemar_test(models[model1], models[model2])
    p_values.append(p_value)
    statistics.append(statistic)

# Apply Benjamini-Hochberg correction
alpha = 0.05
reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

# Create DataFrame and save to csv
results = pd.DataFrame({
    'Model Comparison': model_comparisons,
    'p-value': p_values,
    'corrected p-value': corrected_p_values,
    'reject H0': reject,
    'statistics': statistics
})

print(results)
results.to_csv('results_SR.csv', index=False)


########用val验证集来做
col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

col_cat = ['diabetes', 'pneum', 'heart', 'sex',
       'smoke', 'drink','age_cat']
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import pickle
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
#导入验证集
# Import test set
X_test = pd.read_csv('X_val.csv')
y_test = pd.read_csv("y_val.csv")

# Separate continuous and categorical variables
X_test_cont = X_test[col_xcg]
X_test_cat = X_test[col_cat]

# Load the saved scaler
with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Standardize only continuous variables
X_test_cont = scaler_loaded.transform(X_test_cont)

# Transform the array back to dataframe and assign the column names
X_test_cont = pd.DataFrame(X_test_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_test = pd.concat([X_test_cont, X_test_cat], axis=1)

##导入xgboost的xcg+cat变量选择模型
Pkl_Filename = "save/SR1_xgb_full_model.pkl"  
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(xgboost_final_xcg, file)
#导入模型
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)

pred_full_xg = xgboost_final_xcg.predict(X_test)


###############单一xcg输入的xgboost
import pickle
Pkl_Filename = "save/SR1_xgb_xcg_model.pkl"  
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(xgboost_final_xcg, file)
#导入模型
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)
xgboost_final_xcg.get_params()
#导入测试集
X_test=pd.read_csv('X_val.csv')[col_xcg]

y_test = pd.read_csv('y_val.csv')
X_test = scaler_loaded.transform(X_test)
X_test = pd.DataFrame(X_test, columns=col_xcg).reset_index(drop=True)
pred_xcg_xg = xgboost_final_xcg.predict(X_test)

#####全变量xgboost
import pickle
Pkl_Filename = "save/SR1_xgb_full_all_model.pkl"  
#导入模型
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)
col_xcg = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC',
       'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT',
       'MPV', 'PDW', 'P_LCR', 'IG', 'IG_p', 'NEUT', 'NEUT_p', 'NLR', 'LMR',
       'PWR', 'PNR', 'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p',
       'EOS_p', 'LYM_p']
col_cat = ['diabetes', 'pneum', 'heart', 'sex', 'smoke', 'drink', 'smo_dri','age_cat']

#导入测试集
# Import test set
X_test = pd.read_csv('X_val.csv')
y_test = pd.read_csv("y_val.csv")

# Separate continuous and categorical variables
X_test_cont = X_test[col_xcg]
X_test_cat = X_test[col_cat]

# Load the saved scaler
with open('robust_scaler_all.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Standardize only continuous variables
X_test_cont = scaler_loaded.transform(X_test_cont)

# Transform the array back to dataframe and assign the column names
X_test_cont = pd.DataFrame(X_test_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_test = pd.concat([X_test_cont, X_test_cat], axis=1)

pred_all_xg = xgboost_final_xcg.predict(X_test)







######################
# 获取预测结果
y_test = pd.read_csv("y_val.csv")
Y = y_test  
Y['model_all_xg'] = pred_all_xg
Y['model_full_xg'] = pred_full_xg 
Y['model_xcg_xg'] = pred_xcg_xg

import itertools
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from statsmodels.stats.multitest import multipletests

def compute_mcnemar_test(pred1, pred2, alpha=0.05):
    ct = DataFrame(confusion_matrix(pred1, pred2))
    p_value = mcnemar(ct, exact=True).pvalue
    statistic = mcnemar(ct).statistic

    return p_value, statistic

models = {'model_all_xg': pred_all_xg, 'model_full_xg': pred_full_xg, 'model_xcg_xg': pred_xcg_xg}
model_comparisons = list(itertools.combinations(models.keys(), 2))
p_values = []
statistics = []

for model1, model2 in model_comparisons:
    p_value, statistic = compute_mcnemar_test(models[model1], models[model2])
    p_values.append(p_value)
    statistics.append(statistic)

# Apply Benjamini-Hochberg correction
alpha = 0.05
reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

# Create DataFrame and save to csv
results = pd.DataFrame({
    'Model Comparison': model_comparisons,
    'p-value': p_values,
    'corrected p-value': corrected_p_values,
    'reject H0': reject,
    'statistics': statistics
})

print(results)
results.to_csv('results_SR.csv', index=False)
