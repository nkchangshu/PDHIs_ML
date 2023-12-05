#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:54:12 2022

@author: changshu
"""
#Data Import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import matplotlib as mpl
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
df_combined = pd.read_csv('cohort_hypertension.csv')
df_combined.columns
# Define the age bins
bins = [59, 69, 79, np.inf]
# Define the labels for the bins
labels = [0, 1, 2]
# Use pd.cut() to create the age groups, setting `right` to be `False` to include the right bounds
df_combined['age_cat'] = pd.cut(df_combined['age'], bins=bins, labels=labels, right=False)
columns_to_round = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC', 'HGB', 'HCT',
       'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT', 'MPV', 'PDW',
       'P_LCR', 'IG', 'IG_p', 'NEUT_p', 'NEUT', 'NLR', 'LMR', 'PWR', 'PNR',
       'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p', 'EOS_p',
       'LYM_p']
# rounding to three decimal places
df_combined[columns_to_round] = df_combined[columns_to_round].round(3)
#df_combined.to_csv('df_combined.csv', index=False)
df_combined = pd.read_csv('df_combined.csv')
#The dataset is divided in a ratio of 5:2:3

from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
random_seed = 42
# Shuffle the data
df_combined = df_combined.sample(frac=1, random_state=random_seed).reset_index(drop=True)
# Separate features (X) and target variable (y)
X = df_combined.drop('diagnosis', axis=1) # 'diagnosis' is assumed to be the target variable. Replace with your actual target column
y = df_combined['diagnosis']
# Split the data into train and temporary test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
# Further split the temporary set into actual train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/7, random_state=random_seed)
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
X_train.isna().sum()
#############################################################
###################################################################
###############Performing PDHIs feature selection using SULOV
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

features = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC',
       'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT',
       'MPV', 'PDW', 'P_LCR', 'IG', 'IG_p', 'NEUT', 'NEUT_p', 'NLR', 'LMR',
       'PWR', 'PNR', 'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p',
       'EOS_p', 'LYM_p']
X_train_feature_selection = X_train[features]
########
import featurewiz as FW
wiz = FW.FeatureWiz(corr_limit=0.30, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=2)
X_train_feature_selection = wiz.fit_transform(X_train_feature_selection, y_train)

print('Percentage reduction in features = %0.1f%%' %((1-len(wiz.features)/len(X_train.columns))*100))
X_train_feature_selection.columns
col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

#heatmap for 7 col_xcg
X_train = X_train[col_xcg]
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

corr = X_train.corr()
sns.set(font_scale=1.5)
plt.figure(figsize=(20,20))
a = sns.heatmap(corr, annot=True, fmt='.2f', xticklabels=corr.columns.values, 
    yticklabels=corr.columns.values, annot_kws={"size": 30}, square=True)
a.set_xticklabels(a.get_xticklabels(), rotation=0, fontsize=25)
a.set_yticklabels(a.get_yticklabels(), rotation=45, fontsize=25)


cbar = a.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

plt.title('Correlation Heatmap', fontsize=32)
plt.show()
plt.savefig('SR_xcg_heatmap_7.pdf', dpi=300, bbox_inches='tight')
#######################################

#Performing feature selection for categorical variables

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
y_train.value_counts()

col_cat =['diabetes', 'pneum', 'heart', 'sex',
       'smoke', 'drink', 'smo_dri','age_cat']


#####################################################
from sklearn.feature_selection import chi2

df = X_train[col_cat]
from scipy.stats import chi2_contingency
import numpy as np

def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))

rows= []

for var1 in df:
  col = []
  for var2 in df :
    cramers =cramers_V(df[var1], df[var2]) # Cramer's V test
    col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
  rows.append(col)
  
cramers_results = np.array(rows)
df_chi2_matrix = pd.DataFrame(cramers_results, columns = df.columns, index =df.columns)
#
plt.figure(figsize=(20,10))

sns.heatmap(df_chi2_matrix,annot=True , cmap ='YlOrRd')
plt.title("correlation of features")


df = df.drop(['smo_dri'],axis = 1)

from scipy.stats import chi2_contingency
import numpy as np

def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))

rows= []

for var1 in df:
  col = []
  for var2 in df :
    cramers =cramers_V(df[var1], df[var2]) # Cramer's V test
    col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
  rows.append(col)
  
cramers_results = np.array(rows)
df_chi2_matrix = pd.DataFrame(cramers_results, columns = df.columns, index =df.columns)
#
plt.figure(figsize=(20,10))

sns.heatmap(df_chi2_matrix,annot=True , cmap ='YlOrRd')
plt.title("correlation of features")
plt.savefig('SR_cat_heatmap_7.pdf')

col_cat = ['diabetes', 'pneum', 'heart', 'sex',
       'smoke', 'drink','age_cat']
#################robust normalization
col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']
X_train = pd.read_csv('X_train.csv')
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train=scaler.fit_transform(X_train[col_xcg])
import pickle
# save the scaler
with open('robust_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
#########################################################
##########Hyperparameter selection for XGB-Mixed model
col_cat = ['diabetes', 'pneum', 'heart', 'sex',
       'smoke', 'drink','age_cat']
col_xcg =['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import recall_score, accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna 
col = col_cat+col_xcg
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = pd.read_csv('y_train.csv')
y_val = pd.read_csv('y_val.csv')

from sklearn.preprocessing import RobustScaler
import pickle
# Load the saved scaler
with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)
# Separate continuous and categorical variables
X_train_cont = X_train[col_xcg]
X_train_cat = X_train[col_cat]

# Standardize only continuous variables
X_train_cont = scaler_loaded.transform(X_train_cont)

# Transform the array back to dataframe and assign the column names
X_train_cont = pd.DataFrame(X_train_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_train = pd.concat([X_train_cont, X_train_cat], axis=1)


################################################
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

# Define which features are categorical for SMOTENC
# This is a boolean mask where True indicates a categorical feature
cat_mask = [col in col_cat for col in X_train.columns]

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, .1),
        'subsample': trial.suggest_float('subsample', 0.50, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.50, 1),
        'gamma': trial.suggest_int('gamma', 0, 10),
        'eta': trial.suggest_float('eta', 0.007, 0.013),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'binary:logistic',
        'lambda': trial.suggest_float('lambda', 1e-3, 5.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 5.0)
    }

    gbm = xgb.XGBClassifier(**param)

    # Create a pipeline that first applies SMOTENC and then fits the model
    pipeline = Pipeline([
        ('smote', SMOTENC(categorical_features=cat_mask,random_state=42)), 
        ('gbm', gbm)
    ])

    cv_score = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='balanced_accuracy').mean()

    return cv_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_params_ = study.best_params


########################################
from plotly.offline import plot, iplot, init_notebook_mode
fig = optuna.visualization.plot_optimization_history(study)
fig.layout.yaxis.titlefont.size = 30
fig.layout.yaxis.titlefont.size = 30
fig.layout.xaxis.titlefont.size = 30
fig.layout.yaxis.tickfont.size = 25
fig.layout.xaxis.tickfont.size = 25
fig.update_layout(font=dict(size=20))
plot(fig, filename='SR1_xg_full_plot11.html')
fig = optuna.visualization.plot_param_importances(study)
fig.layout.yaxis.titlefont.size = 30
fig.layout.yaxis.titlefont.size = 30
fig.layout.xaxis.titlefont.size = 30
fig.layout.yaxis.tickfont.size = 20
fig.layout.xaxis.tickfont.size = 25
fig.update_layout(font=dict(size=20))
plot(fig,filename='SR1_xg_full_plot22.html')
a = study.trials_dataframe()
#####################################################
from imblearn.over_sampling import SMOTENC

# Create an array indicating categorical features for SMOTENC
# If the feature is categorical, this will be True; otherwise False.
cat_features = [True if col in col_cat else False for col in X_train.columns]

# Create a SMOTENC instance
smote_nc = SMOTENC(categorical_features=cat_features, random_state=42)

# Fit and resample the data
X_train, y_train = smote_nc.fit_resample(X_train, y_train)

# Now use resampled data to fit the model
xgboost_model = XGBClassifier()
xgboost_final_xcg = xgboost_model.set_params(**best_params_).fit(X_train, y_train)

#calibration
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

# Import val set
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv("y_val.csv")

# Separate continuous and categorical variables
X_val_cont = X_val[col_xcg]
X_val_cat = X_val[col_cat]

# Load the saved scaler
with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Standardize only continuous variables
X_val_cont = scaler_loaded.transform(X_val_cont)

# Transform the array back to dataframe and assign the column names
X_val_cont = pd.DataFrame(X_val_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_val = pd.concat([X_val_cont, X_val_cat], axis=1)

##Importing XGB-Mixed model
Pkl_Filename = "save/SR1_xgb_full_model.pkl"  
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(xgboost_final_xcg, file)

with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)
from sklearn.metrics import brier_score_loss, roc_auc_score

y_test_predict_proba = xgboost_final_xcg.predict_proba(X_val)[:, 1]

from sklearn.calibration import calibration_curve
fraction_of_positives0, mean_predicted_value0 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier0 = brier_score_loss(y_val, y_test_predict_proba)
##################
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(xgboost_final_xcg, method='isotonic', cv=5, n_jobs=-1)
calibrated_clf.fit(X_train, y_train.values.ravel())

y_test_predict_proba = calibrated_clf.predict_proba(X_val)[:, 1]
fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier1 = brier_score_loss(y_val, y_test_predict_proba)

####################
clf_sigmoid = CalibratedClassifierCV(xgboost_final_xcg, cv=5, method='sigmoid', n_jobs = -1)
clf_sigmoid.fit(X_train, y_train.values.ravel())
y_test_predict_proba = clf_sigmoid.predict_proba(X_val)[:, 1]
fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier2 = brier_score_loss(y_val, y_test_predict_proba)

######################
plt.figure(dpi=300,figsize=(10, 6))
plt.rc('font',family='Times New Roman')   
plt.plot(mean_predicted_value2, fraction_of_positives2, 's-', color='orange', label='Calibrated (Platt),Brier:0.0617')
plt.plot(mean_predicted_value0, fraction_of_positives0, 's-', label='Uncalibrated, Brier:0.0576')
plt.plot(mean_predicted_value1, fraction_of_positives1, 's-', color='red', label='Calibrated (Isotonic),Brier:0.0605')
plt.plot([0, 1], [0, 1], '--', color='black')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.rcParams.update({'font.size':18})
plt.gca().legend()
plt.savefig('SR1_xg_full.pdf')
import pickle
Pkl_Filename = "save/SR_xgb_full_model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(calibrated_clf, file)

###################################################
######################Hyperparameter selection for XGB-PDHIs model.
col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import matplotlib as mpl
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import recall_score, accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna 
import os
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
X_train = pd.read_csv('X_train.csv')[col_xcg]
X_val = pd.read_csv('X_val.csv')[col_xcg]
y_train = pd.read_csv('y_train.csv')
y_val = pd.read_csv('y_val.csv')

from sklearn.preprocessing import RobustScaler
import pickle
with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)
# use it to transform your data
X_val = scaler_loaded.transform(X_val)
X_train = scaler_loaded.transform(X_train)

# transform the array back to dataframe and assign the column names
X_train = pd.DataFrame(X_train, columns=col_xcg).reset_index(drop=True)
X_val = pd.DataFrame(X_val, columns=col_xcg).reset_index(drop=True)

################################################
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': 5,#trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, .1),
        'subsample': trial.suggest_float('subsample', 0.50, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.50, 1),
        'gamma': trial.suggest_int('gamma', 0, 10),
        'eta': trial.suggest_float('eta', 0.007, 0.013),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'binary:logistic',
        'lambda': trial.suggest_float('lambda', 1e-3, 5.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 5.0)
    }

    # Create a pipeline that first applies SMOTE and then fits the model
    pipeline = imbPipeline([
        ('smote', SMOTE(random_state=42)), 
        ('gbm', xgb.XGBClassifier(**param))
    ])

    cv_score = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='balanced_accuracy').mean()

    return cv_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_params_ = study.best_params

########################################
from plotly.offline import plot, iplot, init_notebook_mode
fig = optuna.visualization.plot_optimization_history(study)
fig.layout.yaxis.titlefont.size = 30
fig.layout.yaxis.titlefont.size = 30
fig.layout.xaxis.titlefont.size = 30
fig.layout.yaxis.tickfont.size = 25
fig.layout.xaxis.tickfont.size = 25
fig.update_layout(font=dict(size=20))
plot(fig, filename='SR1_xg_xcg_plot1.html')
fig = optuna.visualization.plot_param_importances(study)
fig.layout.yaxis.titlefont.size = 30
fig.layout.yaxis.titlefont.size = 30
fig.layout.xaxis.titlefont.size = 30
fig.layout.yaxis.tickfont.size = 20
fig.layout.xaxis.tickfont.size = 25
fig.update_layout(font=dict(size=20))
plot(fig,filename='SR1_xg_xcg_plot2.html')
a = study.trials_dataframe()
#########################
#########################
###
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

xgboost_model = XGBClassifier(**best_params_)



import pickle
Pkl_Filename = "save/SR1_xgb_xcg_model.pkl"  
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(xgboost_final_xcg, file)

with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)
from sklearn.metrics import brier_score_loss, roc_auc_score

y_test_predict_proba = xgboost_final_xcg.predict_proba(X_val)[:, 1]

from sklearn.calibration import calibration_curve
fraction_of_positives0, mean_predicted_value0 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier0 = brier_score_loss(y_val, y_test_predict_proba)

##################
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(xgboost_final_xcg, method='isotonic', cv=5, n_jobs=-1)
calibrated_clf.fit(X_train, y_train.values.ravel())

y_test_predict_proba = calibrated_clf.predict_proba(X_val)[:, 1]
fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier1 = brier_score_loss(y_val, y_test_predict_proba)

####################
clf_sigmoid = CalibratedClassifierCV(xgboost_final_xcg, cv=5, method='sigmoid', n_jobs = -1)
clf_sigmoid.fit(X_train, y_train.values.ravel())
y_test_predict_proba = clf_sigmoid.predict_proba(X_val)[:, 1]
fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier2 = brier_score_loss(y_val, y_test_predict_proba)

######################
plt.figure(dpi=300,figsize=(10, 6))
plt.rc('font',family='Times New Roman')   
plt.plot(mean_predicted_value2, fraction_of_positives2, 's-', color='orange', label='Calibrated (Platt),Brier:0.0602')
plt.plot(mean_predicted_value0, fraction_of_positives0, 's-', label='Uncalibrated, Brier:0.0572')
plt.plot(mean_predicted_value1, fraction_of_positives1, 's-', color='red', label='Calibrated (Isotonic),Brier:0.0601')
plt.plot([0, 1], [0, 1], '--', color='black')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.rcParams.update({'font.size':18})
plt.gca().legend()
plt.savefig('SR1_xg_xcg.pdf')
import pickle
Pkl_Filename = "save/SR_xgb_xcg_model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(xgboost_final_xcg, file)

###################################################Hyperparameter selection for XGB-All model.



col = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC',
       'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT',
       'MPV', 'PDW', 'P_LCR', 'IG', 'IG_p', 'NEUT', 'NEUT_p', 'NLR', 'LMR',
       'PWR', 'PNR', 'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p',
       'EOS_p', 'LYM_p','diabetes', 'pneum', 'heart', 'sex', 'smoke', 'drink', 'smo_dri','age_cat']
col_xcg = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC',
       'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT',
       'MPV', 'PDW', 'P_LCR', 'IG', 'IG_p', 'NEUT', 'NEUT_p', 'NLR', 'LMR',
       'PWR', 'PNR', 'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p',
       'EOS_p', 'LYM_p']
col_cat = ['diabetes', 'pneum', 'heart', 'sex', 'smoke', 'drink', 'smo_dri','age_cat']
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')


import pickle
X_train = pd.read_csv('X_train.csv')[col]

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train=scaler.fit_transform(X_train[col_xcg])
import pickle
# save the scaler
with open('robust_scaler_all.pkl', 'wb') as file:
    pickle.dump(scaler, file)
#############################################
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = pd.read_csv('y_train.csv')
y_val = pd.read_csv('y_val.csv')



from sklearn.preprocessing import RobustScaler
import pickle
# Load the saved scaler
with open('robust_scaler_all.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)
# Separate continuous and categorical variables
X_train_cont = X_train[col_xcg]
X_train_cat = X_train[col_cat]

# Standardize only continuous variables
X_train_cont = scaler_loaded.transform(X_train_cont)

# Transform the array back to dataframe and assign the column names
X_train_cont = pd.DataFrame(X_train_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_train = pd.concat([X_train_cont, X_train_cat], axis=1)


################################################
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

cat_mask = [col in col_cat for col in X_train.columns]

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': 6, #trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, .1),
        'subsample': trial.suggest_float('subsample', 0.50, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.50, 1),
        'gamma': trial.suggest_int('gamma', 0, 10),
        'eta': trial.suggest_float('eta', 0.007, 0.013),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'binary:logistic',
        'lambda': trial.suggest_float('lambda', 1e-3, 5.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 5.0)
    }

    gbm = xgb.XGBClassifier(**param)

    # Create a pipeline that first applies SMOTENC and then fits the model
    pipeline = Pipeline([
        ('smote', SMOTENC(categorical_features=cat_mask,random_state=42)), 
        ('gbm', gbm)
    ])

    cv_score = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='balanced_accuracy').mean()

    return cv_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_params_ = study.best_params

########################################
from plotly.offline import plot, iplot, init_notebook_mode
fig = optuna.visualization.plot_optimization_history(study)
fig.layout.yaxis.titlefont.size = 30
fig.layout.yaxis.titlefont.size = 30
fig.layout.xaxis.titlefont.size = 30
fig.layout.yaxis.tickfont.size = 25
fig.layout.xaxis.tickfont.size = 25
fig.update_layout(font=dict(size=20))
plot(fig, filename='SR1_all_xg_full_plot11.html')
fig = optuna.visualization.plot_param_importances(study)
fig.layout.yaxis.titlefont.size = 30
fig.layout.yaxis.titlefont.size = 30
fig.layout.xaxis.titlefont.size = 30
fig.layout.yaxis.tickfont.size = 20
fig.layout.xaxis.tickfont.size = 25
fig.update_layout(font=dict(size=20))
plot(fig,filename='SR1_all_xg_full_plot22.html')
a = study.trials_dataframe()
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC


cat_features = [True if col in col_cat else False for col in X_train.columns]

# Create a SMOTENC instance
smote_nc = SMOTENC(categorical_features=cat_features, random_state=42)

# Fit and resample the data
X_train, y_train = smote_nc.fit_resample(X_train, y_train)

# Now use resampled data to fit the model
xgboost_model = XGBClassifier()
xgboost_final_xcg = xgboost_model.set_params(**best_params_).fit(X_train, y_train)

######

col = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC',
       'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT',
       'MPV', 'PDW', 'P_LCR', 'IG', 'IG_p', 'NEUT', 'NEUT_p', 'NLR', 'LMR',
       'PWR', 'PNR', 'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p',
       'EOS_p', 'LYM_p','diabetes', 'pneum', 'heart', 'sex', 'smoke', 'drink', 'smo_dri','age_cat']
col_xcg = ['WBC', 'NEU', 'LYM', 'MON', 'EOS', 'BAS', 'RBC',
       'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_SD', 'RDW_CV', 'PLT', 'PCT',
       'MPV', 'PDW', 'P_LCR', 'IG', 'IG_p', 'NEUT', 'NEUT_p', 'NLR', 'LMR',
       'PWR', 'PNR', 'PLR', 'SIII', 'SIRI', 'RCI', 'MON_p', 'BAS_p', 'NEU_p',
       'EOS_p', 'LYM_p']
col_cat = ['diabetes', 'pneum', 'heart', 'sex', 'smoke', 'drink', 'smo_dri','age_cat']
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import pickle
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')

# Import val set
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv("y_val.csv")

# Separate continuous and categorical variables
X_val_cont = X_val[col_xcg]
X_val_cat = X_val[col_cat]

# Load the saved scaler
with open('robust_scaler_all.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Standardize only continuous variables
X_val_cont = scaler_loaded.transform(X_val_cont)

# Transform the array back to dataframe and assign the column names
X_val_cont = pd.DataFrame(X_val_cont, columns=col_xcg).reset_index(drop=True)

# Concatenate continuous and categorical variables back into one dataframe
X_val = pd.concat([X_val_cont, X_val_cat], axis=1)


Pkl_Filename = "save/SR1_xgb_full_all_model.pkl"  
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(xgboost_final_xcg, file)

with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)



from sklearn.metrics import brier_score_loss, roc_auc_score

y_test_predict_proba = xgboost_final_xcg.predict_proba(X_val)[:, 1]

from sklearn.calibration import calibration_curve
fraction_of_positives0, mean_predicted_value0 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier0 = brier_score_loss(y_val, y_test_predict_proba)
##################
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(xgboost_final_xcg, method='isotonic', cv=5, n_jobs=-1)
calibrated_clf.fit(X_train, y_train.values.ravel())

y_test_predict_proba = calibrated_clf.predict_proba(X_val)[:, 1]
fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier1 = brier_score_loss(y_val, y_test_predict_proba)

####################
clf_sigmoid = CalibratedClassifierCV(xgboost_final_xcg, cv=5, method='sigmoid', n_jobs = -1)
clf_sigmoid.fit(X_train, y_train.values.ravel())
y_test_predict_proba = clf_sigmoid.predict_proba(X_val)[:, 1]
fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_val, y_test_predict_proba, n_bins=10)
brier2 = brier_score_loss(y_val, y_test_predict_proba)

######################
plt.figure(dpi=300,figsize=(10, 6))
plt.rc('font',family='Times New Roman')   
plt.plot(mean_predicted_value2, fraction_of_positives2, 's-', color='orange', label='Calibrated (Platt),Brier:0.0573')
plt.plot(mean_predicted_value0, fraction_of_positives0, 's-', label='Uncalibrated, Brier:0.0542')
plt.plot(mean_predicted_value1, fraction_of_positives1, 's-', color='red', label='Calibrated (Isotonic),Brier:0.0552')
plt.plot([0, 1], [0, 1], '--', color='black')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.rcParams.update({'font.size':18})
plt.gca().legend()
plt.savefig('SR1_xg_full_all.pdf')
import pickle
Pkl_Filename = "save/SR1_xgb_full_all_model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(xgboost_final_xcg, file)








