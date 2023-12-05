#首先进行数据处理
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import matplotlib as mpl
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')

df_combined = pd.read_csv('df_combined.csv')
#进行数据集的划分

from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
random_seed = 2000

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
#################整个模型初筛的过程
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
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
from sklearn.svm import SVC, LinearSVC, NuSVC
#########################XGB-PDHIs,仅包含特征选择后的PDHIs
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

X_train = pd.read_csv('X_train.csv')[col_xcg]
X_val = pd.read_csv('X_val.csv')[col_xcg]

#训练集smote
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train=scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train).reset_index(drop= True)
X_train.columns = X_val.columns
from imblearn.over_sampling import SMOTE
#
smote = SMOTE(random_state=2022)
X_train, y_train = smote.fit_resample(X_train, y_train)

#测试集不进行smote
X_val=scaler.transform(X_val)
X_val = pd.DataFrame(X_val).reset_index(drop= True)
X_val.columns = X_train.columns
#
from collections import Counter
counter = Counter(y_val)
print(counter)

#拟合model using RBT-derived and feature-selected variables

from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
seed = 0
def score_summary(names, classifiers):
    '''
    Given a list of classiers, this function calculates the accuracy, 
    ROC_AUC, Recall, Precision, F1, and PR-AUC and returns the values in a dataframe
    '''
    
    cols=["Classifier", "Bal_Accuracy", "ROC_AUC", "Recall", "Precision", "F1", "PR_AUC"]
    data_table = pd.DataFrame(columns=cols)
    
    for name, clf in zip(names, classifiers):        
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_val)
        accuracy = balanced_accuracy_score(y_val, pred)

        pred_proba = clf.predict_proba(X_val)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)        
        roc_auc = auc(fpr, tpr)
        
        pr_auc = average_precision_score(y_val, pred_proba)
        
        cm = confusion_matrix(y_val, pred) 
        
        recall = cm[1,1]/(cm[1,1] +cm[1,0])
        
        precision = cm[1,1]/(cm[1,1] +cm[0,1])
        
        f1 = 2*recall*precision/(recall + precision)

        df = pd.DataFrame([[name, accuracy*100, roc_auc, recall, precision, f1, pr_auc]], columns=cols)
        data_table = data_table.append(df)     

    return(np.round(data_table.reset_index(drop=True), 2))
names = [
    'Linear SVC',
    'Nu SVC',
    'Nearest Neighbors',
    'SVC',
    'Decision Tree',
    'Random Forest',
    'AdaBoost',
    'Gradient Boosting',
    'GaussianNB',
    'BernoulliNB',
    'Stochastic Gradient',
    "Neural Net",
    "ExtraTreesClassifier",
    'XGBbost',
    "Logistic Regression"
]

classifiers = [
    SVC(kernel='linear',probability=True, random_state=seed),
    NuSVC(probability=True, random_state=seed),
    KNeighborsClassifier(2),
    SVC(probability=True, random_state=seed),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    GaussianNB(),
    BernoulliNB(),
    SGDClassifier(early_stopping=True,loss='log'),
    MLPClassifier(random_state=seed),
    ExtraTreesClassifier(random_state = seed),
    XGBClassifier(objective= 'binary:logistic', random_state=seed),
    LogisticRegression(random_state=seed)
]

a = score_summary(names, classifiers).sort_values(by='Bal_Accuracy' , ascending = False)\
    .style.background_gradient(cmap='coolwarm')\
    .bar(subset=["Bal_Accuracy",], color='#ADD8E6')\
    .bar(subset=["ROC_AUC",], color='#FFA07A')\
    .bar(subset=["Recall"], color='#90EE90')\
    .bar(subset=["Precision"], color='#FFB6C1')\
    .bar(subset=["F1"], color='#D48EF6')\
    .bar(subset=["PR_AUC"], color='#87CEFA').to_html()

with open("primary_models_xcg.html", "w") as file:
    file.write(a)
# #############XGB-Mixed，同时包含进行变量选择后的分类变量

col_cat = ['diabetes', 'pneum', 'heart', 'sex',
       'smoke', 'drink','age_cat']
col_xcg =['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']
col = col_cat+col_xcg
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
X_train = pd.read_csv('X_train.csv')[col]
X_val = pd.read_csv('X_val.csv')[col]
y_train = pd.read_csv('y_train.csv').squeeze()
y_val = pd.read_csv('y_val.csv').squeeze()
#先处理数值型变量
df_num1 = X_train[col_xcg]
df_num1.info()
df_num1.isna().sum()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X=scaler.fit_transform(df_num1)
X

df_num_train = pd.DataFrame(X).reset_index(drop= True)
df_num_train.columns = df_num1.columns
##
#list取差集
df_ret = X_train[col_cat].reset_index(drop = True)
#df_stroke_new = pd.concat([df_stroke_numeric,df_stroke_categorical],axis = 1)
X_train = pd.concat([df_num_train,df_ret],axis = 1)
from imblearn.over_sampling import SMOTENC
# 
cat_indicies=[X_train.columns.get_loc(col) for col in col_cat]
#
smote = SMOTENC(categorical_features=cat_indicies,random_state=2022)
X_train, y_train = smote.fit_resample(X_train, y_train)
from collections import Counter
counter = Counter(y_train)
print(counter)
############测试集
df_num1 = X_val[col_xcg]
######进行和训练集一样的标准化
X1=scaler.transform(df_num1)
df_num_val = pd.DataFrame(X1).reset_index(drop= True)
df_num_val.columns = col_xcg
#list取差集
df_ret = X_val[col_cat].reset_index(drop = True)
X_val = pd.concat([df_num_val,df_ret],axis = 1)

from collections import Counter
counter = Counter(y_val)
print(counter)
##
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
seed = 0
def score_summary(names, classifiers):
    '''
    Given a list of classiers, this function calculates the accuracy, 
    ROC_AUC, Recall, Precision, F1, and PR-AUC and returns the values in a dataframe
    '''
    
    cols=["Classifier", "Bal_Accuracy", "ROC_AUC", "Recall", "Precision", "F1", "PR_AUC"]
    data_table = pd.DataFrame(columns=cols)
    
    for name, clf in zip(names, classifiers):        
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_val)
        accuracy = balanced_accuracy_score(y_val, pred)

        pred_proba = clf.predict_proba(X_val)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)        
        roc_auc = auc(fpr, tpr)
        
        pr_auc = average_precision_score(y_val, pred_proba)
        
        cm = confusion_matrix(y_val, pred) 
        
        recall = cm[1,1]/(cm[1,1] +cm[1,0])
        
        precision = cm[1,1]/(cm[1,1] +cm[0,1])
        
        f1 = 2*recall*precision/(recall + precision)

        df = pd.DataFrame([[name, accuracy*100, roc_auc, recall, precision, f1, pr_auc]], columns=cols)
        data_table = data_table.append(df)     

    return(np.round(data_table.reset_index(drop=True), 2))
names = [
    'Linear SVC',
    'Nu SVC',
    'Nearest Neighbors',
    'SVC',
    'Decision Tree',
    'Random Forest',
    'AdaBoost',
    'Gradient Boosting',
    'GaussianNB',
    'BernoulliNB',
    'Stochastic Gradient',
    "Neural Net",
    "ExtraTreesClassifier",
    'XGBbost',
    "Logistic Regression"
]

classifiers = [
    SVC(kernel='linear',probability=True, random_state=seed),
    NuSVC(probability=True, random_state=seed),
    KNeighborsClassifier(2),
    SVC(probability=True, random_state=seed),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    GaussianNB(),
    BernoulliNB(),
    SGDClassifier(early_stopping=True,loss='log'),
    MLPClassifier(random_state=seed),
    ExtraTreesClassifier(random_state = seed),
    XGBClassifier(objective= 'binary:logistic', random_state=seed),
    LogisticRegression(random_state=seed)
]

a = score_summary(names, classifiers).sort_values(by='Bal_Accuracy' , ascending = False)\
    .style.background_gradient(cmap='coolwarm')\
    .bar(subset=["Bal_Accuracy",], color='#ADD8E6')\
    .bar(subset=["ROC_AUC",], color='#FFA07A')\
    .bar(subset=["Recall"], color='#90EE90')\
    .bar(subset=["Precision"], color='#FFB6C1')\
    .bar(subset=["F1"], color='#D48EF6')\
    .bar(subset=["PR_AUC"], color='#87CEFA').to_html()

with open("primary_models_cat+xcg.html", "w") as file:
    file.write(a)

############XGB-All,所有变量
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
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
X_train = pd.read_csv('X_train.csv')[col]
X_val = pd.read_csv('X_val.csv')[col]
y_train = pd.read_csv('y_train.csv').squeeze()
y_val = pd.read_csv('y_val.csv').squeeze()
#先处理数值型变量
df_num1 = X_train[col_xcg]
df_num1.info()
df_num1.isna().sum()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X=scaler.fit_transform(df_num1)
X

df_num_train = pd.DataFrame(X).reset_index(drop= True)
df_num_train.columns = df_num1.columns
##
#list取差集
df_ret = X_train[col_cat].reset_index(drop = True)
#df_stroke_new = pd.concat([df_stroke_numeric,df_stroke_categorical],axis = 1)
X_train = pd.concat([df_num_train,df_ret],axis = 1)
from imblearn.over_sampling import SMOTENC
# 
cat_indicies=[X_train.columns.get_loc(col) for col in col_cat]
#
smote = SMOTENC(categorical_features=cat_indicies,random_state=2022)
X_train, y_train = smote.fit_resample(X_train, y_train)
from collections import Counter
counter = Counter(y_train)
print(counter)
############测试集
df_num1 = X_val[col_xcg]
######进行和训练集一样的标准化
X1=scaler.transform(df_num1)
df_num_val = pd.DataFrame(X1).reset_index(drop= True)
df_num_val.columns = col_xcg
#list取差集
df_ret = X_val[col_cat].reset_index(drop = True)
X_val = pd.concat([df_num_val,df_ret],axis = 1)

from collections import Counter
counter = Counter(y_val)
print(counter)
##
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
seed = 0
def score_summary(names, classifiers):
    '''
    Given a list of classiers, this function calculates the accuracy, 
    ROC_AUC, Recall, Precision, F1, and PR-AUC and returns the values in a dataframe
    '''
    
    cols=["Classifier", "Bal_Accuracy", "ROC_AUC", "Recall", "Precision", "F1", "PR_AUC"]
    data_table = pd.DataFrame(columns=cols)
    
    for name, clf in zip(names, classifiers):        
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_val)
        accuracy = balanced_accuracy_score(y_val, pred)

        pred_proba = clf.predict_proba(X_val)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)        
        roc_auc = auc(fpr, tpr)
        
        pr_auc = average_precision_score(y_val, pred_proba)
        
        cm = confusion_matrix(y_val, pred) 
        
        recall = cm[1,1]/(cm[1,1] +cm[1,0])
        
        precision = cm[1,1]/(cm[1,1] +cm[0,1])
        
        f1 = 2*recall*precision/(recall + precision)

        df = pd.DataFrame([[name, accuracy*100, roc_auc, recall, precision, f1, pr_auc]], columns=cols)
        data_table = data_table.append(df)     

    return(np.round(data_table.reset_index(drop=True), 2))
names = [
    'Linear SVC',
    'Nu SVC',
    'Nearest Neighbors',
    'SVC',
    'Decision Tree',
    'Random Forest',
    'AdaBoost',
    'Gradient Boosting',
    'GaussianNB',
    'BernoulliNB',
    'Stochastic Gradient',
    "Neural Net",
    "ExtraTreesClassifier",
    'XGBbost',
    "Logistic Regression"
]

classifiers = [
    SVC(kernel='linear',probability=True, random_state=seed),
    NuSVC(probability=True, random_state=seed),
    KNeighborsClassifier(2),
    SVC(probability=True, random_state=seed),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    GaussianNB(),
    BernoulliNB(),
    SGDClassifier(early_stopping=True,loss='log'),
    MLPClassifier(random_state=seed),
    ExtraTreesClassifier(random_state = seed),
    XGBClassifier(objective= 'binary:logistic', random_state=seed),
    LogisticRegression(random_state=seed)
]

a = score_summary(names, classifiers).sort_values(by='Bal_Accuracy' , ascending = False)\
    .style.background_gradient(cmap='coolwarm')\
    .bar(subset=["Bal_Accuracy",], color='#ADD8E6')\
    .bar(subset=["ROC_AUC",], color='#FFA07A')\
    .bar(subset=["Recall"], color='#90EE90')\
    .bar(subset=["Precision"], color='#FFB6C1')\
    .bar(subset=["F1"], color='#D48EF6')\
    .bar(subset=["PR_AUC"], color='#87CEFA').to_html()

with open("primary_models_all.html", "w") as file:
    file.write(a)

###########只包含分类变量
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
col_cat = ['diabetes', 'pneum', 'heart', 'sex',
       'smoke', 'drink','age_cat']
#col_xcg =['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']
col = col_cat
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
X_train = pd.read_csv('X_train.csv')[col]
X_val = pd.read_csv('X_val.csv')[col]
y_train = pd.read_csv('y_train.csv').squeeze()
y_val = pd.read_csv('y_val.csv').squeeze()
#先处理分类变量
#https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTEN.html
from imblearn.over_sampling import SMOTEN
# 
smote = SMOTEN(random_state=2022)
X_train, y_train = smote.fit_resample(X_train, y_train)
from collections import Counter
counter = Counter(y_train)
print(counter)

############测试集没有特殊处理

from collections import Counter
counter = Counter(y_val)
print(counter)
##
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
seed = 0
def score_summary(names, classifiers):
    '''
    Given a list of classiers, this function calculates the accuracy, 
    ROC_AUC, Recall, Precision, F1, and PR-AUC and returns the values in a dataframe
    '''
    
    cols=["Classifier", "Bal_Accuracy", "ROC_AUC", "Recall", "Precision", "F1", "PR_AUC"]
    data_table = pd.DataFrame(columns=cols)
    
    for name, clf in zip(names, classifiers):        
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_val)
        accuracy = balanced_accuracy_score(y_val, pred)

        pred_proba = clf.predict_proba(X_val)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)        
        roc_auc = auc(fpr, tpr)
        
        pr_auc = average_precision_score(y_val, pred_proba)
        
        cm = confusion_matrix(y_val, pred) 
        
        recall = cm[1,1]/(cm[1,1] +cm[1,0])
        
        precision = cm[1,1]/(cm[1,1] +cm[0,1])
        
        f1 = 2*recall*precision/(recall + precision)

        df = pd.DataFrame([[name, accuracy*100, roc_auc, recall, precision, f1, pr_auc]], columns=cols)
        data_table = data_table.append(df)     

    return(np.round(data_table.reset_index(drop=True), 2))
names = [
    'Linear SVC',
    'Nu SVC',
    'Nearest Neighbors',
    'SVC',
    'Decision Tree',
    'Random Forest',
    'AdaBoost',
    'Gradient Boosting',
    'GaussianNB',
    'BernoulliNB',
    'Stochastic Gradient',
    "Neural Net",
    "ExtraTreesClassifier",
    'XGBbost',
    "Logistic Regression"
]

classifiers = [
    SVC(kernel='linear',probability=True, random_state=seed),
    NuSVC(probability=True, random_state=seed),
    KNeighborsClassifier(2),
    SVC(probability=True, random_state=seed),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    GaussianNB(),
    BernoulliNB(),
    SGDClassifier(early_stopping=True,loss='log'),
    MLPClassifier(random_state=seed),
    ExtraTreesClassifier(random_state = seed),
    XGBClassifier(objective= 'binary:logistic', random_state=seed),
    LogisticRegression(random_state=seed)
]

a = score_summary(names, classifiers).sort_values(by='Bal_Accuracy' , ascending = False)\
    .style.background_gradient(cmap='coolwarm')\
    .bar(subset=["Bal_Accuracy",], color='#ADD8E6')\
    .bar(subset=["ROC_AUC",], color='#FFA07A')\
    .bar(subset=["Recall"], color='#90EE90')\
    .bar(subset=["Precision"], color='#FFB6C1')\
    .bar(subset=["F1"], color='#D48EF6')\
    .bar(subset=["PR_AUC"], color='#87CEFA').to_html()

with open("primary_models_cat.html", "w") as file:
    file.write(a)
